# multimodel_chat.py

import streamlit as st
import os
import json
from datetime import datetime
import re
import ollama
from ollama_workbench.providers.ollama_utils import (
    get_available_models, get_all_models, load_api_keys, get_token_embeddings, 
    get_ollama_client, call_ollama_endpoint, get_dynamic_model_default, 
    validate_model_exists, get_available_models_with_fallback
)
from ollama_workbench.providers.openai_utils import call_openai_api, OPENAI_MODELS
from ollama_workbench.providers.groq_utils import get_groq_client, call_groq_api, GROQ_MODELS
from ollama_workbench.providers.mistral_utils import call_mistral_api, MISTRAL_MODELS
from ollama_workbench.ui.prompts import (
    get_agent_prompt, get_metacognitive_prompt, get_voice_prompt
)
import tiktoken
from streamlit_extras.bottom_container import bottom
from ollama_workbench.knowledge.enhanced_corpus import GraphRAGCorpus, OllamaEmbedder
import logging
import time
from .tts_utils import text_to_speech, play_speech
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

SETTINGS_FILE = "multimodel-chat-settings.json"
RAGTEST_DIR = "ragtest"

class MultiModelChat:
    def __init__(self):
        """Initialize the MultiModelChat class."""
        self.load_settings()
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize necessary session state variables."""
        if "models" not in st.session_state:
            st.session_state.models = []
            
        if "multimodel_chat_history" not in st.session_state:
            st.session_state.multimodel_chat_history = []
            
        if "selected_models" not in st.session_state:
            st.session_state.selected_models = []
        
        # Ensure selected_models contains only valid models
        if st.session_state.selected_models:
            try:
                available_models = get_all_models()
                valid_models = [model for model in st.session_state.selected_models if model in available_models]
                if len(valid_models) != len(st.session_state.selected_models):
                    logger.info(f"Filtered selected_models from {len(st.session_state.selected_models)} to {len(valid_models)} valid models")
                    st.session_state.selected_models = valid_models
            except Exception as e:
                logger.error(f"Error validating selected_models: {e}")
                st.session_state.selected_models = []
            
        if "comparison_mode" not in st.session_state:
            st.session_state.comparison_mode = "side-by-side"  # Options: "side-by-side", "tabbed"
            
        if "model_settings" not in st.session_state:
            st.session_state.model_settings = {}
            
        if "shared_context" not in st.session_state:
            st.session_state.shared_context = True
            
        # Ensure multimodel_total_tokens is a dictionary
        if "multimodel_total_tokens" not in st.session_state:
            st.session_state.multimodel_total_tokens = {}
        elif not isinstance(st.session_state.multimodel_total_tokens, dict):
            st.session_state.multimodel_total_tokens = {}
        
        if "api_keys" not in st.session_state:
            st.session_state.api_keys = load_api_keys()
            
        if "response_comparisons" not in st.session_state:
            st.session_state.response_comparisons = []
    
    def load_settings(self):
        """Load saved settings from file with model validation."""
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f:
                    settings = json.load(f)
                    
                    # Get available models for validation
                    try:
                        available_models = get_all_models()
                    except Exception:
                        available_models = []
                    
                    for key, value in settings.items():
                        if value != "None":  # Only set non-None values
                            if key == "selected_models" and isinstance(value, list):
                                # Validate selected models exist
                                valid_models = [model for model in value if model in available_models]
                                if valid_models:
                                    st.session_state[key] = valid_models
                                    logger.info(f"Loaded {len(valid_models)} valid models from {len(value)} saved models")
                                else:
                                    logger.warning(f"No valid models found in saved settings, will use dynamic default")
                            else:
                                st.session_state[key] = value
                                
                logger.info(f"Multi-model chat settings loaded: {settings}")
            except Exception as e:
                logger.error(f"Error loading multi-model chat settings: {str(e)}")
    
    def save_settings(self):
        """Save current settings to file."""
        settings = {
            "selected_models": st.session_state.selected_models,
            "comparison_mode": st.session_state.comparison_mode,
            "model_settings": st.session_state.model_settings,
            "shared_context": st.session_state.shared_context
        }
        
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(settings, f)
            logger.info(f"Multi-model chat settings saved: {settings}")
            st.success("Settings saved successfully!")
        except Exception as e:
            logger.error(f"Error saving multi-model chat settings: {str(e)}")
            st.error(f"Error saving settings: {str(e)}")

    def extract_content_blocks(self, text):
        """Extract code and article blocks from text."""
        if text is None:
            return [], []
        
        # Extract code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', text)
        
        # Remove code blocks from the text
        text_without_code = re.sub(r'```[\s\S]*?```', '', text)
        
        # Extract article blocks that start with 'Title:' and continue until the next 'Title:' or the end of the text
        article_blocks = re.findall(r'^Title:.*?(?=\n^Title:|\Z)', text_without_code, re.MULTILINE | re.DOTALL)
        
        return [block.strip('`').strip() for block in code_blocks], [block.strip() for block in article_blocks]

    def count_tokens(self, text):
        """Count tokens in text."""
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def get_model_display_name(self, model_name):
        """Format model name for display."""
        if model_name.startswith("openai/"):
            return f"OpenAI - {model_name.replace('openai/', '')}"
        elif model_name.startswith("groq/"):
            return f"Groq - {model_name.replace('groq/', '')}"
        elif model_name.startswith("mistral/"):
            return f"Mistral - {model_name.replace('mistral/', '')}"
        else:
            return f"Ollama - {model_name}"
    
    def model_selector(self):
        """Create the model selection UI."""
        st.sidebar.subheader("🤖 Model Selection")
        
        # Ensure multimodel_total_tokens is a dictionary
        if not isinstance(st.session_state.multimodel_total_tokens, dict):
            st.session_state.multimodel_total_tokens = {}
            
        # Get all available models with fallback
        try:
            available_models = get_all_models()
            # Check if we got any real models
            if not available_models:
                available_models = ["No models available - Please install Ollama models first"]
                st.sidebar.error("No models found. Please install Ollama models.")
        except Exception as e:
            available_models = ["Error loading models - Check Ollama installation"]
            st.sidebar.error(f"Error loading models: {e}")
            logger.error(f"Error in model_selector: {e}")
        
        # Filter out invalid models from session state
        valid_selected_models = [
            model for model in st.session_state.selected_models 
            if model in available_models
        ]
        
        # If no valid models and models are available, select first available as default
        if not valid_selected_models and available_models:
            # Check if available_models contains placeholder messages
            if not any("No models available" in str(model) or "Error loading" in str(model) for model in available_models):
                valid_selected_models = [available_models[0]]  # Select first available model
        
        # Allow selecting multiple models
        selected_models = st.sidebar.multiselect(
            "Select Models for Comparison",
            options=available_models,
            default=valid_selected_models,
            help="Select up to 4 models to compare side by side"
        )
        
        # Limit to 4 models max
        if len(selected_models) > 4:
            st.sidebar.warning("Maximum 4 models allowed. Using the first 4 selected models.")
            selected_models = selected_models[:4]
            
        # Update session state
        st.session_state.selected_models = selected_models
        
        # Initialize model settings if needed
        for model in selected_models:
            if model not in st.session_state.model_settings:
                st.session_state.model_settings[model] = {
                    "temperature": 0.7,
                    "max_tokens": 4000,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0
                }
            
            # Initialize token counter if needed
            if model not in st.session_state.multimodel_total_tokens:
                st.session_state.multimodel_total_tokens[model] = 0
        
        # Model options
        st.sidebar.subheader("⚙️ Configuration")
        
        # Comparison mode
        st.session_state.comparison_mode = st.sidebar.radio(
            "Comparison Mode",
            ["side-by-side", "tabbed"],
            index=0 if st.session_state.comparison_mode == "side-by-side" else 1,
            format_func=lambda x: "Side by Side" if x == "side-by-side" else "Tabbed View",
            help="Side by Side: Show all models at once. Tabbed: Show one model at a time in tabs."
        )
        
        # Shared context toggle
        st.session_state.shared_context = st.sidebar.checkbox(
            "Share Context Between Models",
            value=st.session_state.shared_context,
            help="When enabled, all models will receive the same full conversation history."
        )
        
        # Save settings button
        if st.sidebar.button("💾 Save Settings"):
            self.save_settings()
            
        return selected_models
    
    def model_settings_ui(self):
        """Create UI for adjusting model settings."""
        st.sidebar.subheader("🔧 Model Settings")
        
        # Create tabs for each model's settings
        model_tabs = st.sidebar.tabs([self.get_model_display_name(model) for model in st.session_state.selected_models])
        
        for i, model in enumerate(st.session_state.selected_models):
            with model_tabs[i]:
                st.session_state.model_settings[model]["temperature"] = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.model_settings[model]["temperature"],
                    step=0.1,
                    key=f"temp_{model}"
                )
                
                max_tokens_limit = 8000 if model in GROQ_MODELS or model in MISTRAL_MODELS else 16000
                st.session_state.model_settings[model]["max_tokens"] = st.slider(
                    "Max Tokens",
                    min_value=1000,
                    max_value=max_tokens_limit,
                    value=min(st.session_state.model_settings[model]["max_tokens"], max_tokens_limit),
                    step=1000,
                    key=f"max_tokens_{model}"
                )
                
                st.session_state.model_settings[model]["presence_penalty"] = st.slider(
                    "Presence Penalty",
                    min_value=-2.0,
                    max_value=2.0,
                    value=st.session_state.model_settings[model]["presence_penalty"],
                    step=0.1,
                    key=f"presence_{model}"
                )
                
                st.session_state.model_settings[model]["frequency_penalty"] = st.slider(
                    "Frequency Penalty",
                    min_value=-2.0,
                    max_value=2.0,
                    value=st.session_state.model_settings[model]["frequency_penalty"],
                    step=0.1,
                    key=f"frequency_{model}"
                )
                
                st.info(f"Total tokens used: {st.session_state.multimodel_total_tokens.get(model, 0)}")
    
    def create_model_prompt(self, model, user_input):
        """Create a prompt for the specified model."""
        api_keys = st.session_state.api_keys
        
        # Get agent prompts
        agent_type = st.session_state.get("agent_type", "None")
        metacognitive_type = st.session_state.get("metacognitive_type", "None")
        voice_type = st.session_state.get("voice_type", "None")
        
        agent_prompt = ""
        if agent_type != "None":
            agent_prompts = get_agent_prompt()
            if agent_type in agent_prompts:
                if isinstance(agent_prompts[agent_type], dict):
                    agent_prompt = agent_prompts[agent_type]["prompt"]
                else:
                    agent_prompt = agent_prompts[agent_type]
        
        metacognitive_prompt = ""
        if metacognitive_type != "None":
            metacog_prompts = get_metacognitive_prompt()
            if metacognitive_type in metacog_prompts:
                metacognitive_prompt = metacog_prompts[metacognitive_type]
        
        voice_prompt = ""
        if voice_type != "None":
            voice_prompts = get_voice_prompt()
            if voice_type in voice_prompts:
                voice_prompt = voice_prompts[voice_type]
        
        # Get chat history context
        if st.session_state.shared_context:
            # Use full chat history when context is shared
            chat_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.multimodel_chat_history[-5:]])
        else:
            # Use only model-specific history
            model_history = [msg for msg in st.session_state.multimodel_chat_history if msg.get('model') == model or msg['role'] == 'user']
            chat_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in model_history[-5:]])
        
        # Construct corpus context if available
        corpus_context = ""
        selected_corpus = st.session_state.get("selected_corpus", "None")
        if selected_corpus != "None":
            try:
                embedder = OllamaEmbedder()
                corpus = GraphRAGCorpus.load(selected_corpus, embedder)
                results = corpus.query(user_input, n_results=3)
                
                for result in results:
                    corpus_context += f"Relevant Information (Similarity: {result['similarity']:.4f}):\n{result['content']}\n\n"
                
                corpus_context = corpus_context.strip()
            except Exception as e:
                logger.error(f"Error querying GraphRAG corpus: {str(e)}")
                corpus_context = ""
        
        # Get webpage context from query parameters
        query_params = dict(st.query_params)
        web_page_url = query_params.get('web_page_url', [''])[0]
        is_extension = query_params.get('extension', ['false'])[0].lower() == 'true'
        web_page_content = query_params.get('web_page_content', [''])[0]
        
        # Construct the initial part of the prompt
        initial_prompt = ""
        if is_extension and web_page_url:
            initial_prompt = f"You are an AI assistant working within a browser extension. You have access to the current web page's content. Please use this information to answer the user's question.\n\nWebpage URL: {web_page_url}\nWebpage Content:\n{web_page_content}\n\n"
        else:
            initial_prompt = "You are an AI assistant. How can I help you today?\n\n"
        
        # Construct the final prompt
        corpus_context_str = f"Relevant context from the knowledge base:\n{corpus_context}\n" if corpus_context else ""
        final_prompt = f"""
{initial_prompt}
{agent_prompt}
{metacognitive_prompt}
{voice_prompt}

Recent conversation history:
{chat_history}

{corpus_context_str}
Human: {user_input}
Assistant: Let me think about this thoughtfully.
"""
        
        # Count tokens in the prompt
        token_count = self.count_tokens(final_prompt)
        st.session_state.multimodel_total_tokens[model] = st.session_state.multimodel_total_tokens.get(model, 0) + token_count
        
        return final_prompt

    def generate_model_response(self, model, prompt):
        """Generate a response from the specified model."""
        api_keys = st.session_state.api_keys
        model_settings = st.session_state.model_settings.get(model, {})
        
        temperature = model_settings.get("temperature", 0.7)
        max_tokens = model_settings.get("max_tokens", 4000)
        presence_penalty = model_settings.get("presence_penalty", 0.0)
        frequency_penalty = model_settings.get("frequency_penalty", 0.0)
        
        try:
            if model in OPENAI_MODELS:
                # OpenAI API call
                response = call_openai_api(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=min(max_tokens, 16000),
                    openai_api_key=api_keys.get("openai_api_key"),
                    stream=False
                )
                return response
                
            elif model in GROQ_MODELS:
                # Groq API call
                response = call_groq_api(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=min(max_tokens, 8000),
                    groq_api_key=api_keys.get("groq_api_key")
                )
                return response.strip()
                
            elif model in MISTRAL_MODELS:
                # Mistral API call
                response = call_mistral_api(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=min(max_tokens, 8000),
                    mistral_api_key=api_keys.get("mistral_api_key")
                )
                return response.strip()
                
            else:
                # Ollama API call
                client = get_ollama_client()
                if client:
                    response = client.generate(
                        model=model,
                        prompt=prompt,
                        options={
                            "temperature": temperature,
                            "num_predict": min(max_tokens, 16000),
                            "presence_penalty": presence_penalty,
                            "frequency_penalty": frequency_penalty
                        }
                    )
                    return response['response'].strip()
                else:
                    # Fallback for older versions
                    response, _, _, _ = call_ollama_endpoint(
                        model=model,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=min(max_tokens, 16000),
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty
                    )
                    return response.strip()
                    
        except Exception as e:
            error_message = f"Error generating response from {model}: {str(e)}"
            logger.error(error_message)
            return f"Error: {error_message}"
    
    def display_chat_message(self, message, model=None):
        """Display a chat message."""
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                if message.get("content"):
                    # Extract code blocks and articles
                    code_blocks, article_blocks = self.extract_content_blocks(message["content"])
                    
                    # Add model label if this is a model response
                    if model:
                        st.markdown(f"**Model**: {self.get_model_display_name(model)}")
                    
                    # Display code blocks
                    for code_block in code_blocks:
                        st.code(code_block)
                    
                    # Display remaining content
                    non_code_parts = re.split(r'```[\s\S]*?```', message["content"])
                    for part in non_code_parts:
                        if part.strip():
                            st.markdown(part.strip())
                else:
                    st.warning("This message has no content.")
            else:
                if message.get("content"):
                    st.markdown(message["content"])
                else:
                    st.warning("This message has no content.")
    
    def calculate_response_metrics(self, responses):
        """Calculate quality metrics for model responses."""
        if len(responses) < 2:
            return None  # Need at least two responses to compare
            
        # Calculate metrics
        metrics = {}
        
        # Get embeddings for responses
        embeddings = {}
        embedding_dims = {}  # Track embedding dimensions
        
        # First pass: collect embeddings and their dimensions
        for model, response in responses.items():
            try:
                embedding = get_token_embeddings(model, response, st.session_state.api_keys)
                if embedding is not None:
                    embeddings[model] = embedding
                    embedding_dims[model] = embedding.shape[1]
                    logger.info(f"Got embeddings for {model}: shape={embedding.shape}")
                else:
                    embeddings[model] = None
                    logger.warning(f"Could not get embeddings for {model}")
            except Exception as e:
                logger.error(f"Error getting embeddings for {model}: {str(e)}")
                embeddings[model] = None
        
        # Calculate similarity matrix if we have valid embeddings
        similarity_matrix = {}
        model_list = list(responses.keys())
        
        for model1 in model_list:
            similarity_matrix[model1] = {}
            
            # Self-similarity is always 1.0
            similarity_matrix[model1][model1] = 1.0
            
            # Skip if we don't have embeddings for this model
            if model1 not in embeddings or embeddings[model1] is None:
                for model2 in model_list:
                    if model1 != model2:
                        similarity_matrix[model1][model2] = None
                continue
                
            embedding1 = embeddings[model1].reshape(1, -1)
            
            # Calculate similarities with other models
            for model2 in model_list:
                if model1 == model2:
                    continue  # Already set to 1.0
                    
                # Skip if we don't have embeddings for model2
                if model2 not in embeddings or embeddings[model2] is None:
                    similarity_matrix[model1][model2] = None
                    continue
                
                embedding2 = embeddings[model2].reshape(1, -1)
                
                # Check if dimensions match
                if embedding1.shape[1] == embedding2.shape[1]:
                    try:
                        similarity = cosine_similarity(embedding1, embedding2)[0][0]
                        similarity_matrix[model1][model2] = float(similarity)
                    except Exception as e:
                        logger.error(f"Error calculating similarity between {model1} and {model2}: {e}")
                        similarity_matrix[model1][model2] = None
                else:
                    # Dimensions don't match, can't calculate similarity directly
                    logger.warning(f"Embedding dimensions don't match: {model1}={embedding1.shape[1]}, {model2}={embedding2.shape[1]}")
                    
                    # Provide a fallback similarity measure
                    # Here we just mark it as None, but in a full implementation we could:
                    # 1. Use text-based similarity measures (e.g., n-gram overlap)
                    # 2. Project to a common dimensionality
                    # 3. Use a more sophisticated comparison method
                    similarity_matrix[model1][model2] = None
        
        # Calculate response length metrics
        length_metrics = {}
        for model, response in responses.items():
            length_metrics[model] = {
                "chars": len(response),
                "words": len(response.split()),
                "tokens": self.count_tokens(response)
            }
            
        # Calculate additional metrics (response time, hallucination estimation, etc.)
        # This would be expanded in a production system
            
        return {
            "similarity_matrix": similarity_matrix,
            "length_metrics": length_metrics
        }
        
    def render_comparison_view(self, responses):
        """Render the comparison view for multiple model responses."""
        # Calculate response metrics if we have multiple responses
        if len(responses) > 1:
            with st.expander("Response Comparison Metrics", expanded=False):
                metrics = self.calculate_response_metrics(responses)
                
                if metrics and "similarity_matrix" in metrics:
                    st.subheader("Response Similarity")
                    
                    # Convert similarity matrix to DataFrame for better display
                    sim_data = []
                    for model1, similarities in metrics["similarity_matrix"].items():
                        row = {"Model": self.get_model_display_name(model1)}
                        for model2, similarity in similarities.items():
                            if model1 != model2:  # Skip self-similarity
                                row[self.get_model_display_name(model2)] = f"{similarity:.4f}" if similarity is not None else "N/A"
                        sim_data.append(row)
                    
                    if sim_data:
                        df = pd.DataFrame(sim_data)
                        st.dataframe(df.set_index("Model"), use_container_width=True)
                    
                    st.subheader("Response Length Metrics")
                    length_data = []
                    for model, length_info in metrics["length_metrics"].items():
                        length_data.append({
                            "Model": self.get_model_display_name(model),
                            "Characters": length_info["chars"],
                            "Words": length_info["words"],
                            "Tokens": length_info["tokens"]
                        })
                    
                    if length_data:
                        df = pd.DataFrame(length_data)
                        st.dataframe(df.set_index("Model"), use_container_width=True)
        
        # Display responses in selected view mode
        if st.session_state.comparison_mode == "side-by-side":
            columns = st.columns(len(responses))
            for i, (model, response) in enumerate(responses.items()):
                with columns[i]:
                    with st.container(border=True):
                        st.subheader(self.get_model_display_name(model))
                        
                        # Extract code blocks and articles
                        code_blocks, article_blocks = self.extract_content_blocks(response)
                        
                        # Display code blocks
                        for code_block in code_blocks:
                            st.code(code_block)
                        
                        # Display remaining content
                        non_code_parts = re.split(r'```[\s\S]*?```', response)
                        for part in non_code_parts:
                            if part.strip():
                                st.markdown(part.strip())
                        
                        # Show token usage
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"Tokens: {st.session_state.multimodel_total_tokens.get(model, 0)}")
                        with col2:
                            if st.button("🔊 Speak", key=f"speak_{model}"):
                                try:
                                    speech_file = text_to_speech(response.strip())
                                    play_speech(speech_file)
                                except Exception as e:
                                    st.error(f"Error generating speech: {str(e)}")
        else:  # Tabbed view
            tabs = st.tabs([self.get_model_display_name(model) for model in responses.keys()])
            
            for i, (model, response) in enumerate(responses.items()):
                with tabs[i]:
                    with st.container(border=True):
                        # Extract code blocks and articles
                        code_blocks, article_blocks = self.extract_content_blocks(response)
                        
                        # Display code blocks
                        for code_block in code_blocks:
                            st.code(code_block)
                        
                        # Display remaining content
                        non_code_parts = re.split(r'```[\s\S]*?```', response)
                        for part in non_code_parts:
                            if part.strip():
                                st.markdown(part.strip())
                        
                        # Show token usage and TTS button
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"Tokens: {st.session_state.multimodel_total_tokens.get(model, 0)}")
                        with col2:
                            if st.button("🔊 Speak", key=f"speak_tab_{model}"):
                                try:
                                    speech_file = text_to_speech(response.strip())
                                    play_speech(speech_file)
                                except Exception as e:
                                    st.error(f"Error generating speech: {str(e)}")
    
    def multimodel_chat_interface(self):
        """Main interface for the multi-model chat."""
        # Show sidebar configuration
        selected_models = self.model_selector()
        
        # Check if we have any valid models available
        try:
            available_models = get_all_models()
            if not available_models or any("No models available" in str(model) or "Error loading" in str(model) for model in available_models):
                st.error("❌ No models available. Please install Ollama models first.")
                st.info("""
                To install models:
                1. Make sure Ollama is running
                2. Install a model: `ollama pull llama3.2`
                3. Refresh this page
                """)
                return
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            return
        
        if selected_models:
            # Filter out placeholder messages
            valid_models = [model for model in selected_models if not ("No models available" in str(model) or "Error loading" in str(model))]
            if valid_models:
                self.model_settings_ui()
            else:
                st.warning("Please select at least one valid model to begin.")
                return
        else:
            st.warning("Please select at least one model to begin.")
            return
        
        # Display chat history
        for message in st.session_state.multimodel_chat_history:
            if message["role"] == "user":
                self.display_chat_message(message)
            else:
                # For assistant messages, check if it's a comparison or single response
                if message.get("model"):
                    # Single model response
                    self.display_chat_message(message, model=message["model"])
                else:
                    # Comparison response (display is handled by render_comparison_view)
                    st.subheader("Model Comparison")
                    self.render_comparison_view(message["responses"])
        
        # Chat input
        user_input = st.chat_input("Ask a question to compare models...")
        
        if user_input:
            # Add user message to history
            st.session_state.multimodel_chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Display user message
            self.display_chat_message({"role": "user", "content": user_input})
            
            # Show processing indicator
            with st.status("Generating responses...", expanded=True) as status:
                # Generate responses from all selected models
                responses = {}
                generation_times = {}
                
                for model in selected_models:
                    st.write(f"Generating response from {self.get_model_display_name(model)}...")
                    
                    # Create prompt for the model
                    prompt = self.create_model_prompt(model, user_input)
                    
                    # Time the response generation
                    start_time = time.time()
                    
                    # Generate response
                    response = self.generate_model_response(model, prompt)
                    
                    # Calculate generation time
                    generation_time = time.time() - start_time
                    generation_times[model] = generation_time
                    
                    # Store response
                    responses[model] = response
                    
                    # Update token count for response
                    token_count = self.count_tokens(response)
                    st.session_state.multimodel_total_tokens[model] = st.session_state.multimodel_total_tokens.get(model, 0) + token_count
                    
                    st.write(f"✓ Response generated in {generation_time:.2f} seconds")
                
                status.update(label="All responses generated", state="complete")
            
            # Add comparison to history
            st.session_state.multimodel_chat_history.append({
                "role": "assistant",
                "responses": responses,
                "generation_times": generation_times
            })
            
            # Also add individual model responses to the response comparisons for later analysis
            st.session_state.response_comparisons.append({
                "prompt": user_input,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "responses": responses,
                "generation_times": generation_times
            })
            
            # Render the comparison view
            st.subheader("Model Comparison")
            
            # Add export options
            export_col1, export_col2, export_col3 = st.columns([1, 1, 3])
            with export_col1:
                if st.button("📊 Export as CSV"):
                    self.export_comparison_as_csv(user_input, responses, generation_times)
            with export_col2:
                if st.button("📄 Export as Markdown"):
                    self.export_comparison_as_markdown(user_input, responses, generation_times)
            
            # Display the responses
            self.render_comparison_view(responses)
            
            # Force a rerun to update the UI
            st.rerun()
            
    def export_comparison_as_csv(self, prompt, responses, generation_times):
        """Export the comparison results as CSV file."""
        try:
            # Create a DataFrame with the results
            data = []
            for model, response in responses.items():
                response_data = {
                    "Model": self.get_model_display_name(model),
                    "Prompt": prompt,
                    "Response": response.replace("\n", " "),  # Flatten response for CSV
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Generation Time (s)": f"{generation_times.get(model, 0):.2f}",
                    "Tokens": self.count_tokens(response),
                    "Characters": len(response),
                    "Words": len(response.split())
                }
                data.append(response_data)
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_{timestamp}.csv"
            df.to_csv(filename, index=False)
            
            # Notify the user
            st.success(f"Comparison exported to {filename}")
        except Exception as e:
            st.error(f"Error exporting comparison: {str(e)}")
            logger.error(f"Error exporting comparison as CSV: {str(e)}")
    
    def export_comparison_as_markdown(self, prompt, responses, generation_times):
        """Export the comparison results as Markdown file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_{timestamp}.md"
            
            with open(filename, "w") as f:
                # Write the header
                f.write(f"# Model Comparison - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write the prompt
                f.write(f"## Prompt\n\n{prompt}\n\n")
                
                # Write a table with metrics
                f.write("## Metrics\n\n")
                f.write("| Model | Generation Time | Tokens | Characters | Words |\n")
                f.write("|-------|----------------|--------|------------|-------|\n")
                
                for model, response in responses.items():
                    model_name = self.get_model_display_name(model)
                    gen_time = generation_times.get(model, 0)
                    tokens = self.count_tokens(response)
                    chars = len(response)
                    words = len(response.split())
                    
                    f.write(f"| {model_name} | {gen_time:.2f}s | {tokens} | {chars} | {words} |\n")
                
                f.write("\n")
                
                # Write each response
                f.write("## Responses\n\n")
                
                for model, response in responses.items():
                    model_name = self.get_model_display_name(model)
                    f.write(f"### {model_name}\n\n")
                    f.write(f"{response}\n\n")
                
                # Write footer
                f.write("---\n")
                f.write("Generated with Ollama Workbench - Multi-Model Chat\n")
            
            # Notify the user
            st.success(f"Comparison exported to {filename}")
        except Exception as e:
            st.error(f"Error exporting comparison: {str(e)}")
            logger.error(f"Error exporting comparison as Markdown: {str(e)}")

def multimodel_chat_app():
    """Entry point for the multi-model chat application."""
    st.title("Multi-Model Chat")
    st.caption("Compare responses from multiple AI models side by side")
    
    chat = MultiModelChat()
    chat.multimodel_chat_interface()

if __name__ == "__main__":
    multimodel_chat_app()