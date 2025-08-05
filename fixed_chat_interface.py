"""
Fixed Chat Interface for Ollama-Workbench

This module provides a fixed chat interface that integrates all the features
from the original chat interfaces while ensuring proper session handling,
model settings, agent features, and thinking types.
"""

import streamlit as st
import os
import json
import re
import time
import logging
from datetime import datetime

# Import session utilities for consistent session handling
from session_utils import (
    initialize_session_state, load_settings, save_settings,
    save_chat_session, load_chat_session, synchronize_model_variables,
    get_agent_prompt, get_rag_context, safe_rerun, log_message
)

# Import utilities
from ollama_utils import (
    get_available_models, get_all_models, load_api_keys, get_token_embeddings,
    get_ollama_client, call_ollama_endpoint
)

# Import API clients with error handling
try:
    from openai_utils import call_openai_api, OPENAI_MODELS
except ImportError:
    call_openai_api = lambda **kwargs: "OpenAI API not available"
    OPENAI_MODELS = []

try:
    from groq_utils import get_groq_client, call_groq_api, GROQ_MODELS
except ImportError:
    get_groq_client = lambda **kwargs: None
    call_groq_api = lambda **kwargs: "Groq API not available"
    GROQ_MODELS = []

try:
    from mistral_utils import call_mistral_api, MISTRAL_MODELS
except ImportError:
    call_mistral_api = lambda **kwargs: "Mistral API not available"
    MISTRAL_MODELS = []

# Import prompts
try:
    from prompts import get_agent_prompt, get_metacognitive_prompt, get_voice_prompt
except ImportError:
    get_agent_prompt = lambda: {"None": ""}
    get_metacognitive_prompt = lambda: {"None": ""}
    get_voice_prompt = lambda: {"None": ""}

# Import RAG components with fallbacks
try:
    from enhanced_corpus import GraphRAGCorpus, OllamaEmbedder
except ImportError:
    try:
        from fallback_modules import get_enhanced_corpus_modules
        GraphRAGCorpus, OllamaEmbedder = get_enhanced_corpus_modules()
    except ImportError:
        # Simple placeholder classes
        class GraphRAGCorpus:
            @classmethod
            def load(cls, *args, **kwargs): return cls()
            def __init__(self, *args, **kwargs): pass
            def query(self, *args, **kwargs): return []
        class OllamaEmbedder:
            def __init__(self, *args, **kwargs): pass

# Import TTS utils with fallbacks
try:
    from tts_utils import text_to_speech, play_speech
except ImportError:
    try:
        from fallback_modules import get_tts_utils
        text_to_speech, play_speech = get_tts_utils()
    except ImportError:
        # Simple placeholder functions
        text_to_speech = lambda text, **kwargs: None
        play_speech = lambda *args, **kwargs: None

# Import styles with fallbacks
try:
    from styles import apply_styles
except ImportError:
    try:
        from fallback_modules import apply_fallback_styles as apply_styles
    except ImportError:
        # Simple placeholder function
        apply_styles = lambda: ({}, "light")

# Setup logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
SETTINGS_FILE = "chat-settings.json"
SESSIONS_FOLDER = "sessions"
RAGTEST_DIR = "ragtest"

# Ensure sessions folder exists
if not os.path.exists(SESSIONS_FOLDER):
    os.makedirs(SESSIONS_FOLDER)
    logger.info(f"Created sessions folder: {SESSIONS_FOLDER}")

# Candidate Prompts for Instance-Adaptive Zero-Shot CoT Prompting
CANDIDATE_PROMPTS = [
    "Let's think step by step.",
    "Let's solve this problem by splitting it into steps.",
    "First, let's break down the problem.",
    "To approach this, we'll consider each part carefully.",
    "Let's analyze this systematically.",
    "We'll tackle this by addressing each component individually.",
    "Step one is to understand the problem fully.",
    "We'll handle this by dividing it into manageable sections."
]

def extract_code_blocks(text):
    """Extract code blocks from text."""
    if text is None:
        return [], []
    code_blocks = re.findall(r'```[\s\S]*?```', text)
    article_blocks = re.findall(r'^Title:.*?(?=\n^Title:|\Z)', text, re.MULTILINE | re.DOTALL)
    return [block.strip('`').strip() for block in code_blocks], [block.strip() for block in article_blocks]

def extract_content_blocks(text):
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

def count_tokens(text):
    """Count the number of tokens in a text string."""
    try:
        if not text:
            return 0
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        # Fallback: approximate by word count
        return len(text.split())

def display_message(message, idx):
    """Display a chat message with modern styling."""
    role = message["role"]
    content = message["content"]
    
    # Create container for message
    container = st.container()
    
    with container:
        # Create columns for avatar and content
        cols = st.columns([1, 12])
        
        with cols[0]:
            # Display avatar
            if role == "user":
                st.markdown("👤")
            else:
                st.markdown("🤖")
        
        with cols[1]:
            # Display role
            if role == "user":
                st.markdown("**You**")
            else:
                model_name = st.session_state.get("current_model", st.session_state.get("selected_model", "AI"))
                st.markdown(f"**{model_name}**")
            
            # Display content
            st.markdown(content)
            
            # Extract and display code blocks
            code_blocks, _ = extract_code_blocks(content)
            if code_blocks:
                for block in code_blocks:
                    # Check if the block has a language specified
                    match = re.match(r'^(\w+)\n', block)
                    if match:
                        language = match.group(1)
                        code = block[len(language)+1:]
                        st.code(code, language=language)
                    else:
                        st.code(block)

def construct_agent_prompt(agent_type, metacognitive_type, voice_type, selected_prompt=None):
    """Construct the agent prompt based on selected types."""
    logger.info(f"CHECKPOINT: Constructing agent prompt with {agent_type}, {metacognitive_type}, {voice_type}")
    
    # If a custom prompt is provided, use that
    if selected_prompt:
        return selected_prompt
    
    # Start with an empty prompt
    agent_prompt = ""
    
    # Add agent type prompt if selected
    if agent_type != "None":
        agent_prompts = get_agent_prompt()
        if agent_type in agent_prompts:
            agent_prompt += agent_prompts[agent_type] + "\n\n"
    
    # Add metacognitive type prompt if selected
    if metacognitive_type != "None":
        metacog_prompts = get_metacognitive_prompt()
        if metacognitive_type in metacog_prompts:
            agent_prompt += metacog_prompts[metacognitive_type] + "\n\n"
    
    # Add voice type prompt if selected
    if voice_type != "None":
        voice_prompts = get_voice_prompt()
        if voice_type in voice_prompts:
            agent_prompt += voice_prompts[voice_type] + "\n\n"
    
    return agent_prompt

def instance_adaptive_cot(prompt, model, api_keys):
    """Implement Instance-Adaptive Zero-Shot CoT Prompting."""
    # Select a random CoT prompt
    import random
    cot_prompt = random.choice(CANDIDATE_PROMPTS)
    
    # Combine with user prompt
    full_prompt = f"{prompt}\n\n{cot_prompt}"
    
    # Call appropriate API based on model
    if model.startswith("gpt"):
        return call_openai_api(prompt=full_prompt, model=model, openai_api_key=api_keys.get("openai_api_key"))
    elif model.startswith("llama") or model.startswith("mistral") or model.startswith("phi"):
        response, _, _, _ = call_ollama_endpoint(model=model, prompt=full_prompt)
        return response
    else:
        return "Model not supported for CoT prompting."

def advanced_thinking_step(prompt, model, api_keys, step):
    """Process a single thinking step and return the result."""
    # Combine prompt with thinking step
    full_prompt = f"{prompt}\n\n{step}"
    
    # Call appropriate API based on model
    if model.startswith("gpt"):
        return call_openai_api(prompt=full_prompt, model=model, openai_api_key=api_keys.get("openai_api_key"))
    elif model.startswith("llama") or model.startswith("mistral") or model.startswith("phi"):
        response, _, _, _ = call_ollama_endpoint(model=model, prompt=full_prompt)
        return response
    else:
        return "Model not supported for advanced thinking."

def get_graphrag_context(user_input, corpus_name):
    """Get context from GraphRAG corpus."""
    if corpus_name == "None":
        return ""
    
    try:
        # Load corpus
        corpus = GraphRAGCorpus.load(corpus_name)
        
        # Query corpus
        results = corpus.query(user_input)
        
        if not results:
            return ""
        
        # Format results
        context = "Relevant context:\n\n"
        for i, result in enumerate(results[:3]):  # Limit to top 3 results
            context += f"{i+1}. {result['text']}\n\n"
        
        return context
    except Exception as e:
        logger.error(f"Error getting GraphRAG context: {e}")
        return ""

def fixed_chat_interface():
    """
    Fixed chat interface with all features and proper session handling.
    
    This function combines the best features from all chat interface implementations
    while ensuring proper session handling, model settings, agent features, and
    thinking types.
    """
    # Apply modern styling
    try:
        colors, theme = apply_styles()
    except Exception as e:
        logger.error(f"Error applying styles: {e}")
    
    # Initialize session state
    initialize_session_state()
    
    # Load settings
    load_settings()
    
    # Synchronize model variables
    synchronize_model_variables()
    
    # Debug: Log current session state
    logger.info(f"Current model: {st.session_state.get('selected_model')}")
    logger.info(f"Current agent type: {st.session_state.get('agent_type')}")
    
    # Create sidebar
    with st.sidebar:
        st.title("Ollama Workbench")
        
        # Model selection
        available_models = get_available_models()
        
        # Agent settings in expandable sections
        with st.expander("🤖 Agent Settings", expanded=st.session_state.get("settings_expanded", False)):
            # Model selection
            selected_model = st.selectbox(
                "📦 Model:",
                available_models,
                index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
                key="selected_model"
            )
            
            # Agent/Prompt types
            try:
                # Agent type selection
                agent_prompts = get_agent_prompt()
                agent_types = ["None"] + list(agent_prompts.keys())
                selected_agent = st.selectbox(
                    "🧑‍🔧 Agent Type:",
                    agent_types,
                    index=agent_types.index(st.session_state.agent_type) if st.session_state.agent_type in agent_types else 0,
                    key="agent_type"
                )
                
                # Metacognitive type selection
                metacog_prompts = get_metacognitive_prompt()
                metacog_types = ["None"] + list(metacog_prompts.keys())
                selected_metacog = st.selectbox(
                    "🧠 Thinking Type:",
                    metacog_types,
                    index=metacog_types.index(st.session_state.metacognitive_type) if st.session_state.metacognitive_type in metacog_types else 0,
                    key="metacognitive_type"
                )
                
                # Voice type selection
                voice_prompts = get_voice_prompt()
                voice_types = ["None"] + list(voice_prompts.keys())
                selected_voice = st.selectbox(
                    "🗣️ Voice Type:",
                    voice_types,
                    index=voice_types.index(st.session_state.voice_type) if st.session_state.voice_type in voice_types else 0,
                    key="voice_type"
                )
            except Exception as e:
                st.warning(f"Error loading prompt types: {e}")
                logger.error(f"Error loading prompt types: {e}")
            
            # Generation settings
            st.subheader("Generation Settings")
            
            temperature = st.slider(
                "Temperature:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("temperature", 0.7),
                step=0.1,
                key="temperature"
            )
            
            max_tokens = st.number_input(
                "Max Tokens:",
                min_value=100,
                max_value=8000,
                value=st.session_state.get("max_tokens", 4000),
                step=100,
                key="max_tokens"
            )
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                presence_penalty = st.slider(
                    "Presence Penalty:",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.get("presence_penalty", 0.0),
                    step=0.1,
                    key="presence_penalty"
                )
                
                frequency_penalty = st.slider(
                    "Frequency Penalty:",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.get("frequency_penalty", 0.0),
                    step=0.1,
                    key="frequency_penalty"
                )
            
            # RAG settings
            st.subheader("Knowledge Base")
            
            try:
                # List available corpora
                import os
                corpus_dir = "corpus"
                available_corpora = ["None"]
                
                if os.path.exists(corpus_dir):
                    for item in os.listdir(corpus_dir):
                        if os.path.isdir(os.path.join(corpus_dir, item)):
                            available_corpora.append(item)
                
                selected_corpus = st.selectbox(
                    "Corpus:",
                    available_corpora,
                    index=available_corpora.index(st.session_state.selected_corpus) if st.session_state.selected_corpus in available_corpora else 0,
                    key="selected_corpus"
                )
            except Exception as e:
                st.warning(f"Error loading corpora: {e}")
                logger.error(f"Error loading corpora: {e}")
            
            # Save settings button
            if st.button("Save Settings"):
                save_settings()
                st.success("Settings saved!")
        
        # Session management
        with st.expander("💾 Session Management"):
            # Save session button
            if st.button("Save Session"):
                session_path = save_chat_session()
                if session_path:
                    st.success(f"Session saved!")
                else:
                    st.error("Failed to save session")
            
            # Load session
            try:
                import os
                session_files = []
                
                if os.path.exists(SESSIONS_FOLDER):
                    for file in os.listdir(SESSIONS_FOLDER):
                        if file.endswith(".json"):
                            session_files.append(file)
                
                if session_files:
                    selected_session = st.selectbox(
                        "Load Session:",
                        ["None"] + session_files
                    )
                    
                    if selected_session != "None" and st.button("Load"):
                        session_path = os.path.join(SESSIONS_FOLDER, selected_session)
                        if load_chat_session(session_path):
                            st.success("Session loaded!")
                            safe_rerun()
                        else:
                            st.error("Failed to load session")
                else:
                    st.info("No saved sessions found")
            except Exception as e:
                st.warning(f"Error loading sessions: {e}")
                logger.error(f"Error loading sessions: {e}")
        
        # Clear chat button
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.success("Chat cleared!")
            safe_rerun()
    
    # Main chat area
    st.title("Ollama Chat")
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        display_message(message, i)
    
    # Input area
    user_input = st.text_area("Message:", height=100, key="user_input")
    
    # Send button
    if st.button("Send") or (user_input and user_input.strip() and user_input != st.session_state.get("last_input", "")):
        if user_input and user_input.strip():
            # Store last input to prevent duplicate sends
            st.session_state.last_input = user_input
            
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Create placeholder for assistant response
            message_placeholder = st.empty()
            message_placeholder.info("Thinking...")
            
            try:
                # Get model and settings
                model = st.session_state.selected_model
                temperature = st.session_state.temperature
                max_tokens = st.session_state.max_tokens
                presence_penalty = st.session_state.presence_penalty
                frequency_penalty = st.session_state.frequency_penalty
                
                # Build prompt with agent features
                agent_prompt = construct_agent_prompt(
                    st.session_state.agent_type,
                    st.session_state.metacognitive_type,
                    st.session_state.voice_type
                )
                
                # Get RAG context if corpus is selected
                rag_context = ""
                if st.session_state.selected_corpus != "None":
                    rag_context = get_graphrag_context(user_input, st.session_state.selected_corpus)
                
                # Combine everything for the final prompt
                prompt = user_input
                if agent_prompt or rag_context:
                    # Add chat history for context if available
                    history_context = ""
                    if len(st.session_state.chat_history) > 1:  # More than just the current message
                        history_context = "Previous conversation:\n"
                        # Add last few messages for context
                        for msg in st.session_state.chat_history[-4:-1]:  # Last 3 messages before current
                            role = msg["role"].capitalize()
                            content = msg["content"]
                            history_context += f"{role}: {content}\n"
                    
                    prompt = f"{agent_prompt}\n{history_context}\n{rag_context}\nUser: {user_input}\n\nAssistant:"
                
                # Log prompt
                logger.info(f"Prompt: {prompt[:100]}...")
                
                # Generate response based on model type
                full_response = ""
                
                if model.startswith("gpt"):
                    # OpenAI model
                    api_keys = load_api_keys()
                    
                    for chunk in call_openai_api(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        openai_api_key=api_keys.get("openai_api_key"),
                        stream=True
                    ):
                        if hasattr(chunk, 'choices') and chunk.choices:
                            content = chunk.choices[0].delta.content
                            if content is not None:
                                full_response += content
                                message_placeholder.markdown(full_response + "▌")
                        
                elif model.startswith("groq/"):
                    # Groq model
                    model_name = model[5:]  # Remove "groq/" prefix
                    api_keys = load_api_keys()
                    
                    client = get_groq_client(api_keys.get("groq_api_key"))
                    full_response = call_groq_api(
                        client=client,
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    message_placeholder.markdown(full_response)
                    
                elif model.startswith("mistral/"):
                    # Mistral model
                    model_name = model[8:]  # Remove "mistral/" prefix
                    api_keys = load_api_keys()
                    
                    full_response = call_mistral_api(
                        model=model_name,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        mistral_api_key=api_keys.get("mistral_api_key")
                    )
                    message_placeholder.markdown(full_response)
                    
                else:
                    # Ollama model
                    client = get_ollama_client()
                    
                    if client:
                        # Stream response with client if available
                        for response_chunk in client.generate(
                            model=model,
                            prompt=prompt,
                            stream=True,
                            options={
                                "temperature": temperature,
                                "num_predict": max_tokens,
                                "presence_penalty": presence_penalty,
                                "frequency_penalty": frequency_penalty
                            }
                        ):
                            content = response_chunk["response"]
                            full_response += content
                            message_placeholder.markdown(full_response + "▌")
                        
                        message_placeholder.markdown(full_response)
                    else:
                        # Use non-streaming fallback
                        full_response, _, _, _ = call_ollama_endpoint(
                            model=model,
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty
                        )
                        message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
                # Update token count
                st.session_state.total_tokens = st.session_state.get("total_tokens", 0) + count_tokens(user_input) + count_tokens(full_response)
                
                # Log response
                logger.info(f"Response: {full_response[:100]}...")
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                message_placeholder.error(error_message)
                logger.error(f"Error generating response: {e}")
                
                # Add error message to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}\n\nPlease try again or select a different model."})
            
            # Force refresh to show the complete history
            safe_rerun()
    
    # Display token count
    st.caption(f"Total tokens: {st.session_state.get('total_tokens', 0)}")

if __name__ == "__main__":
    fixed_chat_interface()
