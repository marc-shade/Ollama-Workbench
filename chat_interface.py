# chat_interface.py (WITH INSTANCE-ADAPTIVE ZERO-SHOT CoT PROMPTING)

import streamlit as st
import os
import json
from datetime import datetime
import re
import ollama
from ollama_utils import (
    get_available_models, get_all_models, load_api_keys, call_ollama_endpoint, get_token_embeddings
)
from openai_utils import call_openai_api, call_openai_embeddings, OPENAI_MODELS
from groq_utils import get_groq_client, call_groq_api, get_local_embeddings, GROQ_MODELS
from prompts import (
    get_agent_prompt, get_metacognitive_prompt, get_voice_prompt, get_identity_prompt
)
import tiktoken
from streamlit_extras.bottom_container import bottom
from enhanced_corpus import GraphRAGCorpus, OllamaEmbedder
from collections import deque
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging  # Added for logging
import time  # Added for sleep in progress simulation

# Setup logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

SETTINGS_FILE = "chat-settings.json"
RAGTEST_DIR = "ragtest"

# Candidate Prompts for Instance-Adaptive Zero-Shot CoT Prompting
CANDIDATE_PROMPTS = [
    "Let's think step by step.",
    "Don't think. Just feel.",
    "Let's solve this problem by splitting it into steps.",
    "First, let's break down the problem.",
    "To approach this, we'll consider each part carefully.",
    "Let's analyze this systematically.",
    "We'll tackle this by addressing each component individually.",
    "Step one is to understand the problem fully.",
    "We'll handle this by dividing it into manageable sections."
]

class ModelMemoryHandler:
    def __init__(self, model_type):
        self.model_type = model_type
        self.episodic_memory = EpisodicMemory()

    def segment_text(self, model, text, api_keys):
        if self.model_type == "openai":
            return self.segment_text_openai(model, text, api_keys)
        elif self.model_type == "groq":
            return self.segment_text_groq(model, text, api_keys)
        else:
            return self.segment_text_ollama(model, text, api_keys)

    def segment_text_openai(self, model, text, api_keys):
        # Implementation for OpenAI models
        pass

    def segment_text_groq(self, model, text, api_keys):
        # Implementation for Groq models
        pass

    def segment_text_ollama(self, model, text, api_keys):
        # Implementation for Ollama models
        pass

    def retrieve_events(self, query_embedding):
        return self.episodic_memory.retrieve_events(query_embedding)

class EpisodicMemory:
    def __init__(self, similarity_buffer_size: int = 5, contiguity_buffer_size: int = 3):
        self.similarity_buffer = deque(maxlen=similarity_buffer_size)
        self.contiguity_buffer = deque(maxlen=contiguity_buffer_size)
        self.events = []

    def segment_text_into_events(self, model: str, text: str, threshold: float = 0.01, api_keys: dict = None):
        """Segments the text into events based on surprise and refines boundaries."""
        try:
            tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            tokenizer = tiktoken.get_encoding("gpt2")
        
        num_tokens = len(tokenizer.encode(text))
        surprise_indices = [i for i in range(1, num_tokens) if i % 50 == 0]
        embeddings = get_token_embeddings(model, text, api_keys)

        # Ensure embeddings are 2D
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if len(embeddings) > 0:
            refined_boundaries = refine_boundaries(embeddings, surprise_indices)

            self.events = []
            start = 0
            for end in refined_boundaries:
                event_text = text[start:end]
                event_embedding = np.mean(embeddings[start:end], axis=0)
                self.events.append({'text': event_text, 'embedding': event_embedding})
                start = end
            # Handle the last segment
            event_text = text[start:]
            event_embedding = np.mean(embeddings[start:], axis=0)
            self.events.append({'text': event_text, 'embedding': event_embedding})

    def retrieve_events(self, query_embedding: np.ndarray) -> list:
        """Retrieves relevant events based on the query embedding."""
        retrieved_events = []
        if len(self.events) > 0:
            query_embedding = query_embedding.reshape(1, -1)
            event_embeddings = np.array([event['embedding'] for event in self.events])

            if event_embeddings.ndim == 1:
                event_embeddings = event_embeddings.reshape(1, -1)

            similarity_scores = cosine_similarity(query_embedding, event_embeddings)[0]
            top_event_indices = np.argsort(similarity_scores)[-len(self.similarity_buffer):][::-1]
            for index in top_event_indices:
                event = self.events[index]
                retrieved_events.append(event)
                self.similarity_buffer.append(event)
                # Add neighboring events to the contiguity buffer
                for i in range(index - 1, index + 2):
                    if 0 <= i < len(self.events) and i != index:
                        self.contiguity_buffer.append(self.events[i])
            return retrieved_events + list(self.contiguity_buffer)
        return retrieved_events

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
            for key, value in settings.items():
                if value != "None":  # Only set non-None values
                    st.session_state[key] = value
    logger.info(f"Settings loaded: {st.session_state}")

def save_settings():
    settings = {
        "selected_model": st.session_state.selected_model,
        "agent_type": st.session_state.agent_type,
        "metacognitive_type": st.session_state.metacognitive_type,
        "voice_type": st.session_state.voice_type,
        "selected_corpus": st.session_state.selected_corpus,
        "temperature_slider_chat": st.session_state.temperature_slider_chat,
        "max_tokens_slider_chat": min(st.session_state.max_tokens_slider_chat, 8000),
        "presence_penalty_slider_chat": st.session_state.presence_penalty_slider_chat,
        "frequency_penalty_slider_chat": st.session_state.frequency_penalty_slider_chat,
        "episodic_memory_enabled": st.session_state.episodic_memory_enabled,
        "advanced_thinking_enabled": st.session_state.advanced_thinking_enabled,  # Ensure this line is included
        "thinking_steps": st.session_state.thinking_steps,  # Added to save thinking steps
        "instance_adaptive_cot_enabled": st.session_state.instance_adaptive_cot_enabled,
        "cot_strategy": st.session_state.cot_strategy,
        "cot_threshold": st.session_state.cot_threshold,
        "cot_top_n": st.session_state.cot_top_n
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)
    logger.info(f"Settings saved: {settings}")
    st.success("Settings saved successfully!")

def ai_assisted_prompt_writing():
    st.markdown("""
    <style>
    .stModal > div[data-testid="stHorizontalBlock"]:first-child {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader("AI Prompt Writer")
    
    if st.button("X", key="close_modal", help="Cancel assisted prompt writing."):
        st.session_state.show_prompt_modal = False
        st.rerun()
    
    user_need = st.text_input("What do you need help with?")
    if user_need:
        prompt_suggestion = generate_prompt_suggestion(user_need)
        if prompt_suggestion:
            st.write("Suggested prompt:")
            edited_prompt = st.text_area("Edit the prompt before using it:", value=prompt_suggestion)
            if st.button("Use this prompt"):
                st.session_state.chat_input = edited_prompt
                st.session_state.show_prompt_modal = False
                st.rerun()
        else:
            st.warning("Unable to generate a prompt suggestion. Please try again or select a different model.")

def generate_prompt_suggestion(user_need):
    api_keys = load_api_keys()
    model = st.session_state.selected_model
    prompt = f"Your task is to improve the user's prompt and send it to another AI agent to process. Do not respond directly to the user's response, assume whatever they give you is a prompt that needs to be improved for maximum efficiency and effectiveness. Create a prompt that will give the user the best results. Create a detailed and effective improved prompt for an AI assistant based on this user need: {user_need}"

    try:
        if model in OPENAI_MODELS:
            response = call_openai_api(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=st.session_state.temperature_slider_chat,
                max_tokens=min(st.session_state.max_tokens_slider_chat, 16000),
                openai_api_key=api_keys.get("openai_api_key"),
                stream=False
            )
            return response
        elif model in GROQ_MODELS:
            response = call_groq_api(
                model,
                prompt,
                temperature=st.session_state.temperature_slider_chat,
                max_tokens=min(st.session_state.max_tokens_slider_chat, 8000),
                groq_api_key=api_keys.get("groq_api_key")
            )
            return response.strip()
        else:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": st.session_state.temperature_slider_chat,
                    "num_predict": min(st.session_state.max_tokens_slider_chat, 16000)
                }
            )
            return response['response'].strip()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error generating prompt suggestion: {e}")
        return None

def get_graphrag_context(user_input, corpus_name):
    """Get context from GraphRAG corpus."""
    try:
        embedder = OllamaEmbedder()
        corpus = GraphRAGCorpus.load(corpus_name, embedder)
        results = corpus.query(user_input, n_results=3)
        
        context = ""
        for result in results:
            context += f"Relevant Information (Similarity: {result['similarity']:.4f}):\n{result['content']}\n\n"
        
        return context.strip() if context else None
    except Exception as e:
        st.error(f"Error querying GraphRAG corpus: {str(e)}")
        logger.error(f"Error querying GraphRAG corpus: {e}")
        return None

def extract_content_blocks(text):
    if text is None:
        return [], []
    
    # Extract code blocks
    code_blocks = re.findall(r'```[\s\S]*?```', text)
    
    # Remove code blocks from the text
    text_without_code = re.sub(r'```[\s\S]*?```', '', text)
    
    # Extract article blocks that start with 'Title:' and continue until the next 'Title:' or the end of the text
    article_blocks = re.findall(r'^Title:.*?(?=\n^Title:|\Z)', text_without_code, re.MULTILINE | re.DOTALL)
    
    return [block.strip('`').strip() for block in code_blocks], [block.strip() for block in article_blocks]

def construct_agent_prompt(agent_type, metacognitive_type, voice_type, selected_prompt=None):
    prompt = ""

    agent_prompts = get_agent_prompt()
    if agent_type != "None" and agent_type in agent_prompts:
        prompt += f"You are a {agent_type}. {agent_prompts[agent_type]['prompt']}\n\n"
    else:
        prompt += "You are a helpful AI assistant.\n\n"
    
    if metacognitive_type != "None":
        prompt += f"Use the following metacognitive approach: {get_metacognitive_prompt()[metacognitive_type]}\n\n"
    
    if voice_type != "None":
        prompt += f"Speak in the following voice: {get_voice_prompt()[voice_type]}\n\n"
    
    prompt += """When you generate code or an article, it will be automatically saved to the user's Workspace. 
For code, use triple backticks (```) to enclose the code block. 
If the user asks for a blog post, article, or report, start with (Title:) followed by the article title on a new line, then the content.
Keep the formatting clean and consistent for both code and articles.\n\n"""
    
    if selected_prompt:
        prompt += f"{selected_prompt}\n\n"
    
    return prompt

def calculate_modularity(similarity_matrix: np.ndarray, communities: list) -> float:
    """Calculates the modularity of a graph given its similarity matrix and community structure."""
    if similarity_matrix.ndim != 2:
        logger.error(f"similarity_matrix should be 2D, but has {similarity_matrix.ndim} dimensions")
        return 0.0

    m = np.sum(similarity_matrix) / 2  # Total edge weight
    Q = 0
    for community in communities:
        for i in community:
            for j in community:
                if i >= similarity_matrix.shape[0] or j >= similarity_matrix.shape[1]:
                    logger.error(f"Index out of bounds. i={i}, j={j}, matrix shape={similarity_matrix.shape}")
                    continue
                Q += similarity_matrix[i, j] - (np.sum(similarity_matrix[i, :]) * np.sum(similarity_matrix[:, j])) / m
    return Q / (2 * m)

def refine_boundaries(embeddings: np.ndarray, surprise_indices: list) -> list:
    """Refines event boundaries using modularity."""
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Surprise indices: {surprise_indices}")
    
    similarity_matrix = cosine_similarity(embeddings)
    logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    communities = []
    start = 0
    for end in surprise_indices:
        communities.append(list(range(start, end)))
        start = end
    communities.append(list(range(start, len(embeddings))))
    logger.info(f"Number of communities: {len(communities)}")

    best_boundaries = surprise_indices.copy()
    best_modularity = calculate_modularity(similarity_matrix, communities)
    logger.info(f"Initial modularity: {best_modularity}")

    for i in range(len(surprise_indices) - 1):
        for j in range(surprise_indices[i] + 1, surprise_indices[i + 1]):
            temp_boundaries = best_boundaries.copy()
            temp_boundaries[i] = j
            temp_communities = []
            start = 0
            for end in temp_boundaries:
                temp_communities.append(list(range(start, end)))
                start = end
            temp_communities.append(list(range(start, len(embeddings))))
            temp_modularity = calculate_modularity(similarity_matrix, temp_communities)
            temp_modularity = calculate_modularity(similarity_matrix, temp_communities)
            logger.debug(f"Testing boundaries at step {i}, position {j}: modularity={temp_modularity}")
            if temp_modularity > best_modularity:
                best_modularity = temp_modularity
                best_boundaries = temp_boundaries.copy()
                logger.info(f"Updated best modularity to {best_modularity} with boundaries {best_boundaries}")

    return best_boundaries

def get_token_embeddings(model: str, text: str, api_keys: dict) -> np.ndarray:
    """Gets embeddings for each token in the text and returns a 2D array."""
    try:
        if model in OPENAI_MODELS:
            embeddings = call_openai_embeddings(
                model="text-embedding-ada-002",
                input_text=text
            )
        elif model in GROQ_MODELS:
            embeddings = get_local_embeddings(text)
        else:
            response = ollama.embeddings(model=model, prompt=text)
            embeddings = np.array(response['embedding'])
        
        return np.array(embeddings).reshape(-1, embeddings.shape[-1])  # Ensure it's a 2D array
    except Exception as e:
        st.error(f"An error occurred while getting token embeddings: {e}")
        logger.error(f"Error getting token embeddings: {e}")
        return np.zeros((1, 1536))  # Return a default 2D array with 1536 features (common embedding size)

def advanced_thinking_step(prompt: str, model: str, api_keys: dict, step: str) -> str:
    """Processes a single thinking step and returns the result."""
    step_prompt = f"{prompt}\n\nCurrent thinking step: {step}\nProvide your thoughts for this step:"
    logger.info(f"Executing thinking step: {step}")

    try:
        if model in OPENAI_MODELS:
            response = call_openai_api(
                model=model,
                messages=[{"role": "user", "content": step_prompt}],
                temperature=0.7,
                max_tokens=200,
                openai_api_key=api_keys.get("openai_api_key"),
                stream=False
            )
        elif model in GROQ_MODELS:
            response = call_groq_api(
                model=model,
                prompt=step_prompt,
                temperature=0.7,
                max_tokens=200,
                groq_api_key=api_keys.get("groq_api_key")
            )
        else:
            response = ollama.generate(
                model=model,
                prompt=step_prompt,
                options={
                    "temperature": 0.7,
                    "num_predict": 200
                }
            )
            response = response['response']
        
        logger.info(f"Completed thinking step: {step}")
        return f"**{step}**\n\n{response}\n\n"
    except Exception as e:
        error_message = f"Error during {step}: {e}"
        logger.error(error_message)
        return f"**{step}**\n\n{error_message}\n\n"

def instance_adaptive_cot(prompt: str, model: str, api_keys: dict) -> str:
    """Implements Instance-Adaptive Zero-Shot CoT Prompting."""
    # Select the best prompt based on saliency scores
    selected_prompt = select_best_prompt(prompt, model, api_keys)
    if selected_prompt:
        full_prompt = f"{selected_prompt} {prompt}"
    else:
        # Fallback to default behavior if no prompt is selected
        full_prompt = prompt
    return full_prompt

def select_best_prompt(question: str, model: str, api_keys: dict) -> str:
    """Selects the best prompt from candidate prompts based on saliency scores."""
    saliency_scores = []
    for candidate in CANDIDATE_PROMPTS:
        # Generate rationale using the candidate prompt
        rationale = generate_rationale(question, candidate, model, api_keys)
        if not rationale:
            saliency_scores.append(0)
            continue
        
        # Calculate saliency scores
        Iqp, Iqr, Ipr = calculate_saliency_scores(question, candidate, rationale, model, api_keys)
        synthesized_score = synthesize_saliency_score(Iqp, Iqr, Ipr)
        saliency_scores.append(synthesized_score)
    
    if not saliency_scores:
        return None
    
    # Depending on the strategy, select the prompt
    if st.session_state.cot_strategy == "IAP-ss":
        for idx, score in enumerate(saliency_scores):
            if score >= st.session_state.cot_threshold:
                logger.info(f"Selected prompt (IAP-ss): {CANDIDATE_PROMPTS[idx]} with score {score}")
                return CANDIDATE_PROMPTS[idx]
        # If no prompt meets the threshold, return None
        logger.info("No prompt met the threshold in IAP-ss strategy.")
        return None
    elif st.session_state.cot_strategy == "IAP-mv":
        top_n = st.session_state.cot_top_n
        top_indices = np.argsort(saliency_scores)[-top_n:]
        selected_prompts = [CANDIDATE_PROMPTS[idx] for idx in top_indices]
        # Majority vote (here, we'll select the prompt with the highest score)
        best_prompt = selected_prompts[np.argmax([saliency_scores[idx] for idx in top_indices])]
        logger.info(f"Selected prompt (IAP-mv): {best_prompt} with score {max([saliency_scores[idx] for idx in top_indices])}")
        return best_prompt
    else:
        return None

def generate_rationale(question: str, prompt: str, model: str, api_keys: dict) -> str:
    """Generates rationale using the given prompt."""
    full_prompt = f"{prompt} {question}"
    try:
        if model in OPENAI_MODELS:
            response = call_openai_api(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.7,
                max_tokens=500,
                openai_api_key=api_keys.get("openai_api_key"),
                stream=False
            )
            return response
        elif model in GROQ_MODELS:
            response = call_groq_api(
                model=model,
                prompt=full_prompt,
                temperature=0.7,
                max_tokens=500,
                groq_api_key=api_keys.get("groq_api_key")
            )
            return response.strip()
        else:
            response = ollama.generate(
                model=model,
                prompt=full_prompt,
                options={
                    "temperature": 0.7,
                    "num_predict": 500
                }
            )
            return response['response'].strip()
    except Exception as e:
        logger.error(f"Error generating rationale: {e}")
        return ""

def calculate_saliency_scores(question: str, prompt: str, rationale: str, model: str, api_keys: dict):
    """Calculates saliency scores Iqp, Iqr, Ipr."""
    # Placeholder implementation
    # In a real scenario, this would involve accessing the model's attention matrices and gradients
    # Here, we'll use dummy values for demonstration purposes
    Iqp = np.random.rand()
    Iqr = np.random.rand()
    Ipr = np.random.rand()
    logger.debug(f"Saliency scores for prompt '{prompt}': Iqp={Iqp}, Iqr={Iqr}, Ipr={Ipr}")
    return Iqp, Iqr, Ipr

def synthesize_saliency_score(Iqp: float, Iqr: float, Ipr: float) -> float:
    """Synthesizes the saliency score based on Iqp, Iqr, Ipr."""
    lambda1 = 0.4
    lambda2 = 0.3
    lambda3 = 0.3
    S = lambda1 * Iqp + lambda2 * Iqr + lambda3 * Ipr
    logger.debug(f"Synthesized saliency score: {S}")
    return S

def chat_interface():
    load_settings()

    # Initialize session state attributes with default values if not present
    if "cot_top_n" not in st.session_state:
        st.session_state.cot_top_n = 3  # Default value for top N prompts in IAP-mv
    if "cot_strategy" not in st.session_state:
        st.session_state.cot_strategy = "IAP-ss"  # Default strategy
    if "cot_threshold" not in st.session_state:
        st.session_state.cot_threshold = 0.5  # Default threshold for IAP-ss
    if "instance_adaptive_cot_enabled" not in st.session_state:
        st.session_state.instance_adaptive_cot_enabled = False  # Default to disabled

    # Initialize other attributes if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_type" not in st.session_state:
        st.session_state.agent_type = "None"
    if "workspace_items" not in st.session_state:
        st.session_state.workspace_items = []
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""
    if "suggested_prompt" not in st.session_state:
        st.session_state.suggested_prompt = ""
    if "show_prompt_modal" not in st.session_state:
        st.session_state.show_prompt_modal = False
    if "episodic_memory_enabled" not in st.session_state:
        st.session_state.episodic_memory_enabled = False
    if "advanced_thinking_enabled" not in st.session_state:
        st.session_state.advanced_thinking_enabled = False  # Initialize Advanced Thinking
    if "thinking_steps" not in st.session_state:
        st.session_state.thinking_steps = [
            "1. Analyzing the problem",
            "2. Breaking down into subtasks",
            "3. Exploring potential solutions",
            "4. Evaluating approaches",
            "5. Formulating a comprehensive answer"
        ]  # Default thinking steps
    if "model_memory_handler" not in st.session_state:
        st.session_state.model_memory_handler = ModelMemoryHandler("ollama")
    if "groq_client" not in st.session_state:
        api_keys = load_api_keys()
        st.session_state.groq_client = get_groq_client(api_keys.get("groq_api_key"))

    # Ensure 'selected_model' is initialized
    if "selected_model" not in st.session_state or st.session_state.selected_model is None:
        available_models = get_available_models()
        if available_models:
            st.session_state.selected_model = available_models[0]
            logger.info(f"Initialized selected_model to default: {st.session_state.selected_model}")
        else:
            st.session_state.selected_model = None
            logger.warning("No available models found to initialize selected_model.")
    
    st.session_state.agent_type = st.session_state.get("agent_type", "None")
    st.session_state.metacognitive_type = st.session_state.get("metacognitive_type", "None")
    st.session_state.voice_type = st.session_state.get("voice_type", "None")

    st.session_state.selected_corpus = st.session_state.get("selected_corpus", "None")
    st.session_state.temperature_slider_chat = st.session_state.get("temperature_slider_chat", 0.7)
    st.session_state.max_tokens_slider_chat = min(st.session_state.get("max_tokens_slider_chat", 4000), 16000) 
    st.session_state.presence_penalty_slider_chat = st.session_state.get("presence_penalty_slider_chat", 0.0)
    st.session_state.frequency_penalty_slider_chat = st.session_state.get("frequency_penalty_slider_chat", 0.0)

    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0

    # Initialize episodic memory
    if "episodic_memory" not in st.session_state:
        st.session_state.episodic_memory = EpisodicMemory()

    with st.sidebar:
        with st.expander("🤖 Chat Agent Settings", expanded=False):
            available_models = get_all_models()  # Update available models
            st.session_state.selected_model = st.selectbox(
                "📦 Model:",
                available_models,
                index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
            )
            agent_types = ["None"] + list(get_agent_prompt().keys())
            st.session_state.agent_type = st.selectbox("🧑‍🔧 Agent Type:", agent_types, index=agent_types.index(st.session_state.agent_type))
            metacognitive_types = ["None"] + list(get_metacognitive_prompt().keys())
            st.session_state.metacognitive_type = st.selectbox("🧠 Metacognitive Type:", metacognitive_types, index=metacognitive_types.index(st.session_state.metacognitive_type))
            voice_options = ["None"] + list(get_voice_prompt().keys())
            st.session_state.voice_type = st.selectbox("🗣️ Voice Type:", voice_options, index=voice_options.index(st.session_state.voice_type) if st.session_state.voice_type in voice_options else 0)
            corpus_options = ["None"] + [d for d in os.listdir(RAGTEST_DIR) if os.path.isdir(os.path.join(RAGTEST_DIR, d))]
            st.session_state.selected_corpus = st.selectbox("📚 Corpus:", corpus_options, index=corpus_options.index(st.session_state.selected_corpus) if st.session_state.selected_corpus in corpus_options else 0)
            st.button("💾 Save Settings", key="save_settings_general", on_click=save_settings)

        # Advanced Settings
        with st.expander("🛠️ Advanced Settings", expanded=False):
            st.session_state.temperature_slider_chat = st.slider(
                "🌡️ Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature_slider_chat,
                step=0.1
            )
            st.session_state.max_tokens_slider_chat = st.slider(
                "📊 Max Tokens",
                min_value=1000,
                max_value=16000,
                value=st.session_state.max_tokens_slider_chat,
                step=1000
            )  # Enforce max token limit
            st.session_state.presence_penalty_slider_chat = st.slider(
                "🚫 Presence Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=st.session_state.presence_penalty_slider_chat,
                step=0.1
            )
            st.session_state.frequency_penalty_slider_chat = st.slider(
                "🔁 Frequency Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=st.session_state.frequency_penalty_slider_chat,
                step=0.1
            )
            st.session_state.episodic_memory_enabled = st.checkbox(
                "Enable Episodic Memory",
                value=st.session_state.episodic_memory_enabled
            )
            st.session_state.advanced_thinking_enabled = st.checkbox(
                "Enable Advanced Thinking",
                value=st.session_state.advanced_thinking_enabled  # Added Advanced Thinking checkbox
            )

            # Instance-Adaptive Zero-Shot CoT Prompting Settings
            st.markdown("### Instance-Adaptive Zero-Shot CoT Prompting")
            st.session_state.instance_adaptive_cot_enabled = st.checkbox(
                "Enable Instance-Adaptive Zero-Shot CoT Prompting",
                value=st.session_state.get("instance_adaptive_cot_enabled", False)
            )
            if st.session_state.instance_adaptive_cot_enabled:
                cot_strategies = ["IAP-ss", "IAP-mv"]
                st.session_state.cot_strategy = st.selectbox(
                    "📋 CoT Prompting Strategy:",
                    cot_strategies,
                    index=cot_strategies.index(st.session_state.get("cot_strategy", "IAP-ss")) if st.session_state.get("cot_strategy", "IAP-ss") in cot_strategies else 0
                )
                if st.session_state.cot_strategy == "IAP-ss":
                    st.session_state.cot_threshold = st.slider(
                        "🔍 Saliency Score Threshold:",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.get("cot_threshold", 0.5),
                        step=0.05
                    )
                elif st.session_state.cot_strategy == "IAP-mv":
                    st.session_state.cot_top_n = st.slider(
                        "🔝 Number of Top Prompts:",
                        min_value=1,
                        max_value=len(CANDIDATE_PROMPTS),
                        value=st.session_state.get("cot_top_n", 3),
                        step=1
                    )

            # Configurable Thinking Steps
            st.markdown("### Configurable Thinking Steps")
            default_steps = "\n".join(st.session_state.thinking_steps)
            new_thinking_steps = st.text_area(
                "Enter thinking steps (one per line):",
                value=default_steps,
                height=150
            )
            st.session_state.thinking_steps = [step.strip() for step in new_thinking_steps.split('\n') if step.strip()]
            
            st.button("💾 Save Settings", key="save_settings_advanced", on_click=save_settings)

        with st.expander("📁 Saved Chats", expanded=False):
            manage_saved_chats()

        if st.button("📥 Save Chat"):
            save_chat_and_workspace()

    # Define the tabs *before* they are used
    chat_tab, workspace_tab = st.tabs(["💬 Chat", "📜 Workspace"])

    with chat_tab:
        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    if message.get("content"):
                        code_blocks, article_blocks = extract_content_blocks(message["content"])
                        for code_block in code_blocks:
                            st.code(code_block)
                        non_code_parts = re.split(r'```[\s\S]*?```', message["content"])
                        for part in non_code_parts:
                            st.markdown(part.strip())

                        # Add TTS button for AI responses that are not code
                        if not code_blocks and i == len(st.session_state.chat_history) - 1:  # Only for the last response
                            if st.button("🔊 Speak"):
                                try:
                                    agent_prompts = get_agent_prompt()
                                    model_voice = agent_prompts.get(st.session_state.agent_type, {}).get('model_voice')
                                    if model_voice:
                                        speech_file = text_to_speech(part.strip(), voice=model_voice)
                                        play_speech(speech_file)
                                    else:
                                        st.warning(f"No Model Voice found for {st.session_state.agent_type}")
                                except Exception as e:
                                    st.error(f"Error generating or playing speech: {str(e)}")
                    else:
                        st.warning("This message has no content.")
                else:
                    if message.get("content"):
                        st.markdown(message["content"])
                    else:
                        st.warning("This message has no content.")

        with bottom():
            col1, col2 = st.columns([1, 20])
            with col1:
                if st.button("✨", key="prompt_helper", help="Need help writing a prompt?"):
                    st.session_state.show_prompt_modal = True
                    st.rerun()
            with col2:
                user_input = st.chat_input("What is up my person?")

        if st.session_state.get("show_prompt_modal", False):
            ai_assisted_prompt_writing()

        if st.session_state.chat_input:
            user_input = st.session_state.chat_input
            st.session_state.chat_input = ""
        
        if user_input:
            api_keys = load_api_keys()
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.total_tokens += count_tokens(user_input)
            logger.info(f"User input received: {user_input}")

            agent_prompt = construct_agent_prompt(
                st.session_state.agent_type,
                st.session_state.metacognitive_type,
                st.session_state.voice_type
            )

            chat_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_history[-5:]])

            corpus_context = ""
            if st.session_state.selected_corpus != "None":
                graph_response = get_graphrag_context(user_input, st.session_state.selected_corpus)
                if graph_response:
                    corpus_context = f"\nRelevant context from the knowledge base:\n{graph_response}\n"
                else:
                    st.warning(f"No relevant context found in the corpus '{st.session_state.selected_corpus}'. Proceeding without additional context.")
                    logger.warning(f"No relevant context found in corpus '{st.session_state.selected_corpus}'.")

            episodic_context = ""
            if st.session_state.episodic_memory_enabled:
                try:
                    chat_history_text = "\n".join([msg['content'] for msg in st.session_state.chat_history])
                    st.session_state.model_memory_handler.segment_text(st.session_state.selected_model, chat_history_text, api_keys)
                    query_embedding = get_token_embeddings(st.session_state.selected_model, user_input, api_keys)
                    
                    if query_embedding.size > 0:
                        retrieved_events = st.session_state.model_memory_handler.retrieve_events(query_embedding)
                        for event in retrieved_events:
                            if event['text'] is not None:
                                episodic_context += f" {event['text']}"
                    else:
                        st.warning("Failed to generate embeddings for episodic memory. Proceeding without episodic context.")
                        logger.warning("Failed to generate embeddings for episodic memory.")
                except Exception as e:
                    st.error(f"Error in episodic memory processing: {str(e)}. Proceeding without episodic context.")
                    logger.error(f"Error in episodic memory processing: {e}")

            # Get webpage context from query parameters using st.query_params
            query_params = st.query_params  
            web_page_url = query_params.get('web_page_url', [''])[0]
            is_extension = query_params.get('extension', ['false'])[0].lower() == 'true'

            # Construct the initial prompt
            if is_extension and web_page_url:
                initial_prompt = f"You are an AI assistant working within a browser extension. The user is currently on the webpage: {web_page_url}. How can I help you with information related to this page?\n\n"
            else:
                initial_prompt = "You are an AI assistant. How can I help you today?\n\n"

            # Initialize chat history if empty
            if "chat_history" not in st.session_state or not st.session_state.chat_history:
                st.session_state.chat_history = [{"role": "assistant", "content": initial_prompt}]
                logger.info("Initialized chat history with initial prompt.")

            # Construct the final prompt
            final_prompt = ""
            
            # Conditionally add browser extension meta-context
            if is_extension and web_page_url:
                final_prompt += "You are an AI assistant working within a browser extension. You have access to the current web page's content. Please use this information to answer the user's question.\n\n"
                # Assuming web_page_content is defined elsewhere or fetched as needed
                web_page_content = query_params.get('web_page_content', [''])[0]
                final_prompt += f"Webpage URL: {web_page_url}\nWebpage Content:\n{web_page_content}\n\n" 

            final_prompt += f"""
{agent_prompt}

Recent conversation history:
{chat_history}

{corpus_context}

Episodic Memory Context:
{episodic_context}

Human: {user_input}

Assistant: Let me address your request based on the information provided and my capabilities.
"""

            # Apply Instance-Adaptive Zero-Shot CoT Prompting if enabled
            if st.session_state.instance_adaptive_cot_enabled:
                final_prompt = instance_adaptive_cot(final_prompt, st.session_state.selected_model, api_keys)
                if final_prompt is None:
                    final_prompt = f"{agent_prompt}\n\nRecent conversation history:\n{chat_history}\n\n{corpus_context}\n\nEpisodic Memory Context:\n{episodic_context}\n\nHuman: {user_input}\n\nAssistant: Let me address your request based on the information provided and my capabilities.\n\n"

            st.session_state.total_tokens += count_tokens(final_prompt)
            logger.info(f"Constructed final prompt with total tokens: {st.session_state.total_tokens}")

            if st.session_state.advanced_thinking_enabled:
                with st.sidebar.expander("🧠 Advanced Thinking Process", expanded=True):
                    thinking_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    thinking_result_placeholder = st.empty()

                    # Initialize session state for advanced thinking
                    if "advanced_thinking_step_index" not in st.session_state:
                        st.session_state.advanced_thinking_step_index = 0
                        st.session_state.advanced_thinking_result = ""

                    total_steps = len(st.session_state.thinking_steps)

                    # Process each thinking step sequentially
                    for step_index, step in enumerate(st.session_state.thinking_steps):
                        # Update progress bar
                        progress = (step_index) / total_steps
                        progress_bar.progress(progress)

                        # Display current thinking result
                        thinking_result_placeholder.markdown(st.session_state.advanced_thinking_result)

                        # Process the current step
                        step_result = advanced_thinking_step(
                            prompt=final_prompt,
                            model=st.session_state.selected_model,
                            api_keys=api_keys,
                            step=step
                        )

                        # Append the result to the session state
                        st.session_state.advanced_thinking_result += step_result

                        # Allow the UI to update
                        thinking_result_placeholder.markdown(st.session_state.advanced_thinking_result)
                        progress = (step_index + 1) / total_steps
                        progress_bar.progress(progress)
                        time.sleep(0.1)  # Optional: Simulate delay for better visualization

                    # Final update after all steps
                    thinking_placeholder.text("Advanced thinking process completed.")
                    logger.info("Advanced thinking process completed.")

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                try:
                    if st.session_state.selected_model in OPENAI_MODELS:
                        response = call_openai_api(
                            model=st.session_state.selected_model,
                            messages=[{"role": "user", "content": final_prompt}],
                            temperature=st.session_state.temperature_slider_chat,
                            max_tokens=min(st.session_state.max_tokens_slider_chat, 16000),
                            openai_api_key=api_keys.get("openai_api_key"),
                            stream=True
                        )
                        for chunk in response:
                            if chunk.choices[0].delta.content is not None:
                                full_response += chunk.choices[0].delta.content
                                message_placeholder.markdown(full_response + "▌")
                                st.session_state.total_tokens += count_tokens(chunk.choices[0].delta.content)
                        message_placeholder.markdown(full_response)
                    elif st.session_state.selected_model in GROQ_MODELS:
                        full_response = call_groq_api(
                            client=st.session_state.groq_client,
                            model=st.session_state.selected_model,
                            messages=[{"role": "user", "content": final_prompt}],
                            temperature=st.session_state.temperature_slider_chat,
                            max_tokens=min(st.session_state.max_tokens_slider_chat, 8000)
                        )
                        message_placeholder.markdown(full_response)
                    else:
                        for response_chunk in ollama.generate(
                            st.session_state.selected_model,
                            final_prompt,
                            stream=True,
                            options={
                                "temperature": st.session_state.temperature_slider_chat,
                                "num_predict": min(st.session_state.max_tokens_slider_chat, 16000),
                                "presence_penalty": st.session_state.presence_penalty_slider_chat,
                                "frequency_penalty": st.session_state.frequency_penalty_slider_chat,
                            }
                        ):
                            content = response_chunk["response"]
                            full_response += content
                            message_placeholder.markdown(full_response + "▌")
                            st.session_state.total_tokens += count_tokens(content)

                        message_placeholder.markdown(full_response)
                    
                    logger.info("Assistant response generated successfully.")
                except Exception as e:
                    st.error(f"Error during response generation: {e}")
                    logger.error(f"Error during response generation: {e}")

                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

                # Extract and save content blocks
                code_blocks, article_blocks = extract_content_blocks(full_response)

                for code_block in code_blocks:
                    st.session_state.workspace_items.append({
                        "type": "code",
                        "content": code_block,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                for article_block in article_blocks:
                    st.session_state.workspace_items.append({
                        "type": "article",
                        "content": article_block,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                if code_blocks or article_blocks:
                    st.success(f"{len(code_blocks)} code block(s) and {len(article_blocks)} article(s) automatically saved to Workspace")
                    logger.info(f"Saved {len(code_blocks)} code blocks and {len(article_blocks)} articles to Workspace.")

    with workspace_tab:
        for index, item in enumerate(st.session_state.workspace_items):
            with st.expander(f"Item {index + 1} - {item['timestamp']}"):
                if item['type'] == 'code':
                    st.code(item['content'])
                elif item['type'] == 'article':
                    lines = item['content'].split('\n')
                    st.subheader(lines[0].replace('Title:', '').strip())
                    st.markdown('\n'.join(lines[1:]))
                else:
                    st.write(item['content'])
                if st.button(f"Remove Item {index + 1}"):
                    st.session_state.workspace_items.pop(index)
                    logger.info(f"Removed item {index + 1} from Workspace.")
                    st.rerun()

        new_item = st.text_area("Add a new item to the workspace:", key="new_workspace_item")
        if st.button("✚ Add to Workspace"):
            if new_item:
                st.session_state.workspace_items.append({
                    "type": "text",
                    "content": new_item,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.success("New item added to Workspace")
                logger.info("Added a new text item to Workspace.")
                st.rerun()

    # Update model memory handler when model is changed
    if st.session_state.selected_model != st.session_state.get("previous_model"):
        if st.session_state.selected_model in OPENAI_MODELS:
            st.session_state.model_memory_handler = ModelMemoryHandler("openai")
        elif st.session_state.selected_model in GROQ_MODELS:
            st.session_state.model_memory_handler = ModelMemoryHandler("groq")
        else:
            st.session_state.model_memory_handler = ModelMemoryHandler("ollama")
        st.session_state.previous_model = st.session_state.selected_model
        logger.info(f"Switched model to {st.session_state.selected_model}")

def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def extract_code_blocks(text):
    if text is None:
        return [], []
    code_blocks = re.findall(r'```[\s\S]*?```', text)
    article_blocks = re.findall(r'^Title:.*?(?=\n^Title:|\Z)', text, re.MULTILINE | re.DOTALL)
    return [block.strip('`').strip() for block in code_blocks], [block.strip() for block in article_blocks]

def save_chat_and_workspace():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"{timestamp}"
    chat_name = st.text_input("Enter a name for the save:", value=default_filename, key="save_chat_name")
    if chat_name and st.button("Confirm Save"):
        save_data = {
            "chat_history": st.session_state.chat_history,
            "workspace_items": st.session_state.workspace_items,
            "total_tokens": st.session_state.total_tokens,
            "thinking_steps": st.session_state.thinking_steps,  # Save thinking steps
            "instance_adaptive_cot_enabled": st.session_state.get("instance_adaptive_cot_enabled", False),
            "cot_strategy": st.session_state.get("cot_strategy", "IAP-ss"),
            "cot_threshold": st.session_state.get("cot_threshold", 0.5),
            "cot_top_n": st.session_state.get("cot_top_n", 3)
        }
        sessions_folder = "sessions"
        if not os.path.exists(sessions_folder):
            os.makedirs(sessions_folder)
        file_path = os.path.join(sessions_folder, chat_name + ".json")
        try:
            with open(file_path, "w") as f:
                json.dump(save_data, f)
            st.success(f"Chat and Workspace saved to {chat_name}")
            logger.info(f"Chat and Workspace saved to {chat_name}.json")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to save chat: {e}")
            logger.error(f"Failed to save chat: {e}")

def manage_saved_chats():
    st.sidebar.subheader("Saved Chats and Workspaces")
    sessions_folder = "sessions"
    if not os.path.exists(sessions_folder):
        os.makedirs(sessions_folder)
    saved_files = [f for f in os.listdir(sessions_folder) if f.endswith(".json")]

    if "rename_file" not in st.session_state:
        st.session_state.rename_file = None

    for file in saved_files:
        col1, col2, col3 = st.sidebar.columns([3, 1, 1])
        with col1:
            file_name = os.path.splitext(file)[0]
            if st.button(file_name, key=f"load_{file}"):
                load_chat_and_workspace(os.path.join(sessions_folder, file))
        with col2:
            if st.button("✏️", key=f"rename_{file}"):
                st.session_state.rename_file = file
                st.rerun()
        with col3:
            if st.button("🗑️", key=f"delete_{file}"):
                delete_chat_and_workspace(os.path.join(sessions_folder, file))

    if st.session_state.rename_file:
        rename_chat_and_workspace(st.session_state.rename_file, sessions_folder)

def load_chat_and_workspace(file_path):
    try:
        with open(file_path, "r") as f:
            loaded_data = json.load(f)
        st.session_state.chat_history = loaded_data.get("chat_history", [])
        st.session_state.workspace_items = loaded_data.get("workspace_items", [])
        st.session_state.total_tokens = loaded_data.get("total_tokens", 0)
        st.session_state.thinking_steps = loaded_data.get("thinking_steps", st.session_state.thinking_steps)
        st.session_state.instance_adaptive_cot_enabled = loaded_data.get("instance_adaptive_cot_enabled", False)
        st.session_state.cot_strategy = loaded_data.get("cot_strategy", "IAP-ss")
        st.session_state.cot_threshold = loaded_data.get("cot_threshold", 0.5)
        st.session_state.cot_top_n = loaded_data.get("cot_top_n", 3)
        st.success(f"Loaded {os.path.basename(file_path)}")
        logger.info(f"Loaded chat and workspace from {file_path}")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to load chat: {e}")
        logger.error(f"Failed to load chat from {file_path}: {e}")

def delete_chat_and_workspace(file_path):
    try:
        os.remove(file_path)
        st.success(f"File {os.path.basename(file_path)} deleted.")
        logger.info(f"Deleted file {file_path}")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to delete file: {e}")
        logger.error(f"Failed to delete file {file_path}: {e}")

def rename_chat_and_workspace(file_to_rename, sessions_folder):
    current_name = os.path.splitext(file_to_rename)[0]
    new_name = st.sidebar.text_input("Rename file:", value=current_name, key="rename_file_input")
    if st.sidebar.button("Confirm Rename"):
        if new_name and new_name != current_name:
            old_file_path = os.path.join(sessions_folder, file_to_rename)
            new_file_path = os.path.join(sessions_folder, new_name + ".json")
            try:
                if not os.path.exists(new_file_path):
                    os.rename(old_file_path, new_file_path)
                    st.sidebar.success(f"File renamed to {new_name}")
                    logger.info(f"Renamed file from {file_to_rename} to {new_name}.json")
                    st.session_state.rename_file = None
                    st.rerun()
                else:
                    st.sidebar.error("A file with the new name already exists.")
            except Exception as e:
                st.sidebar.error(f"Failed to rename file: {e}")
                logger.error(f"Failed to rename file from {file_to_rename} to {new_name}.json: {e}")
        else:
            st.sidebar.error("Please enter a new name different from the current one.")
    
    if st.sidebar.button("Cancel"):
        st.session_state.rename_file = None
        st.rerun()

if __name__ == "__main__":
    chat_interface()
