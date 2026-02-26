"""
Session utilities for Ollama-Workbench

This module provides utilities for managing session state across different
chat interface implementations, ensuring consistent behavior and preserving
all agent features and customization options.
"""

import streamlit as st
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Constants
SESSIONS_FOLDER = "sessions"
SETTINGS_FILE = "chat-settings.json"

# Ensure sessions folder exists
if not os.path.exists(SESSIONS_FOLDER):
    os.makedirs(SESSIONS_FOLDER)
    logger.info(f"Created sessions folder: {SESSIONS_FOLDER}")


@st.cache_data(ttl=60)
def _cached_model_default():
    """Return the default model name, cached for 60 seconds to avoid
    hitting the Ollama HTTP API on every Streamlit rerun."""
    from ollama_workbench.providers.ollama_utils import get_dynamic_model_default
    return get_dynamic_model_default()


def initialize_session_state():
    """
    Initialize session state variables consistently across all interfaces.

    This function ensures that all necessary session state variables are
    initialized with appropriate default values. This is the single
    canonical source for session state initialization.
    """
    logger.info("CHECKPOINT: Initializing session state")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize model selection with cached dynamic detection
    if "selected_model" not in st.session_state:
        try:
            default_model = _cached_model_default()
            if not default_model:
                default_model = None
                logger.warning("No models available - setting default_model to None")
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            default_model = None
        st.session_state.selected_model = default_model

    # Initialize agent settings
    if "agent_type" not in st.session_state:
        st.session_state.agent_type = "None"
    if "metacognitive_type" not in st.session_state:
        st.session_state.metacognitive_type = "None"
    if "voice_type" not in st.session_state:
        st.session_state.voice_type = "None"

    # Initialize generation settings
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 4000
    if "presence_penalty" not in st.session_state:
        st.session_state.presence_penalty = 0.0
    if "frequency_penalty" not in st.session_state:
        st.session_state.frequency_penalty = 0.0

    # Initialize corpus settings
    if "selected_corpus" not in st.session_state:
        st.session_state.selected_corpus = "None"

    # Initialize UI state
    if "settings_expanded" not in st.session_state:
        st.session_state.settings_expanded = False
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "enhanced_mode" not in st.session_state:
        st.session_state.enhanced_mode = True

    # -- Keys from main.py --
    if "selected_test" not in st.session_state:
        st.session_state.selected_test = "Chat"
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []
    if "bm_tasks" not in st.session_state:
        st.session_state.bm_tasks = []
    if "show_resource_usage" not in st.session_state:
        st.session_state.show_resource_usage = False
    if "web_page_content" not in st.session_state:
        st.session_state.web_page_content = None
    if "show_prompt" not in st.session_state:
        st.session_state.show_prompt = True

    # -- Keys from projects.py --
    if "projects" not in st.session_state:
        st.session_state.projects = []
    if "selected_project" not in st.session_state:
        st.session_state.selected_project = None
    if "tasks" not in st.session_state:
        st.session_state.tasks = []
    if "agents" not in st.session_state:
        st.session_state.agents = {}
    if "generated_tasks" not in st.session_state:
        st.session_state.generated_tasks = []
    if "generated_agents" not in st.session_state:
        st.session_state.generated_agents = {}
    if "project_manager_settings" not in st.session_state:
        st.session_state.project_manager_settings = {
            "model": "gpt-3.5-turbo",
            "agent_type": "Task Planner",
            "temperature": 0.7,
            "max_tokens": 4000,
        }

    # -- Keys from RAG interfaces --
    if "rag_corpus_name" not in st.session_state:
        st.session_state.rag_corpus_name = "default"
    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []
    if "rag_sources" not in st.session_state:
        st.session_state.rag_sources = {}
    if "rag_last_query" not in st.session_state:
        st.session_state.rag_last_query = ""
    if "rag_embedding_model" not in st.session_state:
        st.session_state.rag_embedding_model = "llama2"
    if "rag_llm_model" not in st.session_state:
        st.session_state.rag_llm_model = "llama2"
    if "rag_temperature" not in st.session_state:
        st.session_state.rag_temperature = 0.7
    if "rag_corpus_status" not in st.session_state:
        st.session_state.rag_corpus_status = {}

    logger.info("CHECKPOINT: Session state initialization complete")

def save_chat_session():
    """
    Save the current chat session to a file.
    
    This function saves the chat history and model selection to a file
    in the sessions folder, with a timestamp in the filename.
    
    Returns:
        str: Path to the saved session file, or None if save failed
    """
    logger.info("CHECKPOINT: Saving chat session")
    
    # Ensure session state is initialized
    if "chat_history" not in st.session_state:
        logger.warning("Cannot save session: chat_history not in session state")
        return None
    
    # Create session data
    session_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chat_history": st.session_state.chat_history,
        "model": st.session_state.get("selected_model"),
        "agent_type": st.session_state.agent_type if "agent_type" in st.session_state else "None",
        "metacognitive_type": st.session_state.metacognitive_type if "metacognitive_type" in st.session_state else "None",
        "voice_type": st.session_state.voice_type if "voice_type" in st.session_state else "None",
        "temperature": st.session_state.temperature if "temperature" in st.session_state else 0.7,
        "max_tokens": st.session_state.max_tokens if "max_tokens" in st.session_state else 4000
    }
    
    # Create filename with timestamp
    filename = f"chat_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(SESSIONS_FOLDER, filename)
    
    try:
        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)
        logger.info(f"Chat session saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving chat session: {e}")
        return None

def load_chat_session(file_path):
    """
    Load a chat session from a file.
    
    This function loads the chat history and model selection from a file
    and updates the session state accordingly.
    
    Args:
        file_path: Path to the session file to load
    
    Returns:
        bool: True if session was loaded successfully, False otherwise
    """
    logger.info(f"CHECKPOINT: Loading chat session from {file_path}")
    
    try:
        with open(file_path, "r") as f:
            session_data = json.load(f)
        
        # Update session state
        st.session_state.chat_history = session_data["chat_history"]
        logger.info(f"Loaded {len(session_data['chat_history'])} messages")
        
        # Update model selection
        if "model" in session_data:
            st.session_state.selected_model = session_data["model"]
            logger.debug(f"Loaded model: {session_data['model']}")

        # Update agent settings
        if "agent_type" in session_data:
            st.session_state.agent_type = session_data["agent_type"]
            logger.debug(f"Loaded agent_type: {session_data['agent_type']}")

        if "metacognitive_type" in session_data:
            st.session_state.metacognitive_type = session_data["metacognitive_type"]
            logger.debug(f"Loaded metacognitive_type: {session_data['metacognitive_type']}")

        if "voice_type" in session_data:
            st.session_state.voice_type = session_data["voice_type"]
            logger.debug(f"Loaded voice_type: {session_data['voice_type']}")

        # Update generation settings
        if "temperature" in session_data:
            st.session_state.temperature = session_data["temperature"]
            logger.debug(f"Loaded temperature: {session_data['temperature']}")

        if "max_tokens" in session_data:
            st.session_state.max_tokens = session_data["max_tokens"]
            logger.debug(f"Loaded max_tokens: {session_data['max_tokens']}")
        
        logger.info(f"Chat session loaded from {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading chat session: {e}")
        return False

def load_settings():
    """
    Load settings from the settings file.
    
    This function loads settings from the settings file and updates
    the session state accordingly.
    
    Returns:
        bool: True if settings were loaded successfully, False otherwise
    """
    logger.info("CHECKPOINT: Loading settings")
    
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
            
            # Update session state with settings
            for key, value in settings.items():
                st.session_state[key] = value
                logger.debug(f"Loaded setting {key}: {value}")
            
            logger.info(f"Settings loaded from {SETTINGS_FILE}")
            return True
        else:
            logger.warning(f"Settings file {SETTINGS_FILE} not found")
            return False
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        return False

def save_settings():
    """
    Save settings to the settings file.
    
    This function saves the current settings from the session state
    to the settings file.
    
    Returns:
        bool: True if settings were saved successfully, False otherwise
    """
    logger.info("CHECKPOINT: Saving settings")
    
    # Collect settings from session state
    settings = {}
    
    # Model selection
    if "selected_model" in st.session_state:
        settings["selected_model"] = st.session_state.selected_model
        logger.debug(f"Saving selected_model: {st.session_state.selected_model}")

    # Agent settings
    if "agent_type" in st.session_state:
        settings["agent_type"] = st.session_state.agent_type
        logger.debug(f"Saving agent_type: {st.session_state.agent_type}")

    if "metacognitive_type" in st.session_state:
        settings["metacognitive_type"] = st.session_state.metacognitive_type
        logger.debug(f"Saving metacognitive_type: {st.session_state.metacognitive_type}")

    if "voice_type" in st.session_state:
        settings["voice_type"] = st.session_state.voice_type
        logger.debug(f"Saving voice_type: {st.session_state.voice_type}")

    # Generation settings
    if "temperature" in st.session_state:
        settings["temperature"] = st.session_state.temperature
        logger.debug(f"Saving temperature: {st.session_state.temperature}")

    if "max_tokens" in st.session_state:
        settings["max_tokens"] = st.session_state.max_tokens
        logger.debug(f"Saving max_tokens: {st.session_state.max_tokens}")

    if "presence_penalty" in st.session_state:
        settings["presence_penalty"] = st.session_state.presence_penalty
        logger.debug(f"Saving presence_penalty: {st.session_state.presence_penalty}")

    if "frequency_penalty" in st.session_state:
        settings["frequency_penalty"] = st.session_state.frequency_penalty
        logger.debug(f"Saving frequency_penalty: {st.session_state.frequency_penalty}")

    # Corpus settings
    if "selected_corpus" in st.session_state:
        settings["selected_corpus"] = st.session_state.selected_corpus
        logger.debug(f"Saving selected_corpus: {st.session_state.selected_corpus}")
    
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
        logger.info(f"Settings saved to {SETTINGS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return False

def get_agent_prompt():
    """
    Get the agent prompt based on the current session state.
    
    This function constructs an agent prompt based on the agent_type,
    metacognitive_type, and voice_type in the session state.
    
    Returns:
        str: The constructed agent prompt
    """
    logger.info("CHECKPOINT: Getting agent prompt")
    
    try:
        from ollama_workbench.chat.chat_interface import construct_agent_prompt
        
        agent_type = st.session_state.agent_type if "agent_type" in st.session_state else "None"
        metacog_type = st.session_state.metacognitive_type if "metacognitive_type" in st.session_state else "None"
        voice_type = st.session_state.voice_type if "voice_type" in st.session_state else "None"
        
        logger.debug(f"Constructing agent prompt with: {agent_type}, {metacog_type}, {voice_type}")
        
        prompt = construct_agent_prompt(agent_type, metacog_type, voice_type)
        return prompt
    except Exception as e:
        logger.error(f"Error getting agent prompt: {e}")
        return ""

def get_rag_context(query):
    """
    Get RAG context for the query.
    
    This function retrieves context from the selected corpus for the query.
    
    Args:
        query: The query to get context for
    
    Returns:
        str: The retrieved context
    """
    logger.info(f"CHECKPOINT: Getting RAG context for query: {query}")
    
    try:
        from ollama_workbench.chat.chat_interface import get_graphrag_context
        
        corpus_name = st.session_state.selected_corpus if "selected_corpus" in st.session_state else "None"
        
        if corpus_name == "None":
            logger.info("No corpus selected, skipping RAG context retrieval")
            return ""
        
        logger.info(f"Retrieving context from corpus: {corpus_name}")
        
        context = get_graphrag_context(query, corpus_name)
        return context
    except Exception as e:
        logger.error(f"Error getting RAG context: {e}")
        return ""

def safe_rerun():
    """
    Safely rerun the Streamlit app.
    
    This function wraps st.rerun() in a try-except block to prevent crashes.
    """
    logger.info("CHECKPOINT: Safe rerun")
    
    try:
        st.rerun()
    except Exception as e:
        logger.error(f"Error during rerun: {e}")
        # Continue without rerun

def log_message(message, level="info"):
    """
    Log a message with the specified level.
    
    Args:
        message: The message to log
        level: The log level (info, warning, error)
    """
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    else:
        logger.info(message)

def log_chat_message(role, content):
    """
    Log a chat message.
    
    Args:
        role: The role of the message (user or assistant)
        content: The content of the message
    """
    logger.info(f"CHECKPOINT: {role.upper()} message: {content[:50]}...")

def log_model_response(model, prompt_tokens, completion_tokens, latency):
    """
    Log a model response.
    
    Args:
        model: The model used
        prompt_tokens: The number of prompt tokens
        completion_tokens: The number of completion tokens
        latency: The latency of the response
    """
    logger.info(f"CHECKPOINT: Model response from {model}")
    logger.info(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Latency: {latency:.2f}s")

def handle_error(error, message="An error occurred"):
    """
    Handle an error.
    
    Args:
        error: The error to handle
        message: A message to display
    
    Returns:
        str: An error message
    """
    logger.error(f"{message}: {error}")
    return f"{message}: {str(error)}"
