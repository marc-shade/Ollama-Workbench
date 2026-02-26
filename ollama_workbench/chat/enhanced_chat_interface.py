"""
Enhanced Chat Interface for Ollama Workbench

This module preserves all the original chat_interface.py functionality
while adding modern Open WebUI-inspired styling improvements.
"""

import streamlit as st
import os
import json
import re
import time
import logging
from datetime import datetime
import tiktoken
import numpy as np
from collections import deque
from streamlit_extras.bottom_container import bottom

logger = logging.getLogger("enhanced_chat_interface")

# Import original chat_interface.py functionality
from .chat_interface import (
    chat_interface, count_tokens,
    extract_code_blocks, extract_content_blocks, calculate_modularity,
    refine_boundaries, get_graphrag_context, ModelMemoryHandler, 
    instance_adaptive_cot, construct_agent_prompt, advanced_thinking_step,
    RAGTEST_DIR, CANDIDATE_PROMPTS
)

# Import fixed settings functions that handle Streamlit session state better
try:
    from fixed_chat_settings import fixed_load_settings as load_settings, fixed_save_settings as save_settings, SETTINGS_FILE
    logger.info("Successfully imported fixed_chat_settings")
except ImportError as e:
    # Fallback implementation for tests
    logger.warning(f"Error importing fixed_chat_settings: {e}. Using fallback implementation.")
    
    # Define fallback functions for testing
    SETTINGS_FILE = "chat-settings.json"
    
    def load_settings():
        """Fallback load_settings function for tests"""
        logger.info("CHECKPOINT: Using fallback load_settings function for tests")
        # In test environment, just set some defaults
        import streamlit as st
        if isinstance(st.session_state, dict):
            if "selected_model" not in st.session_state:
                st.session_state["selected_model"] = "llama2"
            if "agent_type" not in st.session_state:
                st.session_state["agent_type"] = "Researcher"
            if "metacognitive_type" not in st.session_state:
                st.session_state["metacognitive_type"] = "Analytical"
            if "voice_type" not in st.session_state:
                st.session_state["voice_type"] = "Professional"
            if "temperature" not in st.session_state:
                st.session_state["temperature"] = 0.8
        return True
    
    def save_settings():
        """Fallback save_settings function for tests"""
        logger.info("CHECKPOINT: Using fallback save_settings function for tests")
        return True

# Import utility functions
from ollama_workbench.providers.ollama_utils import (
    get_available_models, get_all_models, load_api_keys, get_token_embeddings
)
from ollama_workbench.providers.openai_utils import call_openai_api, OPENAI_MODELS
from ollama_workbench.providers.groq_utils import get_groq_client, call_groq_api, GROQ_MODELS
from ollama_workbench.providers.mistral_utils import call_mistral_api, MISTRAL_MODELS
from ollama_workbench.ui.prompts import (
    get_agent_prompt, get_metacognitive_prompt, get_voice_prompt
)
from ollama_workbench.knowledge.enhanced_corpus import GraphRAGCorpus, OllamaEmbedder
from .tts_utils import text_to_speech, play_speech
logger = logging.getLogger("enhanced_chat_interface")

def apply_modern_styling():
    """Apply modern Open WebUI-inspired styling while preserving functionality."""
    st.markdown("""
        <style>
        /* Make all containers use full width */
        .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        /* Modern UI elements */
        .main {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        /* Chat message styling */
        .stChatMessage .stChatMessageContent {
            border-radius: 0.75rem !important;
            padding: 0.75rem 1rem !important;
            margin-bottom: 0.5rem !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        }
        
        /* User message styling */
        .stChatMessage.user .stChatMessageContent {
            background-color: #3B82F6 !important;
            color: white !important;
            border-radius: 0.75rem 0.75rem 0 0.75rem !important;
        }
        
        /* Assistant message styling */
        .stChatMessage.assistant .stChatMessageContent {
            background-color: #F3F4F6 !important;
            color: #111827 !important;
            border-radius: 0 0.75rem 0.75rem 0.75rem !important;
        }
        
        /* Improve chat input styling */
        .stChatInputContainer {
            padding: 0.5rem !important;
            border-radius: 0.5rem !important;
            border: 1px solid #E5E7EB !important;
        }
        
        /* Improve sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #F9FAFB !important;
        }
        
        /* Improve expander styling */
        .st-expander {
            border: 1px solid #E5E7EB !important;
            border-radius: 0.5rem !important;
            margin-bottom: 1rem !important;
        }
        
        .st-expander-header {
            font-weight: 500 !important;
        }
        
        /* Improve buttons */
        div.stButton > button {
            border-radius: 0.375rem !important;
            font-weight: 500 !important;
            transition: all 0.15s ease-in-out !important;
        }
        
        /* Code blocks */
        pre {
            background-color: #F3F4F6 !important;
            border-radius: 0.375rem !important;
            padding: 0.75rem !important;
            margin: 0.75rem 0 !important;
        }
        
        /* Dark mode compatible styles */
        @media (prefers-color-scheme: dark) {
            .stChatMessage.assistant .stChatMessageContent {
                background-color: #1F2937 !important;
                color: #F9FAFB !important;
            }
            
            [data-testid="stSidebar"] {
                background-color: #111827 !important;
            }
            
            pre {
                background-color: #1F2937 !important;
            }
            
            .st-expander {
                border-color: #374151 !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)

def enhanced_chat_interface():
    """
    Enhanced chat interface that preserves original functionality with improved styling.
    This is a wrapper around the original chat_interface() function with fixes for Streamlit form elements.
    """
    # Apply modern styling
    apply_modern_styling()
    
    # Fix for model dropdown - ensure session state is loaded properly
    load_settings()
    
    # Debug: Log current session state for model selection
    logger.info(f"Current model selection: {st.session_state.get('selected_model')}")
    logger.info(f"Current agent type: {st.session_state.get('agent_type')}")
    
    # Create a callback to handle form element changes without reloading the page
    def handle_session_state_update(key, value):
        st.session_state[key] = value
        # No rerun here - let the user control when to save
    
    # Ensure chat_interface function knows it's being run in enhanced mode
    if "enhanced_mode" not in st.session_state:
        st.session_state.enhanced_mode = True
    
    # Fix model dropdown issue by adding a custom handler for selectbox
    original_selectbox = st.selectbox
    
    def fixed_selectbox(*args, **kwargs):
        """Custom selectbox that prevents unwanted reruns"""
        # Check if this is a model or agent selectbox that needs special handling
        key = kwargs.get("key")
        if key in ["selected_model", "agent_type", "metacognitive_type", "voice_type", "selected_corpus"]:
            # Get the current value from session state
            current_value = st.session_state.get(key)
            
            # Make sure we have the correct index for the current value
            options = args[1] if len(args) > 1 else kwargs.get("options", [])
            if current_value in options:
                # Override the default index to ensure current value is selected
                kwargs["index"] = options.index(current_value)
            
            # Call the original selectbox with our modifications
            result = original_selectbox(*args, **kwargs)
            
            # After user selects something, update session state without rerun
            if result != current_value:
                # Set value in session state
                st.session_state[key] = result
                # Log the change
                logger.info(f"Updated {key} from {current_value} to {result}")
            
            return result
        else:
            # For other selectboxes, use normal behavior
            return original_selectbox(*args, **kwargs)
    
    # Temporarily replace st.selectbox with our fixed version
    st.selectbox = fixed_selectbox
    
    try:
        # Run the original chat interface with all functionality preserved
        chat_interface()
    finally:
        # Restore original selectbox function
        st.selectbox = original_selectbox
    
    # Optional: Add any additional UI elements at the bottom without disturbing original functionality
    # No footer text needed

if __name__ == "__main__":
    enhanced_chat_interface()