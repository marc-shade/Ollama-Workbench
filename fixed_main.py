"""
Fixed version of main.py for Ollama Workbench.

This version incorporates fixes for Streamlit form elements and session state
to ensure that dropdown selections are properly retained.
"""

import streamlit as st
import json
import os
import logging
import functools
from streamlit_option_menu import option_menu
from ui_fix import apply_all_fixes, restore_original, UIFixer, stable_selectbox

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fixed_main")

# Set page config for wide layout with maximum width
st.set_page_config(layout="wide", page_title="Ollama Workbench", page_icon="🦙")

# Add CSS to ensure 100% width on all pages
st.markdown("""
    <style>
    .block-container {
        max-width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Fix dropdown issues */
    .stSelectbox span {
        display: inline-block;
        min-width: 100px;
    }
    
    /* Prevent unwanted reflows */
    .stButton button {
        width: 100%;
    }
    
    /* Fix Streamlit form elements */
    div[data-testid="stForm"] {
        padding-bottom: 1rem !important;
    }
    
    /* Force sidebar to stay in place */
    [data-testid="stSidebar"] {
        position: fixed !important;
        overflow-y: auto !important;
        height: 100vh !important;
    }
    </style>
""", unsafe_allow_html=True)

# Import our fixed utilities and chat interface
from fixed_chat_settings import fixed_load_settings, fixed_save_settings
from enhanced_chat_interface import enhanced_chat_interface
from simplified_rag import enhanced_rag_interface
from multimodel_fix import fixed_multimodel_chat
from local_models_fix import fixed_list_local_models, apply_local_models_fix

# Apply UI fixes to the entire application
apply_all_fixes()

# Apply fix for local models
apply_local_models_fix()

# Import other required modules
from ui_elements import (
    model_comparison_test, feature_test, 
    pull_models, show_model_details, remove_model_ui,
    vision_comparison_test, update_models, files_tab,
    server_configuration, server_monitoring
)
# Note: We're not importing list_local_models as we have our fixed version
from welcome import display_welcome_message

# Make sure settings directory exists
os.makedirs("sessions", exist_ok=True)

# Load settings from file
if os.path.exists("chat-settings.json"):
    try:
        with open("chat-settings.json", "r") as f:
            settings = json.load(f)
            # Apply settings to session state
            for key, value in settings.items():
                if key not in st.session_state:
                    st.session_state[key] = value
    except Exception as e:
        logger.error(f"Error loading settings: {e}")

# Define sidebar sections
SIDEBAR_SECTIONS = {
    "Chat": [
        ("💬 Chat", "Chat"),
        ("🖼️ Vision Chat", "Multimodal Chat"),
        ("🔄 Multi-Model Chat", "Multi-Model Chat"),
        ("🎤 Voice Chat", "Voice Chat"),
    ],
    "Knowledge": [
        ("📚 RAG Interface", "Enhanced RAG"),
        ("📝 Workspace", "Collaborative Workspace"),
    ],
    "Testing": [
        ("🧪 Feature Test", "Feature Test"),
        ("🔍 Model Comparison", "Model Comparison"),
        ("👁️ Vision Test", "Vision"),
    ],
    "Models": [
        ("📋 List Models", "List Local Models"),
        ("⬇️ Pull Model", "Pull a Model"),
        ("ℹ️ Model Info", "Show Model Information"),
        ("❌ Remove Model", "Remove a Model"),
        ("🔄 Update Models", "Update Models"),
    ],
    "System": [
        ("⚙️ Server Config", "Server Configuration"),
        ("📊 Monitoring", "Server Monitoring"),
        ("❓ Help", "Help"),
    ]
}

def initialize_session_state():
    """Initialize session state variables for Ollama Workbench."""
    if 'selected_test' not in st.session_state:
        st.session_state.selected_test = "Chat"
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "agent_type" not in st.session_state:
        st.session_state.agent_type = "None"
    if "metacognitive_type" not in st.session_state:
        st.session_state.metacognitive_type = "None"
    if "voice_type" not in st.session_state:
        st.session_state.voice_type = "None"
    if "selected_corpus" not in st.session_state:
        st.session_state.selected_corpus = "None"
    
    # Initialize multimodel chat variables
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = {}
    elif not isinstance(st.session_state.total_tokens, dict):
        # Reset if not a dictionary (prevents TypeError)
        logger.warning(f"total_tokens was {type(st.session_state.total_tokens)}, resetting to dict")
        st.session_state.total_tokens = {}
    
    if "model_costs" not in st.session_state:
        st.session_state.model_costs = {}

def create_sidebar():
    """Create and populate the sidebar with improved selection handling."""
    with st.sidebar:
        st.markdown(
            '<div style="text-align: left;">'
            '<h1 class="logo" style="font-size: 24px; font-weight: 300;">🦙 Ollama <span style="color: orange;">Workbench</span></h1>'
            "</div>",
            unsafe_allow_html=True
        )
        
        # Log the current selected test for debugging
        logger.info(f"Current selected test: {st.session_state.get('selected_test')}")
        
        # Create tabs for main sections
        main_sections = list(SIDEBAR_SECTIONS.keys())
        selected_section = stable_selectbox(
            "Main Section:", 
            main_sections,
            key="selected_section", 
            index=0
        )
        
        # Show options for the selected section
        options = SIDEBAR_SECTIONS[selected_section]
        
        # Create buttons for each option in the section
        for label, value in options:
            if st.button(label, key=f"nav_{value}", use_container_width=True):
                # Update session state and log
                st.session_state.selected_test = value
                logger.info(f"Selected test: {value}")
                # Force refresh
                st.rerun()

def main_content():
    """Display main content based on selected test."""
    current_test = st.session_state.get("selected_test", "Chat")
    logger.info(f"Rendering main content for: {current_test}")
    
    if current_test == "Chat":
        # Use our enhanced chat interface with all fixes
        enhanced_chat_interface()
    elif current_test == "Enhanced RAG":
        enhanced_rag_interface()
    elif current_test == "Multi-Model Chat":
        # Use our fixed version of multimodel chat
        fixed_multimodel_chat()  
    elif current_test == "Feature Test":
        feature_test()
    elif current_test == "Model Comparison":
        model_comparison_test()
    elif current_test == "Vision":
        vision_comparison_test()
    elif current_test == "List Local Models":
        # Use our fixed version for local models
        fixed_list_local_models()
    elif current_test == "Show Model Information":
        show_model_details()
    elif current_test == "Pull a Model":
        pull_models()
    elif current_test == "Remove a Model":
        remove_model_ui()
    elif current_test == "Update Models":
        update_models()
    elif current_test == "Server Configuration":
        server_configuration()
    elif current_test == "Server Monitoring":
        server_monitoring()
    elif current_test == "Help":
        display_welcome_message()
    else:
        # Default to chat
        enhanced_chat_interface()

def main():
    """Main entry point for the application."""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Force load settings to ensure the model selection is remembered
        fixed_load_settings()
        
        # Create the sidebar navigation
        create_sidebar()
        
        # Display the main content
        main_content()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        st.error(f"An error occurred. Please try refreshing the page.")
    finally:
        # Restore original UI functions when done
        restore_original()

if __name__ == "__main__":
    main()