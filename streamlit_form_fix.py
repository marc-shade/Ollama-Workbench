"""
Modern Streamlit solution for model selection issue in Ollama Workbench

This file provides a proper fix for the model selection issue by:
1. Using Streamlit forms to prevent auto-rerun on selection
2. Implementing modern session state handling
3. Properly synchronizing UI state with application state
"""

import streamlit as st
import json
import os
import logging

# Setup logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("streamlit_form_fix")

# Settings file path
SETTINGS_FILE = "chat-settings.json"

def fix_model_selection_ui(available_models, model_descriptions=None):
    """
    Fix model selection UI using proper Streamlit form pattern
    
    Args:
        available_models: List of available model names
        model_descriptions: Optional dict of model descriptions
    
    Returns:
        None
    """
    # Create a form to prevent auto-rerun
    with st.form(key="model_selection_form"):
        st.subheader("Model Selection")
        
        # Find the current model index
        model_index = 0
        try:
            if st.session_state.get("selected_model") in available_models:
                model_index = available_models.index(st.session_state.get("selected_model"))
        except Exception as e:
            logger.error(f"Error finding model index: {e}")
        
        # Show model selector with descriptions
        model_choice = st.selectbox(
            "📦 Model:",
            available_models,
            index=model_index,
            help=model_descriptions.get(st.session_state.get("selected_model"), "An Ollama model") if model_descriptions else None,
        )
        
        # Form submit button
        submit_button = st.form_submit_button(label="Save Model Selection")
        
        # Only update on form submission
        if submit_button:
            # Update the actual model
            previous_model = st.session_state.get("selected_model")
            st.session_state.selected_model = model_choice
            
            # Save to settings file
            save_model_to_settings(model_choice)
            
            # Show success
            st.success(f"Model changed from '{previous_model}' to '{model_choice}'")
            logger.info(f"Model changed from '{previous_model}' to '{model_choice}'")
            
            # The form prevents auto-rerun, so we need to do it manually
            st.rerun()

def save_model_to_settings(model_name):
    """
    Save model selection to settings file
    
    Args:
        model_name: The selected model name
    
    Returns:
        bool: True if successful
    """
    try:
        # Load existing settings or create new
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
        else:
            settings = {}
        
        # Update model selection
        settings["selected_model"] = model_name
        
        # Write settings to file
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
        
        logger.info(f"Model selection saved to settings: {model_name}")
        return True
    except Exception as e:
        logger.error(f"Error saving model to settings: {e}")
        return False

def get_current_model():
    """
    Get the currently selected model from settings
    
    Returns:
        str: The current model name or None
    """
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
            return settings.get("selected_model")
        return None
    except Exception as e:
        logger.error(f"Error reading model from settings: {e}")
        return None