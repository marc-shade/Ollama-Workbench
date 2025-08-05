"""
Fixes for the Multi-Model Chat feature in Ollama Workbench.

This module patches the multimodel_chat.py to fix issues with model selection
and ensure form elements properly retain their values.
"""

import streamlit as st
import json
import os
import logging
import time
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("multimodel_fix")

# Settings file
MULTIMODEL_SETTINGS_FILE = "multimodel-chat-settings.json"

def load_multimodel_settings():
    """Load Multi-Model Chat settings from file."""
    # Ensure settings file exists
    if not os.path.exists(MULTIMODEL_SETTINGS_FILE):
        save_multimodel_settings({
            "multimodel_selected_models": []
        })
    
    try:
        with open(MULTIMODEL_SETTINGS_FILE, "r") as f:
            settings = json.load(f)
        
        # Update session state
        for key, value in settings.items():
            st.session_state[key] = value
        
        logger.info(f"Loaded multimodel settings: {settings}")
        return settings
    except Exception as e:
        logger.error(f"Error loading multimodel settings: {e}")
        return {}

def save_multimodel_settings(settings=None):
    """Save Multi-Model Chat settings to file."""
    if settings is None:
        # Get settings from session state
        settings = {
            "multimodel_selected_models": st.session_state.get("multimodel_selected_models", [])
        }
    
    try:
        with open(MULTIMODEL_SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
        
        logger.info(f"Saved multimodel settings: {settings}")
        return True
    except Exception as e:
        logger.error(f"Error saving multimodel settings: {e}")
        return False

def fixed_multimodel_chat():
    """
    Wrap the original multimodel_chat_app function with fixes for dropdowns.
    
    This function should be imported and used in place of multimodel_chat_app.
    """
    # Make sure session state is initialized
    if "multimodel_selected_models" not in st.session_state:
        st.session_state.multimodel_selected_models = []
    
    # Make sure total_tokens is a dictionary
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = {}
    elif not isinstance(st.session_state.total_tokens, dict):
        logger.warning(f"total_tokens was {type(st.session_state.total_tokens)}, resetting to dict")
        st.session_state.total_tokens = {}
    
    # Make sure model_costs is a dictionary    
    if "model_costs" not in st.session_state:
        st.session_state.model_costs = {}
    elif not isinstance(st.session_state.model_costs, dict):
        logger.warning(f"model_costs was {type(st.session_state.model_costs)}, resetting to dict")
        st.session_state.model_costs = {}
    
    # Load settings
    load_multimodel_settings()
    
    # Add CSS to fix multiselect
    st.markdown("""
        <style>
        /* Fix for multiselect dropdowns */
        div[data-baseweb="select"] {
            min-width: 300px !important; 
        }
        
        /* Make sure dropdown options don't get cut off */
        div[role="listbox"] {
            z-index: 999 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Store original multiselect function
    original_multiselect = st.multiselect
    
    def fixed_model_multiselect(*args, **kwargs):
        """Fixed version of multiselect for model selection."""
        label = args[0] if args else kwargs.get("label", "")
        
        if label == "Select Models for Comparison":
            # Get options and current selection
            options = args[1] if len(args) > 1 else kwargs.get("options", [])
            current_selection = st.session_state.get("multimodel_selected_models", [])
            
            # Filter current selection to only include valid options
            valid_selection = [model for model in current_selection if model in options]
            
            # Set default if valid_selection is empty
            # We don't force a default selection - user should be able to select multiple models
            # if not valid_selection and options:
            #     valid_selection = [options[0]]
            
            # Update session state with valid selection
            st.session_state.multimodel_selected_models = valid_selection
            
            # Set default value for the multiselect
            kwargs["default"] = valid_selection
            
            # Call original function with fixed arguments
            result = original_multiselect(*args, **kwargs)
            
            # Update session state with result
            if result != st.session_state.multimodel_selected_models:
                st.session_state.multimodel_selected_models = result
                # Save settings when selection changes
                save_multimodel_settings()
                logger.info(f"Updated multimodel_selected_models: {result}")
            
            return result
        else:
            # For other multiselects, use original behavior
            return original_multiselect(*args, **kwargs)
    
    # Replace multiselect with our fixed version
    st.multiselect = fixed_model_multiselect
    
    try:
        # Import and run the original multimodel_chat_app
        from multimodel_chat import multimodel_chat_app
        multimodel_chat_app()
    finally:
        # Restore original function
        st.multiselect = original_multiselect