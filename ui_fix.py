"""
UI Fix Module for Ollama Workbench

This module provides fixes for Streamlit UI components across the entire application,
particularly focusing on form elements that don't retain selections.
"""

import streamlit as st
import logging
import functools
import os
import json
import time
from typing import List, Dict, Any, Callable, Optional, Union, Tuple

# Setup logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ui_fix")

# Original widget references
_original_selectbox = st.selectbox
_original_multiselect = st.multiselect
_original_slider = st.slider
_original_checkbox = st.checkbox
_original_radio = st.radio

class UIFixer:
    """Class to manage UI fixes and patches for Streamlit elements."""
    
    @staticmethod
    def init():
        """Initialize UI fixes for the entire application."""
        logger.info("Initializing UI fixes")
        
        # Apply monkey patches
        st.selectbox = UIFixer.fixed_selectbox
        st.multiselect = UIFixer.fixed_multiselect
        st.slider = UIFixer.fixed_slider
        st.checkbox = UIFixer.fixed_checkbox
        st.radio = UIFixer.fixed_radio
        
        # Add helper CSS
        UIFixer.inject_helper_css()
        
        return True
    
    @staticmethod
    def restore():
        """Restore original Streamlit UI elements."""
        logger.info("Restoring original UI elements")
        
        # Restore original functions
        st.selectbox = _original_selectbox
        st.multiselect = _original_multiselect
        st.slider = _original_slider
        st.checkbox = _original_checkbox
        st.radio = _original_radio
        
        return True
    
    @staticmethod
    def inject_helper_css():
        """Inject helper CSS to fix UI issues."""
        st.markdown("""
            <style>
            /* Fix for selectbox styling */
            div[data-testid="stSelectbox"] {
                min-width: 200px !important;
            }
            
            /* Fix for multiselect styling */
            div[data-testid="stMultiSelect"] {
                min-width: 200px !important;
            }
            
            /* Prevent container reflow */
            .stButton button {
                width: 100% !important;
            }
            
            /* Fix checkbox alignment */
            .stCheckbox label p {
                display: inline-block !important;
            }
            
            /* Fix selectbox display */
            div[data-baseweb="select"] > div:first-child {
                min-height: 36px !important;
            }
            </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def fixed_selectbox(*args, **kwargs):
        """Fixed version of st.selectbox that properly maintains selections."""
        key = kwargs.get("key")
        
        # Handle special cases that need extra care
        critical_keys = [
            "selected_model", "selected_models", "agent_type", 
            "metacognitive_type", "voice_type", "selected_corpus", 
            "multimodal_provider", "model_name", "tool_model_selector"
        ]
        
        # Get options and find the index for the current value
        options = kwargs.get("options", args[1] if len(args) > 1 else [])
        
        if key in critical_keys or any(k in str(key) for k in ["model", "agent"]):
            # Save current value 
            current_value = st.session_state.get(key)
            
            # Make sure index is set correctly if value exists
            if current_value is not None and current_value in options:
                kwargs["index"] = options.index(current_value)
            
            # Debug log
            logger.debug(f"Fixed selectbox for {key}: current={current_value}, options={options[:5] if len(options) > 5 else options}...")
            
            # Call original with our modified arguments
            result = _original_selectbox(*args, **kwargs)
            
            # Make sure session state is updated
            if result != current_value:
                st.session_state[key] = result
                logger.info(f"Updated {key}: {current_value} -> {result}")
                
            return result
        else:
            # For non-critical keys, use original behavior
            return _original_selectbox(*args, **kwargs)
    
    @staticmethod
    def fixed_multiselect(*args, **kwargs):
        """Fixed version of st.multiselect that properly maintains selections."""
        key = kwargs.get("key")
        
        # Special handling for model selection
        if key in ["model_selection", "selected_models"] or "model" in str(key):
            current_values = st.session_state.get(key, [])
            
            # Debug log
            logger.debug(f"Fixed multiselect for {key}: current={current_values}")
            
            # Call original
            result = _original_multiselect(*args, **kwargs)
            
            # Update session state if changed
            if result != current_values:
                st.session_state[key] = result
                logger.info(f"Updated {key} multiselect: {len(current_values)} -> {len(result)} items")
            
            return result
        else:
            # For other keys, use original behavior
            return _original_multiselect(*args, **kwargs)
    
    @staticmethod
    def fixed_slider(*args, **kwargs):
        """Fixed version of st.slider that properly retains values."""
        key = kwargs.get("key")
        
        if key and ("temperature" in key or "tokens" in key or "penalty" in key):
            current_value = st.session_state.get(key)
            
            # Call original
            result = _original_slider(*args, **kwargs)
            
            # Update session state if changed
            if current_value is not None and result != current_value:
                st.session_state[key] = result
                logger.info(f"Updated {key} slider: {current_value} -> {result}")
            
            return result
        else:
            # For other sliders, use original behavior
            return _original_slider(*args, **kwargs)
    
    @staticmethod
    def fixed_checkbox(*args, **kwargs):
        """Fixed version of st.checkbox that properly retains state."""
        key = kwargs.get("key")
        
        if key and any(k in key for k in ["enabled", "memory", "thinking"]):
            # Get current value
            current_value = st.session_state.get(key, kwargs.get("value", False))
            
            # Make sure value is set in kwargs
            kwargs["value"] = current_value
            
            # Call original
            result = _original_checkbox(*args, **kwargs)
            
            # Update session state if changed
            if result != current_value:
                st.session_state[key] = result
                logger.info(f"Updated {key} checkbox: {current_value} -> {result}")
            
            return result
        else:
            # For other checkboxes, use original behavior
            return _original_checkbox(*args, **kwargs)
    
    @staticmethod
    def fixed_radio(*args, **kwargs):
        """Fixed version of st.radio that properly retains selections."""
        key = kwargs.get("key")
        
        if key and any(k in key for k in ["strategy", "type", "model"]):
            # Get current value
            current_value = st.session_state.get(key)
            
            # Get options and find the index for the current value
            options = kwargs.get("options", args[1] if len(args) > 1 else [])
            
            if current_value is not None and current_value in options:
                kwargs["index"] = options.index(current_value)
            
            # Call original
            result = _original_radio(*args, **kwargs)
            
            # Update session state if changed
            if current_value is not None and result != current_value:
                st.session_state[key] = result
                logger.info(f"Updated {key} radio: {current_value} -> {result}")
            
            return result
        else:
            # For other radio buttons, use original behavior
            return _original_radio(*args, **kwargs)

# Helper functions for specific components

def fix_model_selection(models):
    """
    Fix model selection issues for various forms and components.
    
    This ensures the correct models are loaded and displayed in selection widgets.
    """
    # Update available models in session state
    if "available_models" not in st.session_state or not st.session_state.available_models:
        st.session_state.available_models = models
    
    # Make sure "selected_model" exists and has a valid value
    if "selected_model" not in st.session_state or st.session_state.selected_model not in models:
        # Check if there's a value in chat-settings.json
        if os.path.exists("chat-settings.json"):
            try:
                with open("chat-settings.json", "r") as f:
                    settings = json.load(f)
                    model = settings.get("selected_model")
                    if model in models:
                        st.session_state.selected_model = model
                    else:
                        st.session_state.selected_model = models[0] if models else None
            except:
                st.session_state.selected_model = models[0] if models else None
        else:
            st.session_state.selected_model = models[0] if models else None
    
    return models

def fix_multimodel_chat():
    """Fix issues specific to the multimodel_chat.py module."""
    logger.info("Applying fixes for multimodel_chat.py")
    
    # Fix session state for model selection
    if "multimodel_selected_models" not in st.session_state:
        st.session_state.multimodel_selected_models = []
    
    # Ensure model list is populated and settings are stored
    if "multimodel_settings_file" not in st.session_state:
        st.session_state.multimodel_settings_file = "multimodel-chat-settings.json"
    
    # Force model selection to save properly
    old_multiselect = st.multiselect
    
    def fixed_multimodel_multiselect(*args, **kwargs):
        if args and args[0] == "Select Models for Comparison":
            # Get current values
            current_models = st.session_state.get("multimodel_selected_models", [])
            
            # Call original
            models = old_multiselect(*args, **kwargs)
            
            # Update session state
            st.session_state.multimodel_selected_models = models
            
            # Debug log the change
            if models != current_models:
                logger.info(f"Updated multimodel_selected_models: {current_models} -> {models}")
            
            return models
        else:
            return old_multiselect(*args, **kwargs)
    
    # Apply the patch
    st.multiselect = fixed_multimodel_multiselect
    
    return True

def fix_local_models_listing():
    """Fix issues with the local_models.py module."""
    logger.info("Applying fixes for local_models.py")
    
    # Monkey patch the get_ollama_models function to ensure it works
    try:
        from ollama_utils import get_ollama_models, get_ollama_client
        import requests
        
        original_get_ollama_models = get_ollama_models
        
        @functools.wraps(original_get_ollama_models)
        def fixed_get_ollama_models():
            """Fixed version that ensures models are properly retrieved."""
            try:
                # Try to get models from the original function
                models = original_get_ollama_models()
                
                # If models is empty, try alternate method
                if not models:
                    # Try to use the client directly
                    client = get_ollama_client()
                    if client:
                        try:
                            models_list = client.list()
                            models = models_list.get("models", [])
                        except:
                            pass
                
                # If still empty, try direct API call
                if not models:
                    try:
                        response = requests.get("http://localhost:11434/api/tags")
                        if response.status_code == 200:
                            models = response.json().get("models", [])
                    except:
                        pass
                
                # Last resort fallback
                if not models:
                    logger.warning("Could not fetch models, using fallback")
                    # Return a fallback model list
                    return [
                        {"name": "llama2", "model": "llama2", "modified_at": "2023-01-01T00:00:00Z", "size": 0},
                        {"name": "gemma", "model": "gemma", "modified_at": "2023-01-01T00:00:00Z", "size": 0}
                    ]
                
                return models
            except Exception as e:
                logger.error(f"Error in fixed_get_ollama_models: {e}")
                # Return a minimal fallback
                return [{"name": "llama2", "model": "llama2", "modified_at": "2023-01-01T00:00:00Z", "size": 0}]
        
        # Apply the patch
        import ollama_utils
        ollama_utils.get_ollama_models = fixed_get_ollama_models
    except ImportError:
        logger.error("Could not import ollama_utils for patching")
    
    return True

# Function to apply all fixes
def apply_all_fixes():
    """Apply all UI fixes for the entire application."""
    # Initialize UI fixer
    UIFixer.init()
    
    # Apply specific module fixes
    fix_multimodel_chat()
    fix_local_models_listing()
    
    # Log success
    logger.info("All UI fixes applied successfully")
    
    return True

# Restore original functions
def restore_original():
    """Restore original Streamlit functions."""
    UIFixer.restore()
    
    # Log
    logger.info("Restored original UI functions")
    
    return True

# Helper function to create a stable selectbox
def stable_selectbox(label, options, key=None, index=0):
    """A selectbox implementation that retains its selection state."""
    if key and key in st.session_state:
        current_value = st.session_state[key]
        if current_value in options:
            index = options.index(current_value)
    
    result = st.selectbox(label, options, index=index, key=key)
    
    if key:
        st.session_state[key] = result
    
    return result