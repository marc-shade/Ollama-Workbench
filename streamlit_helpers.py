"""
Helper functions for Streamlit UI in Ollama Workbench.

This module provides utility functions to fix common issues with Streamlit,
particularly around form elements and session state.
"""

import streamlit as st
import logging

# Setup logging
logging.basicConfig(
    filename='app.log', 
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("streamlit_helpers")

def fix_selection_sync(key, value):
    """
    Fix for Streamlit's selected values getting out of sync.
    Call this when you need to ensure a session state value is updated.
    """
    # Store the new value in session state
    prev_value = st.session_state.get(key)
    st.session_state[key] = value
    
    # Log the change for debugging
    if prev_value != value:
        logger.info(f"Selection sync: Updated {key} from '{prev_value}' to '{value}'")
    
    return value

def on_change_callback(key):
    """
    Create a callback function for on_change events.
    This helps with debugging session state changes.
    """
    def callback():
        logger.info(f"Changed {key} to {st.session_state[key]}")
    return callback

def stable_selectbox(label, options, key=None, index=0, help=None, value=None):
    """
    A more stable version of st.selectbox that works better with session state.
    
    This function ensures that:
    1. The selected value is correctly maintained in session state
    2. The widget displays the correct value from session state
    3. Changes are properly synchronized
    """
    # Make sure key is provided
    if key is None:
        key = label.lower().replace(" ", "_")
    
    # Get the current value from session state or use default
    if key in st.session_state:
        current_value = st.session_state[key]
        # Find index of current value in options
        try:
            if current_value in options:
                index = options.index(current_value)
        except ValueError:
            # If not found, use the provided index
            pass
    elif value is not None and value in options:
        # If a value is provided, use it
        index = options.index(value)
        # Store in session state
        st.session_state[key] = value
    
    # Create the selectbox
    result = st.selectbox(
        label=label,
        options=options,
        index=index, 
        key=key,
        help=help,
        on_change=on_change_callback(key)
    )
    
    # Ensure session state is updated
    st.session_state[key] = result
    
    return result

def safe_button(label, key=None, **kwargs):
    """
    A safer version of st.button that doesn't trigger reruns when clicked.
    """
    # Make sure key is provided
    if key is None:
        key = f"btn_{label.lower().replace(' ', '_')}"
    
    # Create the button
    result = st.button(label, key=key, **kwargs)
    
    # If clicked, store in session state
    if result:
        st.session_state[f"{key}_clicked"] = True
        logger.info(f"Button {label} clicked")
    
    # Return true if clicked in this session
    return result