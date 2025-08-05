#!/usr/bin/env python3
"""
Quick fix for the MultiModel Chat module to ensure total_tokens is always a dictionary.
This resolves the 'int' is not iterable error that can occur in multimodel_chat.py.
"""

import streamlit as st
import logging
import os

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_multimodel_chat")

def fix_multimodel_chat_session_state():
    """Reset the total_tokens if it's not a dictionary."""
    try:
        # Initialize session state if needed
        if "total_tokens" not in st.session_state:
            st.session_state.total_tokens = {}
            logger.info("Created total_tokens as dictionary")
        elif not isinstance(st.session_state.total_tokens, dict):
            # If it's not a dictionary, reset it
            logger.warning(f"total_tokens was {type(st.session_state.total_tokens)}, resetting to dict")
            st.session_state.total_tokens = {}
        
        # Initialize other required session state variables
        if "model_costs" not in st.session_state:
            st.session_state.model_costs = {}
            
        if "multimodel_selected_models" not in st.session_state:
            st.session_state.multimodel_selected_models = []
            
        if "model_settings" not in st.session_state:
            st.session_state.model_settings = {}
        
        logger.info("Successfully fixed multimodel chat session state")
        return True
    except Exception as e:
        logger.error(f"Error fixing multimodel chat session state: {e}")
        return False

if __name__ == "__main__":
    success = fix_multimodel_chat_session_state()
    if success:
        print("Successfully fixed multimodel chat session state")
    else:
        print("Failed to fix multimodel chat session state")