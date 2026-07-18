"""
Enhanced Chat Interface for Ollama Workbench

This module preserves all the original chat_interface.py functionality
while adding modern Open WebUI-inspired styling improvements.
"""

import streamlit as st
import logging

logger = logging.getLogger("enhanced_chat_interface")

# Import original chat_interface.py functionality
from .chat_interface import chat_interface, load_settings


def apply_modern_styling():
    """Layout-only polish for the chat view.

    Colors, fonts, radii, and the sidebar palette come from the app theme
    in .streamlit/config.toml. The old hardcoded light-mode CSS here
    (forced #F9FAFB sidebar, blue user bubbles, light code blocks with a
    prefers-color-scheme dark variant) fought the theme system - keep this
    block free of color values.
    """
    st.markdown("""
        <style>
        /* Full-width chat layout */
        .block-container {
            max-width: 100% !important;
            padding: 1rem 1rem !important;
        }

        /* Subtle depth on chat bubbles; colors come from the theme */
        .stChatMessage {
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
        }
        </style>
    """, unsafe_allow_html=True)

def enhanced_chat_interface():
    """
    Enhanced chat interface that preserves original functionality with improved styling.
    This is a wrapper around the original chat_interface() function.

    The underlying chat_interface() selectboxes now handle their own index
    computation from session state, so no monkey-patching is needed.
    """
    # Apply modern styling
    apply_modern_styling()

    # Ensure session state is loaded properly
    load_settings()

    # Ensure chat_interface function knows it's being run in enhanced mode
    if "enhanced_mode" not in st.session_state:
        st.session_state.enhanced_mode = True

    # Run the original chat interface with all functionality preserved
    chat_interface()

if __name__ == "__main__":
    enhanced_chat_interface()
