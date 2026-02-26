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
