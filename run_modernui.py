"""
Modern UI Test for Ollama Workbench

This script provides a simplified way to test the modern UI changes without running the full application.
"""

import streamlit as st
from modern_chat_interface import modern_chat_interface
from styles import apply_styles
import os
import sys

# Set page config for wide layout
st.set_page_config(layout="wide", page_title="Ollama Workbench - Modern UI Test", page_icon="🦙")

# Apply the new modern styling
colors, theme = apply_styles()

def main():
    # Header
    st.markdown(
        '<div style="text-align: center; margin-bottom: 1rem;">'
        '<h1 style="font-size: 1.5rem; font-weight: 600;">Ollama Workbench - Modern UI Test</h1>'
        '<p style="font-size: 0.9rem; opacity: 0.8;">Testing the new modern UI design based on Open WebUI</p>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Display test mode warning
    st.info("This is a test version of the modern UI. Some features may not work as expected.")
    
    # Run the modern chat interface
    modern_chat_interface()

if __name__ == "__main__":
    main()