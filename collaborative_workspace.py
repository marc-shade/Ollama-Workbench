"""
Collaborative Workspace for Ollama Workbench

This module provides a Canvas-like collaborative workspace where both the user and AI can 
edit content in real-time, similar to the Canvas feature in ChatGPT and Claude.
"""

import streamlit as st
import time
from typing import Callable, Dict, Any
import logging
from canvas import DocumentState, canvas_ui

logger = logging.getLogger(__name__)

def collaborative_workspace_ui(model_callback: Callable = None):
    """
    Main UI for the collaborative workspace.
    
    Args:
        model_callback: Optional callback function for AI model integration
    """
    st.title("🖋️ Collaborative Workspace")
    st.write("Work collaboratively with AI to edit documents in real-time.")
    
    # Use the canvas_ui from the canvas module
    canvas_ui(model_callback)

if __name__ == "__main__":
    def dummy_model_callback(prompt):
        """Dummy callback for testing"""
        # In a real implementation, this would call a model API
        time.sleep(1)  # Simulate API call delay
        return f"""
        I've analyzed your document. Here are some suggestions:
        
        ### BEGIN CONTENT ###
        # Example Heading
        
        This is some example content that I'm suggesting as a new block.
        
        - Item 1
        - Item 2
        - Item 3
        ### END CONTENT ###
        
        I also think you should add this code:
        
        ### BEGIN CONTENT ###
        ```python
        def example_function():
            print("This is an example")
            
        example_function()
        ```
        ### END CONTENT ###
        """
    
    collaborative_workspace_ui(model_callback=dummy_model_callback)