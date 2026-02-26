"""
Test application for the Collaborative Workspace feature.

This is a standalone Streamlit application that only includes the collaborative workspace
functionality, making it easier to test without the entire Ollama Workbench.
"""

import streamlit as st
import os
import json
from datetime import datetime
import uuid
import logging
from pathlib import Path
from ollama_workbench.chat.collaborative_workspace import collaborative_workspace_ui

# Setup logging
logging.basicConfig(
    filename='workspace_test.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Set page config for wide layout
st.set_page_config(layout="wide", page_title="Collaborative Workspace Test", page_icon="🖋️")

def model_callback(prompt):
    """
    A simple model callback function that returns a response for testing.
    In production, this would call an actual AI model.
    """
    logger.info(f"Received prompt: {prompt[:100]}...")
    
    # Simple response for testing
    return f"""
    I've analyzed your workspace. Here are some suggestions based on your request: "{prompt[-100:]}..."
    
    ### BEGIN CONTENT ###
    # Improved Sample Content
    
    This is a sample markdown block that I'm suggesting as a replacement or new addition.
    
    - The collaborative workspace allows real-time editing
    - You can add different types of blocks (text, code, markdown, table)
    - Each block maintains version history for tracking changes
    ### END CONTENT ###
    
    I also think this code example might be helpful:
    
    ### BEGIN CONTENT ###
    ```python
    def sample_function():
        \"\"\"This is a sample function to demonstrate code blocks in the workspace.\"\"\"
        print("Welcome to the Collaborative Workspace!")
        
        # You can add code with syntax highlighting
        results = [x for x in range(10) if x % 2 == 0]
        return results
    ```
    ### END CONTENT ###
    
    Let me know if you'd like any other assistance with your workspace!
    """

# Main app
def main():
    st.title("Collaborative Workspace Test App")
    
    st.markdown("""
    This is a standalone test application for the Collaborative Workspace feature.
    It provides a simpler environment to test and validate the functionality before 
    integrating it fully with the Ollama Workbench.
    
    ## Features:
    - Add different types of blocks (text, code, markdown, table)
    - Edit blocks in real-time
    - Track version history
    - Save and load workspaces
    - Get AI assistance with workspace content
    
    Try it out below!
    """)
    
    # Initialize dummy session state 
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "dummy-model"
    if "selected_provider" not in st.session_state:
        st.session_state.selected_provider = "ollama"
    
    # Main workspace UI
    collaborative_workspace_ui(model_callback=model_callback)
    
    # Add some debugging info
    with st.expander("Debug Information", expanded=False):
        st.write("Session State Keys:")
        st.json({key: str(type(value)) for key, value in st.session_state.items()})

if __name__ == "__main__":
    main()