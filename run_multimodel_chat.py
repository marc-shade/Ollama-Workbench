#!/usr/bin/env python3
# run_multimodel_chat.py

import streamlit as st
from multimodel_chat import multimodel_chat_app

if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="Ollama Workbench - Multi-Model Chat",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Run the multi-model chat application
    multimodel_chat_app()