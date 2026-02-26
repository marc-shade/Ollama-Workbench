import streamlit as st
import time
from ollama_workbench.chat.multimodal_chat import multimodal_chat_interface

# Set page config
st.set_page_config(page_title="Test Multimodal", layout="wide")

# Run the multimodal interface
multimodal_chat_interface()