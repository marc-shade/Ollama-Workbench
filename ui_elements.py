# ui_elements.py
import streamlit as st
import re
from ollama_utils import *
from model_tests import *
from local_models import *
from model_comparison import *
from vision_comparison import *
from chat_interface import *
from brainstorm import *
from web_to_corpus import *
from contextual_response import *
from feature_test import *
from file_management import *
from corpus_management import *
from server_configuration import *
from server_monitoring import *
from pull_model import *
from show_model import *
from remove_model import *
from update_models import *

def manage_prompts_interface():
    manage_prompts()

def update_model_selection(selected_models, key):
    """Callback function to update session state during form submission."""
    st.session_state[key] = selected_models

def extract_code_blocks(text):
    # Simple regex to extract code blocks (text between triple backticks)
    code_blocks = re.findall(r'```[\s\S]*?```', text)
    # Remove the backticks
    return [block.strip('`').strip() for block in code_blocks]

def brainstorm_interface():
    brainstorm_session()

