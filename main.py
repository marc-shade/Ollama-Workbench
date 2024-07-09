# main.py
import os
import json
import queue
import streamlit as st
from ollama_utils import *
from model_tests import *
from ui_elements import (
    model_comparison_test, contextual_response_test, feature_test,
    list_local_models, pull_models, show_model_details, remove_model_ui,
    vision_comparison_test, chat_interface, update_models, files_tab,
    server_configuration, server_monitoring
)
from repo_docs import main as repo_docs_main
from web_to_corpus import main as web_to_corpus_main
from streamlit_extras.buy_me_a_coffee import button
from welcome import display_welcome_message
from projects import projects_main, Task
import threading
import pandas as pd
import time
from visjs_component import visjs_graph
from datetime import datetime, timedelta
from prompts import manage_prompts
from brainstorm import brainstorm_interface
from manage_corpus import manage_corpus
from ollama_utils import get_ollama_resource_usage

# Set page config for wide layout
st.set_page_config(layout="wide", page_title="Ollama Workbench", page_icon="🦙")

# Define constants
SIDEBAR_SECTIONS = {
    "⚙️ Workflow": [
        ("🧠 Brainstorm", "Brainstorm"),
        ("🚀 Projects", "Manage Projects"),
        ("✨ Prompts", "Prompts"),
    ],
    "🗄 Document": [
        ("🗂 Manage Corpus", "Manage Corpus"),
        ("📂 Manage Files", "Files"),
        ("🕸️ Web to Corpus File", "Web to Corpus File"),
        ("✔️ Repository Analyzer", "Repository Analyzer"),
    ],
    "🛠️ Maintain": [
        ("📋 List Local Models", "List Local Models"),
        ("🦙 Show Model Information", "Show Model Information"),
        ("⬇ Pull a Model", "Pull a Model"),
        ("🗑️ Remove a Model", "Remove a Model"),
        ("🔄 Update Models", "Update Models"),
        ("⚙️ Server Configuration", "Server Configuration"),
        ("🖥️ Server Monitoring", "Server Monitoring"),
    ],
    "📊 Test": [
        ("🧪 Model Feature Test", "Model Feature Test"),
        ("🎯 Model Comparison by Response Quality", "Model Comparison by Response Quality"),
        ("💬 Contextual Response Test by Model", "Contextual Response Test by Model"),
        ("👁️ Vision Model Comparison", "Vision Model Comparison"),
    ],
}


def check_secret_key(file_path, expected_key):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data.get('secret_key') == expected_key
    return False


def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'selected_test' not in st.session_state:
        st.session_state.selected_test = None
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if 'bm_tasks' not in st.session_state:
        st.session_state.bm_tasks = []

def display_resource_usage_sidebar():
    """Displays resource usage in the sidebar."""
    usage = get_ollama_resource_usage()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"🚦: {usage['status']}")
    with col2:
        st.markdown(f"CPU: {usage['cpu_usage']}")
    with col3:
        st.markdown(f"RAM: {usage['memory_usage']}")
    with col4:
        st.markdown(f": {usage['gpu_usage']}")

def server_monitoring():
    st.header("🖥️ Server Monitoring")

    # Add checkbox to enable/disable resource usage display
    show_resource_usage = st.checkbox("Show Resource Usage in Sidebar", value=st.session_state.get("show_resource_usage", False))
    st.session_state.show_resource_usage = show_resource_usage

    # Resource Usage
    st.subheader("Resource Usage")
    usage = get_ollama_resource_usage()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", usage["status"])
    col2.metric("CPU Usage", usage["cpu_usage"])
    col3.metric("Memory Usage", usage["memory_usage"])
    col4.metric("GPU Usage", usage["gpu_usage"])

def create_sidebar():
    """Create and populate the sidebar."""
    with st.sidebar:
        st.markdown(
            '<div style="text-align: left;">'
            '<h1 class="logo" style="font-size: 50px;">🦙 Ollama <span style="color: orange;">Workbench</span></h1>'
            "</div>",
            unsafe_allow_html=True,
        )

        # Display resource usage if enabled
        if st.session_state.get("show_resource_usage", False):
            display_resource_usage_sidebar()

        st.markdown('<style>div.row-widget.stButton > button {width:100%;}</style>', unsafe_allow_html=True)
        if st.button("💬 Chat", key="sidebar_button_chat"):
            st.session_state.selected_test = "Chat"

        for section, buttons in SIDEBAR_SECTIONS.items():
            with st.expander(section, expanded=False):
                for button_text, test_name in buttons:
                    if st.button(button_text, key=f"sidebar_button_{test_name.lower().replace(' ', '_')}"):
                        st.session_state.selected_test = test_name

        # Check if the secret key JSON file exists and has the correct key
        secret_key_file = 'secret_key_off.json'
        secret_key_value = 'I_am_an_honest_person'
        if not check_secret_key(secret_key_file, secret_key_value):
            # Add Buy Me a Coffee button and image in a 2-column layout
            st.markdown("---")  # Add a separator

            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(
                    '<a href="https://github.com/marc-shade" target="_blank"><img src="https://2acrestudios.com/wp-content/uploads/2024/06/marc-cyberpunk.png" '
                    'style="border-radius: 50%; max-width: 70px; object-fit: cover;" /></a>',
                    unsafe_allow_html=True,
                )
            with col2:
                button(
                    username=os.getenv("BUYMEACOFFEE_USERNAME", "marcshade"),
                    floating=False,
                    text="Support Marc",
                    emoji="☕",
                    bg_color="#FF5F5F",
                    font_color="#FFFFFF",
                )
            st.markdown('<span style="font-size:17px; font-weight:normal; font-family:Courier;">Find this tool useful? Your support means a lot! Give a donation of $10 or more to remove this notice.</span>',
                    unsafe_allow_html=True,)

# Callback function to update task status in session state
def update_task_status(task_index, status, result=None):
    if task_index < len(st.session_state.bm_tasks):
        st.session_state.bm_tasks[task_index]["status"] = status
        if result is not None:
            st.session_state.bm_tasks[task_index]["result"] = result

def handle_user_input(step, task_data):
    """Handles user input for a specific task step."""
    user_input_config = step.get("user_input")
    if user_input_config:
        input_type = user_input_config["type"]
        prompt = user_input_config["prompt"]

        if input_type == "file_path":
            file_path = st.text_input(prompt, key=f"user_input_{step['agent']}")
            if file_path:
                task_data["file_path"] = file_path
            else:
                st.warning("Please provide a file path.")
                return False  # Indicate that user input is not complete

        elif input_type == "options":
            options = user_input_config.get("options", [])
            selected_option = st.selectbox(prompt, options, key=f"user_input_{step['agent']}")
            if selected_option:
                task_data["selected_option"] = selected_option
            else:
                st.warning("Please select an option.")
                return False  # Indicate that user input is not complete

        elif input_type == "confirmation":
            if not st.button(prompt, key=f"user_input_{step['agent']}"):
                st.warning("Task skipped due to unconfirmed user input.")
                return False  # Indicate that user input is not complete

    return True  # Indicate that user input is complete

def main_content():
    if 'bm_tasks' not in st.session_state:
        st.session_state.bm_tasks = []
    if st.session_state.selected_test == "Model Comparison by Response Quality":
        model_comparison_test()
    elif st.session_state.selected_test == "Contextual Response Test by Model":
        contextual_response_test()
    elif st.session_state.selected_test == "Model Feature Test":
        feature_test()
    elif st.session_state.selected_test == "List Local Models":
        list_local_models()
    elif st.session_state.selected_test == "Pull a Model":
        pull_models()
    elif st.session_state.selected_test == "Show Model Information":
        show_model_details()
    elif st.session_state.selected_test == "Remove a Model":
        remove_model_ui()
    elif st.session_state.selected_test == "Vision Model Comparison":
        vision_comparison_test()
    elif st.session_state.selected_test == "Chat":
        chat_interface()
    elif st.session_state.selected_test == "Update Models":
        update_models()
    elif st.session_state.selected_test == "Repository Analyzer":
        repo_docs_main()
    elif st.session_state.selected_test == "Web to Corpus File":
        web_to_corpus_main()
    elif st.session_state.selected_test == "Files":
        files_tab()
    elif st.session_state.selected_test == "Prompts":
        manage_prompts()  # Call the manage_prompts function directly
    elif st.session_state.selected_test == "Manage Corpus":
        manage_corpus()
    elif st.session_state.selected_test == "Manage Projects":
        projects_main()
    elif st.session_state.selected_test == "Brainstorm":
        brainstorm_interface()  # Call the brainstorm_interface function
    elif st.session_state.selected_test == "Server Configuration":
        server_configuration()
    elif st.session_state.selected_test == "Server Monitoring":
        server_monitoring()
    else:
        display_welcome_message()

def main():
    initialize_session_state()
    create_sidebar()
    main_content()

if __name__ == "__main__":
    main()