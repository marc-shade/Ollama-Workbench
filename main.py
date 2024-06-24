# main.py
import os
import streamlit as st
from ollama_utils import *
from model_tests import *
from ui_elements import (
    model_comparison_test, contextual_response_test, feature_test,
    list_local_models, pull_models, show_model_details, remove_model_ui,
    vision_comparison_test, chat_interface, update_models, files_tab, manage_prompts,
    manage_corpus  # Add manage_corpus import
)
from repo_docs import main as repo_docs_main
from web_to_corpus import main as web_to_corpus_main
from streamlit_extras.buy_me_a_coffee import button

# Set page config for wide layout
st.set_page_config(layout="wide", page_title="Ollama Workbench", page_icon="🦙")

# Define constants
SIDEBAR_SECTIONS = {
    "Maintain": [
        ("List Local Models", "List Local Models"),
        ("Show Model Information", "Show Model Information"),
        ("Pull a Model", "Pull a Model"),
        ("Remove a Model", "Remove a Model"),
        ("Update Models", "Update Models"),
    ],
    "Test": [
        ("Model Feature Test", "Model Feature Test"),
        ("Model Comparison by Response Quality", "Model Comparison by Response Quality"),
        ("Contextual Response Test by Model", "Contextual Response Test by Model"),
        ("Vision Model Comparison", "Vision Model Comparison"),
    ],
    "Document": [
        ("Repository Analyzer", "Repository Analyzer"),
        ("Web to Corpus File", "Web to Corpus File"),
        ("Manage Files", "Files"),
        ("Manage Prompts", "Prompts"),
        ("Manage Corpus", "Manage Corpus"),  # Add Manage Corpus button
    ],
}

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'selected_test' not in st.session_state:
        st.session_state.selected_test = None
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

def create_sidebar():
    """Create and populate the sidebar."""
    with st.sidebar:
        st.markdown(
            '<div style="text-align: left;">'
            '<h1 class="logo" style="font-size: 50px;">🦙 Ollama <span style="color: orange;">Workbench</span></h1>'
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<style>div.row-widget.stButton > button {width:100%;}</style>', unsafe_allow_html=True)
        if st.button("Chat", key="button_chat"):
            st.session_state.selected_test = "Chat"

        for section, buttons in SIDEBAR_SECTIONS.items():
            with st.expander(section, expanded=False):
                for button_text, test_name in buttons:
                    if st.button(button_text, key=f"button_{test_name.lower().replace(' ', '_')}"):
                        st.session_state.selected_test = test_name

        # Add Buy Me a Coffee button to the bottom of the sidebar
        st.markdown("---")  # Add a separator
        st.markdown("If you find this tool useful, consider supporting its development:")
        button(
            username=os.getenv("BUYMEACOFFEE_USERNAME", "marcshade"),
            floating=False,
            width=221,
            text="Support Marc",
            emoji="☕",
            bg_color="#FF5F5F",
            font_color="#FFFFFF",
        )

def main_content():
    """Display the main content based on the selected test."""
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
        manage_prompts()
    elif st.session_state.selected_test == "Manage Corpus":  # Add condition for Manage Corpus
        manage_corpus()
    else:
        display_welcome_message()

def display_welcome_message():
    """Display the welcome message and feature overview."""
    st.write("""
        ### Welcome to the Ollama Workbench!
        This application provides tools for managing, testing, and interacting with your Ollama models.

        #### **Chat**
        Engage in a real-time chat with a selected model, enhanced with various features:
        - **Agent Types:** Choose from a variety of predefined agent types, each with specific prompts to guide the model's behavior (e.g., Coder, Analyst, Creative Writer).
        - **Metacognitive Types:** Enhance the model's reasoning abilities by selecting a metacognitive type (e.g., Visualization of Thought, Chain of Thought).
        - **Corpus Integration:** Load a corpus of text from the 'Files' section to provide contextual information to the model, improving its responses.
        - **Advanced Settings:** Fine-tune the model's output by adjusting parameters like temperature, max tokens, presence penalty, and frequency penalty.
        - **Workspace:** Save and manage code snippets and other text generated during your chat sessions.
        - **Save/Load Sessions:** Save your chat history and workspace for later use, or load previously saved sessions.

        #### **Maintain**
        - **List Local Models:** View a list of all locally available models, including their size and last modified date.
        - **Show Model Information:** Display detailed information about a selected model.
        - **Pull a Model:** Download a new model from the Ollama library.
        - **Remove a Model**: Delete a selected model from the local storage.
        - **Update Models**: Update all local models.

        #### **Test**
        - **Model Feature Test**: Test a model's capability to handle JSON and function calls.
        - **Model Comparison by Response Quality**: Compare the response quality and performance of multiple models for a given prompt.
        - **Contextual Response Test by Model**: Test how well a model maintains context across multiple prompts.
        - **Vision Model Comparison**: Compare the performance of vision models using the same test image.

        #### **Document**
        - **Repository Analyzer**: Analyze your Python repository, generate documentation, debug reports, or a README.md file.
        - **Web to Corpus File**: Convert web content into a corpus for analysis or training.
        - **Manage Files**: Upload, view, edit, and delete files.
        - **Manage Prompts**: Create, edit, and delete custom prompts for Agent Type and Metacognitive Type.
        - **Manage Corpus**: Create, edit, and delete corpus from files or URLs.
    """)

def main():
    initialize_session_state()
    create_sidebar()
    main_content()

if __name__ == "__main__":
    main()
