# main.py
import streamlit as st
from ollama_utils import *
from model_tests import *
from ui_elements import model_comparison_test, contextual_response_test, feature_test, list_local_models, pull_models, show_model_details, remove_model_ui, vision_comparison_test, chat_interface, update_models
from repo_docs import main as repo_docs_main

def main():
    # Initialize session state variables if they don't exist
    if 'selected_test' not in st.session_state:
        st.session_state.selected_test = None
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    with st.sidebar:
        st.markdown(
            '<div style="text-align: left;">'
            '<h1 class="logo" style="font-size: 50px;">🦙 Ollama <span style="color: orange;">Workbench</span></h1>'
            "</div>",
            unsafe_allow_html=True,
        )

        st.subheader("Chat")
        st.markdown('<style>div.row-widget.stButton > button {width:100%;}</style>', unsafe_allow_html=True)
        if st.button("Chat", key="button_chat"):
            st.session_state.selected_test = "Chat"

        # Maintain Section (Collapsible)
        with st.expander("Maintain", expanded=False):
            st.markdown('<style>div.row-widget.stButton > button {width:100%;}</style>', unsafe_allow_html=True)
            if st.button("List Local Models", key="button_list_models"):
                st.session_state.selected_test = "List Local Models"
            if st.button("Show Model Information", key="button_show_model_info"):
                st.session_state.selected_test = "Show Model Information"
            if st.button("Pull a Model", key="button_pull_model"):
                st.session_state.selected_test = "Pull a Model"
            if st.button("Remove a Model", key="button_remove_model"):
                st.session_state.selected_test = "Remove a Model"
            if st.button("Update Models", key="button_update_models"):
                st.session_state.selected_test = "Update Models"

        # Test Section (Collapsible)
        with st.expander("Test", expanded=False):
            st.markdown('<style>div.row-widget.stButton > button {width:100%;}</style>', unsafe_allow_html=True)
            if st.button("Model Feature Test", key="button_feature_test"):
                st.session_state.selected_test = "Model Feature Test"
            if st.button("Model Comparison by Response Quality", key="button_model_comparison"):
                st.session_state.selected_test = "Model Comparison by Response Quality"
            if st.button("Contextual Response Test by Model", key="button_contextual_response"):
                st.session_state.selected_test = "Contextual Response Test by Model"
            if st.button("Vision Model Comparison", key="button_vision_model_comparison"):
                st.session_state.selected_test = "Vision Model Comparison"

        # Document Section (Collapsible)
        with st.expander("Document", expanded=False):
            st.markdown('<style>div.row-widget.stButton > button {width:100%;}</style>', unsafe_allow_html=True)
            if st.button("Repository Analyzer", key="button_repo_analyzer"):
                st.session_state.selected_test = "Repository Analyzer"

    # Main content area based on selected_test
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
    else:
        st.write("""
            ### Welcome to the Ollama Workbench!
            This application provides tools for managing and testing your Ollama models.

            #### **Chat**
            Engage in a real-time chat with a selected model.

            #### **Maintain**
            - **List Local Models**: View a list of all locally available models, including their size and last modified date.
            - **Show Model Information**: Display detailed information about a selected model.
            - **Pull a Model**: Download a new model from the Ollama library.
            - **Remove a Model**: Delete a selected model from the local storage.
            - **Update Models**: Update all local models.

            #### **Test**
            - **Model Feature Test**: Test a model's capability to handle JSON and function calls.
            - **Model Comparison by Response Quality**: Compare the response quality and performance of multiple models for a given prompt.
            - **Contextual Response Test by Model**: Test how well a model maintains context across multiple prompts.
            - **Vision Model Comparison**: Compare the performance of vision models using the same test image.

            #### **Document**
            - **Repository Analyzer**: Analyze your Python repository, generate documentation, debug reports, or a README.md file.
        """)

if __name__ == "__main__":
    main()
