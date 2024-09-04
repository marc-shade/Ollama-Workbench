# show_model.py
import streamlit as st
from ollama_utils import get_available_models, show_model_info  # Import show_model_info

def show_model_details():
    st.header("🦙 Show Ollama Model Information")
    
    # Refresh available_models list
    available_models = get_available_models()

    # Initialize selected_model in session state if it doesn't exist
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = available_models[0] if available_models else None

    # Use a separate key for the selectbox
    selectbox_key = "show_model_details_model_selector"

    # Update selected_model when selectbox changes
    if selectbox_key in st.session_state:
        st.session_state.selected_model = st.session_state[selectbox_key]

    selected_model = st.selectbox(
        "Select the model you want to show details for:", 
        available_models, 
        key=selectbox_key,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
    )

    if st.button("Show Model Information", key="show_model_information"):
        details = show_model_info(selected_model)
        st.json(details)