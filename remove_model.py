# remove_model.py
import streamlit as st
from ollama_utils import *

def remove_model_ui():
    st.header("🗑️ Remove an Ollama Model")
    
    # Refresh available_models list
    available_models = get_available_models()

    # Initialize selected_model in session state if it doesn't exist
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = available_models[0] if available_models else None

    # Use a separate key for the selectbox
    selectbox_key = "remove_model_ui_model_selector"

    # Update selected_model when selectbox changes
    if selectbox_key in st.session_state:
        st.session_state.selected_model = st.session_state[selectbox_key]

    selected_model = st.selectbox(
        "Select the model you want to remove:", 
        available_models, 
        key=selectbox_key,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
    )

    confirm_label = f"❌ Confirm removal of model `{selected_model}`"
    confirm = st.checkbox(confirm_label)
    if st.button("Remove Model", key="remove_model") and confirm:
        if selected_model:
            result = remove_model(selected_model)
            st.write(result["message"])

            # Clear the cache of get_available_models
            get_available_models.clear()

            # Update the list of available models
            st.session_state.available_models = get_available_models()
            # Update selected_model if it was removed
            if selected_model not in st.session_state.available_models:
                st.session_state.selected_model = st.session_state.available_models[0] if st.session_state.available_models else None
            st.rerun()
        else:
            st.error("Please select a model.")