# remove_model.py
import streamlit as st
from ollama_workbench.providers.ollama_utils import *

def remove_model_ui():
    st.header("🗑️ Remove an Ollama Model")
    
    # Clear cache to ensure fresh results
    get_available_models.clear()
    
    # Refresh available_models list
    available_models = get_available_models()

    if not available_models:
        st.warning("No Ollama models found. Please install some models first.")
        return

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

    # Show model size and details if available
    try:
        with st.spinner(f"Loading model details for {selected_model}..."):
            model_info = show_model_info(selected_model)
            if model_info and "size" in model_info:
                size_mb = model_info["size"] / (1024 * 1024)
                st.info(f"Model size: {size_mb:.1f} MB")
    except:
        pass  # Silently fail if we can't get model info

    # Add warning about removal being permanent
    st.warning("⚠️ Removing a model is permanent. You will need to download it again if you want to use it in the future.")

    confirm_label = f"❌ Confirm removal of model `{selected_model}`"
    confirm = st.checkbox(confirm_label)
    
    remove_col, cancel_col = st.columns(2)
    
    with remove_col:
        if st.button("🗑️ Remove Model", key="remove_model", use_container_width=True) and confirm:
            if selected_model:
                with st.spinner(f"Removing model {selected_model}..."):
                    result = remove_model(selected_model)
                
                if result["status"] == "success":
                    st.success(result["message"])
                else:
                    st.error(result["message"])

                # Clear the cache of get_available_models
                get_available_models.clear()

                # Update the list of available models
                updated_models = get_available_models()
                
                # Update selected_model if it was removed
                if selected_model not in updated_models:
                    if updated_models:
                        st.session_state.selected_model = updated_models[0]
                    else:
                        st.session_state.selected_model = None
                
                st.rerun()
            else:
                st.error("Please select a model.")
    
    with cancel_col:
        if st.button("Cancel", key="cancel_remove", use_container_width=True):
            st.rerun()