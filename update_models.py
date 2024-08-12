# update_models.py
import streamlit as st
from ollama_utils import *

def update_models():
    st.header("🔄 Update Local Ollama Models")
    available_models = get_available_models()
    if st.button("⚡ Update All Models Now!"):
        for model_name in available_models:
            # Skip custom models (those with a ':' in the name)
            if 'gpt' in model_name:
                st.write(f"Skipping custom model: `{model_name}`")
                continue
            st.write(f"Updating model: `{model_name}`")
            pull_model(model_name)
        st.success("All models updated successfully!")