# local_models.py
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from ollama_utils import *

def list_models():
    st.header("List Local Models")
    models = list_local_models()
    if models:
        # Prepare data for the dataframe
        data = []
        for model in models:
            size_gb = model.get('size', 0) / (1024**3)  # Convert bytes to GB
            modified_at = model.get('modified_at', 'Unknown')
            if modified_at != 'Unknown':
                modified_at = datetime.fromisoformat(modified_at).strftime('%Y-%m-%d %H:%M:%S')
            data.append({
                "Model Name": model['name'],
                "Size (GB)": size_gb,
                "Modified At": modified_at
            })
        
        # Create a pandas dataframe
        df = pd.DataFrame(data)

        # Calculate height based on the number of rows
        row_height = 35  # Set row height
        height = row_height * len(df) + 35  # Calculate height
        
        # Display the dataframe with Streamlit
        st.dataframe(df, use_container_width=True, height=height, hide_index=True)

def list_local_models():
    st.title("🤖 Local Models")
    response = requests.get(f"{OLLAMA_URL}/tags")
    response.raise_for_status()
    models = response.json().get("models", [])
    if not models:
        st.write("No local models available.")
        return

    # Prepare data for the dataframe
    data = []
    for model in models:
        size_gb = model.get('size', 0) / (1024**3)  # Convert bytes to GB
        modified_at = model.get('modified_at', 'Unknown')
        if modified_at != 'Unknown':
            modified_at = datetime.fromisoformat(modified_at).strftime('%Y-%m-%d %H:%M:%S')
        data.append({
            "Model Name": model['name'],
            "Size (GB)": size_gb,
            "Modified At": modified_at
        })

    # Create a pandas dataframe
    df = pd.DataFrame(data)

    # Display the dataframe with Streamlit
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Add Preload and Keep-Alive controls
    st.subheader("⚡ Model Actions")
    for model_name in df["Model Name"]:
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"Preload {model_name}"):
                preload_model(model_name)
        with col2:
            keep_alive = st.text_input(f"Keep-Alive for {model_name}", value="", key=f"keep_alive_{model_name}")
            if st.button(f"Apply Keep-Alive for {model_name}"):
                apply_model_keep_alive(model_name, keep_alive)
