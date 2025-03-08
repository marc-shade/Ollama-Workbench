# local_models.py
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from ollama_utils import *

def list_local_models():
    st.title("🤖 Local Ollama Models")
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
            "Size (GB)": f"{size_gb:.2f}",
            "Modified At": modified_at,
            "Preload": False,
            "Keep-Alive": ""
        })

    # Create a pandas dataframe
    df = pd.DataFrame(data)

    # Display the editable dataframe
    edited_df = st.data_editor(
        df,
        hide_index=True,
        column_config={
            "Model Name": st.column_config.TextColumn("Model Name", disabled=True),
            "Size (GB)": st.column_config.NumberColumn("Size (GB)", format="%.2f", disabled=True),
            "Modified At": st.column_config.TextColumn("Modified At", disabled=True),
            "Preload": st.column_config.CheckboxColumn("Preload"),
            "Keep-Alive": st.column_config.TextColumn("Keep-Alive")
        },
        column_order=("Model Name", "Size (GB)", "Modified At", "Preload", "Keep-Alive"),
        key="model_editor"
    )

    # Add Apply Changes button
    if st.button("Apply Changes"):
        for index, row in edited_df.iterrows():
            model_name = row["Model Name"]
            if row["Preload"]:
                preload_model(model_name)
            if row["Keep-Alive"]:
                apply_model_keep_alive(model_name, row["Keep-Alive"])
        st.success("Changes applied successfully!")