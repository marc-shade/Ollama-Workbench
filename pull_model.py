# pull_model.py
import streamlit as st
import requests
import json
import time

def pull_model(model_name):
    """
    Pull a model from the Ollama library.
    
    Args:
    model_name (str): The name of the model to pull.
    
    Yields:
    dict: Status updates during the pull process, including progress information.
    """
    url = "http://localhost:11434/api/pull"
    payload = {"name": model_name, "stream": True}
    headers = {"Content-Type": "application/json"}

    try:
        with requests.post(url, json=payload, headers=headers, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "status" in data:
                        if data["status"] == "pulling manifest":
                            yield {"status": "Initializing...", "progress": 0}
                        elif "total" in data and "completed" in data:
                            progress = data["completed"] / data["total"]
                            yield {"status": "Downloading", "progress": progress}
                        elif data["status"] in ["verifying sha256 digest", "writing manifest", "removing any unused layers"]:
                            yield {"status": "Finalizing...", "progress": 1}
                        elif data["status"] == "success":
                            yield {"status": "Complete", "progress": 1}
    except requests.RequestException as e:
        yield {"status": f"Error: {str(e)}", "progress": 0}

def pull_models():
    st.header("⬇ Pull a Model from Ollama Library")
    st.write("Enter the exact name of the model you want to pull from the Ollama library. You can just paste the whole model snippet from the model library page like 'ollama run llava-phi3' or you can just enter the model name like 'llava-phi3' and then click 'Pull Model' to begin the download. The progress of the download will be displayed below.")

    col1, col2 = st.columns([10, 1], vertical_alignment="bottom")

    with col1:
        model_name = st.text_input("Enter the name of the model you want to pull:")
    
    with col2:
        pull_button = st.button("Pull Model", key="pull_model")

    if pull_button:
        if model_name:
            # Strip off "ollama run" or "ollama pull" from the beginning
            model_name = model_name.replace("ollama run ", "").replace("ollama pull ", "").strip()

            progress_bar = st.progress(0)
            status_text = st.empty()

            for update in pull_model(model_name):
                progress_bar.progress(update["progress"])
                status_text.text(update["status"])
                time.sleep(0.1)  # Small delay to prevent overwhelming the UI

            if update["status"] == "Complete":
                st.success(f"Model '{model_name}' pulled successfully!")
            elif "Error" in update["status"]:
                st.error(update["status"])
        else:
            st.error("Please enter a model name.")

if __name__ == "__main__":
    pull_models()