# ollama_utils.py
import requests
import json
import io
import time
import streamlit as st
import ollama
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api"

@st.cache_data  # Cache the list of available models
def get_available_models():
    response = requests.get(f"{OLLAMA_URL}/tags")
    response.raise_for_status()
    models = [
        model["name"]
        for model in response.json()["models"]
        if "embed" not in model["name"]
    ]
    return models

def call_ollama_endpoint(model, prompt=None, image=None, temperature=0.5, max_tokens=150, presence_penalty=0.0, frequency_penalty=0.0, context=None):
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "context": context if context is not None else [],
    }
    if prompt:
        payload["prompt"] = prompt
    if image:
        # Read image data into BytesIO
        image_bytesio = io.BytesIO(image.read())

        # Determine image format and filename
        image_format = "image/jpeg" if image.type == "image/jpeg" else "image/png"
        filename = "image.jpg" if image.type == "image/jpeg" else "image.png"

        # Send image data using multipart/form-data
        files = {"file": (filename, image_bytesio, image_format)}
        response = requests.post(f"{OLLAMA_URL}/generate", data=payload, files=files, stream=True)
    else:
        response = requests.post(f"{OLLAMA_URL}/generate", json=payload, stream=True)
    try:
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {str(e)}", None, None, None  # Return None for eval_count and eval_duration

    response_parts = []
    eval_count = None
    eval_duration = None
    for line in response.iter_lines():
        part = json.loads(line)
        response_parts.append(part.get("response", ""))
        if part.get("done", False):
            eval_count = part.get("eval_count", None)
            eval_duration = part.get("eval_duration", None)
            break
    return "".join(response_parts), part.get("context", None), eval_count, eval_duration

def check_json_handling(model, temperature, max_tokens, presence_penalty, frequency_penalty):
    prompt = "Return the following data in JSON format: name: John, age: 30, city: New York"
    result, _, _, _ = call_ollama_endpoint(model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)
    try:
        json.loads(result)
        return True
    except json.JSONDecodeError:
        return False

def check_function_calling(model, temperature, max_tokens, presence_penalty, frequency_penalty):
    prompt = "Define a function named 'add' that takes two numbers and returns their sum. Then call the function with arguments 5 and 3."
    result, _, _, _ = call_ollama_endpoint(model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)
    return "8" in result

def pull_model(model_name):
    payload = {"name": model_name, "stream": True}
    response = requests.post(f"{OLLAMA_URL}/pull", json=payload, stream=True)
    response.raise_for_status()
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    total = None
    st.write(f"📥 Pulling model: `{model_name}`")
    for line in response.iter_lines():
        line = line.decode("utf-8")  # Decode the line from bytes to str
        data = json.loads(line)
        
        if "total" in data and "completed" in data:
            total = data["total"]
            completed = data["completed"]
            progress = completed / total
            progress_bar.progress(progress)
            status_text.text(f"Progress: {progress * 100:.2f}%")
        else:
            progress = None
            if not data["status"].startswith("pulling"):
                status_text.text(data["status"])
        
        if data["status"] == "success":
            break
        
    return results

def show_model_info(model_name):
    payload = {"name": model_name}
    response = requests.post(f"{OLLAMA_URL}/show", json=payload)
    response.raise_for_status()
    return response.json()

def remove_model(model_name):
    payload = {"name": model_name}
    response = requests.delete(f"{OLLAMA_URL}/delete", json=payload)
    if response.status_code == 200:
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"status": "success", "message": f"Model '{model_name}' removed successfully."}
    else:
        return {"status": "error", "message": f"Failed to remove model '{model_name}'. Status code: {response.status_code}"}

def save_chat_history(chat_history, filename="chat_history.json"):
    with open(filename, "w") as f:
        json.dump(chat_history, f)

def load_chat_history(filename):
    with open(filename, "r") as f:
        return json.load(f)

def update_model_selection(selected_models, key):
    st.session_state[key] = selected_models