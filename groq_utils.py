# groq_utils.py
import requests
import os
import json
import streamlit as st

GROQ_MODELS = [
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-groq-8b-8192-tool-use-preview",
    "llama-guard-3-8b",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
    "gemma2-9b-it",
    "whisper-large-v3"
]

API_KEYS_FILE = "api_keys.json"
GROQ_API_URL = "https://api.groq.com/openai/v1"

def load_api_keys():
    """Loads API keys from the JSON file."""
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_api_keys(api_keys):
    """Saves API keys to the JSON file."""
    with open(API_KEYS_FILE, "w") as f:
        json.dump(api_keys, f, indent=4)

def call_groq_api(model, prompt, temperature=0.7, max_tokens=1000, groq_api_key=None):
    """Calls the Groq API for chat completions."""
    url = f"{GROQ_API_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 429:
            retry_after = response.headers.get('retry-after', 1)
            st.error(f"Rate limit exceeded. Please retry after {retry_after} seconds.")
        else:
            st.error(f"HTTP error occurred: {http_err}")
        return None
    except Exception as e:
        st.error(f"Error calling Groq API: {e}")
        return None

def display_groq_settings():
    """Displays the Groq API key settings."""
    st.sidebar.subheader("Groq API Key")
    api_keys = load_api_keys()
    groq_api_key = st.sidebar.text_input(
        "Enter your Groq API key:",
        value=api_keys.get("groq_api_key", ""),
        type="password",
    )
    if st.sidebar.button("Save Groq API Key"):
        api_keys["groq_api_key"] = groq_api_key
        save_api_keys(api_keys)
        st.success("Groq API key saved!")

def list_groq_models(groq_api_key=None):
    """Lists all available models from the Groq API."""
    url = f"{GROQ_API_URL}/models"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        models = response.json()
        return models
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
        return []
    except Exception as e:
        st.error(f"Error retrieving models from Groq API: {e}")
        return []

def call_groq_embeddings(model, text, groq_api_key=None):
    """Calls the Groq API to generate embeddings for the given text."""
    url = f"{GROQ_API_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "input": text
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['embedding']
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred while generating Groq embeddings: {http_err}")
        return None
    except Exception as e:
        st.error(f"Error generating Groq embeddings: {e}")
        return None
