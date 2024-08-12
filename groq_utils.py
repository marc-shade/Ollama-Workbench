# groq_utils.py
import requests
import os
import json
import streamlit as st

GROQ_MODELS = ["llama2-70b-4096", "mixtral-8x7b-32768", "gemma-7b-it"]

API_KEYS_FILE = "api_keys.json"
GROQ_API_URL = "https://api.groq.com/v1"

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
    """Calls the Groq API."""
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
    except Exception as e:
        st.error(f"Error calling Groq API: {e}")
        return None

def display_groq_settings():
    """Displays the Groq API key settings in the sidebar."""
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