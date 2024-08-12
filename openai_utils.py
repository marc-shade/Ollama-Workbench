# openai_utils.py
import openai
import os
import json
import streamlit as st

OPENAI_MODELS = [
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo",
    "gpt-4o-mini",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-0314",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-instruct",
    "text-davinci-003",
    "text-davinci-002",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001",
    "code-davinci-002",
    "code-cushman-001"
]

API_KEYS_FILE = "api_keys.json"

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

def set_openai_api_key(api_key):
    """Sets the OpenAI API key."""
    openai.api_key = api_key

def call_openai_api(model, messages, temperature=0.7, max_tokens=1000, openai_api_key=None):
    """Calls the OpenAI Chat Completion API."""
    set_openai_api_key(openai_api_key)
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return None

def display_openai_settings():
    """Displays the OpenAI API key settings in the sidebar."""
    st.sidebar.subheader("OpenAI API Key")
    api_keys = load_api_keys()
    openai_api_key = st.sidebar.text_input(
        "Enter your OpenAI API key:",
        value=api_keys.get("openai_api_key", ""),
        type="password",
    )
    if st.sidebar.button("Save OpenAI API Key"):
        api_keys["openai_api_key"] = openai_api_key
        save_api_keys(api_keys)
        st.success("OpenAI API key saved!")