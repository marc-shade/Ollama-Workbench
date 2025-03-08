# openai_utils.py
import os
import json
import streamlit as st
try:
    from openai import OpenAI
except ImportError:
    print("Warning: openai package not found, using fallback implementation")
    # Fallback implementation for openai
    class OpenAI:
        def __init__(self, api_key=None):
            self.api_key = api_key
            
        class Completion:
            @staticmethod
            def create(*args, **kwargs):
                return {"choices": [{"text": "OpenAI API not available - please install the openai package"}]}
        
        class ChatCompletion:
            @staticmethod
            def create(*args, **kwargs):
                return {"choices": [{"message": {"content": "OpenAI API not available - please install the openai package"}}]}
                
        class Embedding:
            @staticmethod
            def create(*args, **kwargs):
                return {"data": [{"embedding": [0.0] * 1536}]}
                
        completion = Completion()
        chat = ChatCompletion()
        embeddings = Embedding()

OPENAI_MODELS = [
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo",
    "gpt-4o-mini",
    "gpt-4o-2024-08-06",
    "gpt-4o",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-0314",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-instruct"
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
    api_keys = load_api_keys()
    api_keys['openai_api_key'] = api_key
    save_api_keys(api_keys)
    st.success("OpenAI API key has been set.")

def call_openai_api(model, messages, temperature=0.7, max_tokens=1000, frequency_penalty=0.0, presence_penalty=0.0, stream=False, openai_api_key=None):
    """Wrapper function to call the OpenAI Chat API with a unified interface."""
    client = OpenAI(api_key=openai_api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=stream
        )
        
        if stream:
            return response  # Return the stream object
        else:
            return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return None

def call_openai_embeddings(model, input_text):
    """Calls the OpenAI Embeddings API."""
    client = OpenAI(api_key=load_api_keys().get('openai_api_key'))
    
    try:
        response = client.embeddings.create(
            model=model,
            input=input_text
        )
        return response.data[0].embedding
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
        set_openai_api_key(openai_api_key)