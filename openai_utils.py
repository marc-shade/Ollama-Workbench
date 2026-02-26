# openai_utils.py
import streamlit as st
from openai import OpenAI
from ollama_utils import load_api_keys, save_api_keys

OPENAI_MODELS = [
    # GPT-4.1 models (2025)
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",

    # GPT-4o models (2024/2025)
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-08-06",
    "gpt-4o",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-mini",

    # GPT-4 Turbo models
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",

    # Legacy GPT-4 models
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-0314",

    # GPT-3.5 models
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-instruct",

    # Reasoning models
    "o3-mini",
    "o4-mini",
]


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