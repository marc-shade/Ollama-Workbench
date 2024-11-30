# mistral_utils.py
import os
import json
import streamlit as st
from mistralai import Mistral
from typing import List, Dict, Union, AsyncGenerator

MISTRAL_MODELS = [
    "mistral-large-latest",
    "mistral-tiny",
    "mistral-embed",
    "pixtral-12b-2409",
    "open-mistral-nemo",
    "open-codestral-mamba"
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

def get_mistral_client(api_key: str = None) -> Union[Mistral, None]:
    """Returns a Mistral client instance if API key is available."""
    if not api_key:
        api_key = load_api_keys().get('mistral_api_key')
    if not api_key:
        return None
    try:
        return Mistral(api_key=api_key)
    except Exception as e:
        st.warning(f"Error initializing Mistral client: {str(e)}")
        return None

def call_mistral_api(
    model: str,
    messages: List[Dict[str, str]] = None,
    prompt: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    stream: bool = False,
    json_response: bool = False,
    mistral_api_key: str = None
) -> Union[str, AsyncGenerator]:
    """Wrapper function to call the Mistral Chat API with a unified interface."""
    client = get_mistral_client(mistral_api_key)
    if not client:
        return None

    try:
        kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Convert prompt to messages format if needed
        if prompt is not None:
            kwargs["messages"] = [{"role": "user", "content": prompt}]
        elif messages is not None:
            kwargs["messages"] = messages
        else:
            raise ValueError("Either prompt or messages must be provided")

        if json_response:
            kwargs["response_format"] = {"type": "json_object"}

        if stream:
            response = client.chat.stream(**kwargs)
            return response
        else:
            response = client.chat.complete(**kwargs)
            return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling Mistral API: {e}")
        return None

async def call_mistral_api_async(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 1000,
    json_response: bool = False,
    mistral_api_key: str = None
) -> AsyncGenerator:
    """Async wrapper function to call the Mistral Chat API."""
    client = get_mistral_client(mistral_api_key)
    if not client:
        return None

    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if json_response:
            kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.stream_async(**kwargs)
        return response
    except Exception as e:
        st.error(f"Error calling Mistral API async: {e}")
        return None

def call_mistral_embeddings(text: Union[str, List[str]], model: str = "mistral-embed", mistral_api_key: str = None):
    """Calls the Mistral Embeddings API."""
    client = get_mistral_client(mistral_api_key)
    if not client:
        return None

    try:
        if isinstance(text, str):
            text = [text]
        
        response = client.embeddings.create(
            model=model,
            inputs=text
        )
        
        if len(text) == 1:
            return response.data[0].embedding
        return [item.embedding for item in response.data]
    except Exception as e:
        st.error(f"Error calling Mistral Embeddings API: {e}")
        return None

def display_mistral_settings():
    """Displays the Mistral API key settings in the sidebar."""
    st.sidebar.subheader("Mistral API Key")
    api_keys = load_api_keys()
    mistral_api_key = st.sidebar.text_input(
        "Enter your Mistral API key:",
        value=api_keys.get("mistral_api_key", ""),
        type="password",
    )
    if st.sidebar.button("Save Mistral API Key"):
        api_keys["mistral_api_key"] = mistral_api_key
        save_api_keys(api_keys)
        st.success("Mistral API key saved!")
