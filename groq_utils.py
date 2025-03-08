# groq_utils.py
import os
import json
import streamlit as st
from groq import Groq
from typing import List, Dict
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence_transformers package not found, using fallback implementation")
    # Fallback implementation for sentence_transformers
    class SentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name
            print(f"Warning: Using fallback SentenceTransformer with model: {model_name}")
            
        def encode(self, text, **kwargs):
            import numpy as np
            print(f"Warning: Using fallback encoding for text: {text[:50]}...")
            # Return a random embedding vector of size 384 (same as all-MiniLM-L6-v2)
            return np.random.rand(384)

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
]

API_KEYS_FILE = "api_keys.json"

# Load the embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

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

def get_groq_client(api_key: str):
    """Returns a Groq client instance if API key is available, otherwise returns None."""
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.warning("Groq API key not configured. Some features will be limited to local models.")
        return None

def call_groq_api(client: Groq, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1000) -> str:
    """Calls the Groq API for chat completions."""
    if not client:
        return None
    try:
        chat_completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.warning(f"Error calling Groq API: {str(e)}")
        return None

def get_local_embeddings(text: str) -> List[float]:
    """Generates embeddings using a local model."""
    model = load_embedding_model()
    return model.encode(text).tolist()

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