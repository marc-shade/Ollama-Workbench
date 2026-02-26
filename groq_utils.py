# groq_utils.py
import streamlit as st
from groq import Groq
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from ollama_utils import load_api_keys, save_api_keys

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# Load the embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_groq_client(api_key: str):
    """Returns a Groq client instance if API key is available, otherwise returns None."""
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.warning("Groq API key not configured. Some features will be limited to local models.")
        return None

def call_groq_api(model, messages=None, temperature=0.7, max_tokens=1000, groq_api_key=None, **kwargs) -> str:
    """Calls the Groq API for chat completions.

    Args:
        model: Model name string (e.g. "llama3-70b-8192"). For backward
               compatibility, if a Groq client object is passed the next
               positional arg is treated as the model name.
        messages: A list of message dicts, or a plain string (auto-wrapped).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        groq_api_key: Groq API key. Loaded from api_keys.json when None.
        **kwargs: Accepts (and ignores) extra keyword args such as ``client``
                  for backward compatibility.
    """
    # --- backward compat: old callers passed (client, model, messages, ...) ---
    if isinstance(model, Groq):
        # Shift positional args: model is actually the client, messages is the model, etc.
        _client_ignored = model
        model = messages  # second positional was the real model name
        messages = temperature if not isinstance(temperature, (int, float)) else None
        # Pull remaining kwargs the old caller may have passed by name
        messages = kwargs.get("messages", messages)
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        groq_api_key = kwargs.get("groq_api_key", groq_api_key)

    # --- normalise messages ------------------------------------------------
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not messages:
        return None

    # --- resolve API key and build client ----------------------------------
    if not groq_api_key:
        groq_api_key = load_api_keys().get("groq_api_key")
    if not groq_api_key:
        st.warning("Groq API key not configured.")
        return None

    client = Groq(api_key=groq_api_key)

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