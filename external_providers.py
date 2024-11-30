# external_providers.py
import streamlit as st
from ollama_utils import load_api_keys, save_api_keys

ADVANCED_GROQ_MODELS = [
    "llama-3.1-405b-reasoning",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
]

def external_providers_ui():
    st.title("☁️ External Providers")

    api_keys = load_api_keys()

    col1, col2 = st.columns(2)

    with col1:
        st.header("Search Providers")
        api_keys["serpapi_api_key"] = st.text_input("SerpApi API Key", value=api_keys.get("serpapi_api_key", ""), type="password")
        api_keys["serper_api_key"] = st.text_input("Serper API Key", value=api_keys.get("serper_api_key", ""), type="password")
        api_keys["google_api_key"] = st.text_input("Google Custom Search API Key", value=api_keys.get("google_api_key", ""), type="password")
        api_keys["google_cse_id"] = st.text_input("Google Custom Search Engine ID", value=api_keys.get("google_cse_id", ""), type="password")
        api_keys["bing_api_key"] = st.text_input("Bing Search API Key", value=api_keys.get("bing_api_key", ""), type="password")

    with col2:
        st.header("AI Model Providers")
        api_keys["openai_api_key"] = st.text_input("OpenAI API Key", value=api_keys.get("openai_api_key", ""), type="password")
        api_keys["groq_api_key"] = st.text_input("Groq API Key", value=api_keys.get("groq_api_key", ""), type="password")
        api_keys["mistral_api_key"] = st.text_input("Mistral API Key", value=api_keys.get("mistral_api_key", ""), type="password")
        
        # Add checkbox for advanced Groq models
        api_keys["use_advanced_groq_models"] = st.checkbox(
            "Enable Advanced Groq Models",
            value=api_keys.get("use_advanced_groq_models", False),
            help="Check this box if your Groq account is approved to use advanced models."
        )

    if st.button("💾 Save API Keys"):
        save_api_keys(api_keys)
        st.success("🟢 API keys saved!")

def get_available_groq_models(api_keys):
    base_models = [
        "llama3-8b-8192",
        "llama3-70b-8192",
        "llama3-groq-8b-8192-tool-use-preview",
        "llama3-groq-70b-8192-tool-use-preview",
        "mixtral-8x7b-32768",
        "gemma-7b-it",
        "gemma2-9b-it",
    ]
    
    if api_keys.get("use_advanced_groq_models", False):
        return base_models + ADVANCED_GROQ_MODELS
    else:
        return base_models