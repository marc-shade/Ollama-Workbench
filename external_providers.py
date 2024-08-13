# external_providers.py
import streamlit as st
from ollama_utils import load_api_keys, save_api_keys

def external_providers_ui():
    st.title("☁️ External Providers")

    api_keys = load_api_keys()

    st.header("API Key Settings")

    col1, col2 = st.columns(2)

    with col1:
        api_keys["serpapi_api_key"] = st.text_input("SerpApi API Key", value=api_keys.get("serpapi_api_key", ""), type="password")
        api_keys["serper_api_key"] = st.text_input("Serper API Key", value=api_keys.get("serper_api_key", ""), type="password")
        api_keys["google_api_key"] = st.text_input("Google Custom Search API Key", value=api_keys.get("google_api_key", ""), type="password")
        api_keys["google_cse_id"] = st.text_input("Google Custom Search Engine ID", value=api_keys.get("google_cse_id", ""), type="password")

    with col2:
        api_keys["bing_api_key"] = st.text_input("Bing Search API Key", value=api_keys.get("bing_api_key", ""), type="password")
        api_keys["openai_api_key"] = st.text_input("OpenAI API Key", value=api_keys.get("openai_api_key", ""), type="password")
        api_keys["groq_api_key"] = st.text_input("Groq API Key", value=api_keys.get("groq_api_key", ""), type="password")

    if st.button("💾 Save API Keys"):
        save_api_keys(api_keys)
        st.success("🟢 API keys saved!")