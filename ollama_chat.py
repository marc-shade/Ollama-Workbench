# ollama_chat.py
import streamlit as st
import streamlit.components.v1 as components
from ollama_utils import call_ollama_endpoint

def ollama_chat():
    """Streamlit component for the Ollama Workbench chat interface."""
    model = st.session_state.get('model', 'mistral:instruct')
    web_page_content = st.session_state.get('web_page_content', None)
    web_page_url = st.session_state.get('url', None)

    # Construct the initial prompt
    initial_prompt = ""
    if web_page_content:
        initial_prompt = f"You are an AI assistant working within a browser extension. You have access to the current web page's content. Please use this information to answer the user's question.\n\nWebpage URL: {web_page_url}\nWebpage Content:\n{web_page_content}\n\n"
    else:
        initial_prompt = "You are an AI assistant. How can I help you today?\n\n"

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": initial_prompt}]

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Enter your message:")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Construct the full prompt
        prompt = initial_prompt + f"User: {user_input}\nAssistant:"
        
        # Call the Ollama API
        response, _, _, _ = call_ollama_endpoint(model, prompt=prompt, temperature=0.7, max_tokens=1000) # Hard-coded settings for now
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.experimental_rerun()

# Declare the Streamlit component
components.declare_component(
    "ollama_chat",
    url="http://localhost:8502/chat_endpoint"  # Hard-coded URL
)