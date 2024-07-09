# artifact.py
import streamlit as st
import json
import os
import re
from datetime import datetime
from ollama_utils import get_available_models, call_ollama_endpoint, save_chat_history, load_chat_history, pull_model, show_model_info, remove_model

# Function to extract code blocks from text
def extract_code_blocks(text):
    code_blocks = re.findall(r'```[\s\S]*?```', text)
    return [block.strip('`').strip() for block in code_blocks]

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'artifacts' not in st.session_state:
    st.session_state['artifacts'] = []

# Streamlit app layout
st.set_page_config(layout="wide")

# Sidebar for model selection and management
with st.sidebar:
    st.title("Ollama Model Chat")
    models = get_available_models()
    selected_model = st.selectbox("Select Model", models)
    
    if st.button("Pull Model"):
        pull_model(selected_model)
        st.success(f"Model '{selected_model}' pulled successfully.")
        
    if st.button("Show Model Info"):
        model_info = show_model_info(selected_model)
        st.json(model_info)
        
    if st.button("Remove Model"):
        result = remove_model(selected_model)
        st.write(result)

# Main layout with two columns
left_column, right_column = st.columns(2)

# Chat history on the left
with left_column:
    st.header("Chat Stream")
    for chat in st.session_state['chat_history']:
        with st.chat_message(chat["role"]):
            if chat["role"] == "assistant":
                code_blocks = extract_code_blocks(chat["content"])
                for code_block in code_blocks:
                    st.code(code_block)
                non_code_parts = re.split(r'```[\s\S]*?```', chat["content"])
                for part in non_code_parts:
                    st.markdown(part.strip())
            else:
                st.markdown(chat["content"])

# Artifacts container on the right
with right_column:
    st.header("Artifacts")
    for artifact in st.session_state['artifacts']:
        st.write(artifact)

# User input and response handling
user_input = st.text_input("You:", "")
if st.button("Send"):
    if user_input:
        st.session_state['chat_history'].append({"role": "user", "content": user_input})
        
        # Display user input
        with left_column:
            with st.chat_message("user"):
                st.markdown(user_input)
        
        # Generate and display the assistant's response
        response, _, _, _ = call_ollama_endpoint(selected_model, prompt=user_input)
        st.session_state['chat_history'].append({"role": "assistant", "content": response})
        
        with left_column:
            with st.chat_message("assistant"):
                code_blocks = extract_code_blocks(response)
                for code_block in code_blocks:
                    st.code(code_block)
                non_code_parts = re.split(r'```[\s\S]*?```', response)
                for part in non_code_parts:
                    st.markdown(part.strip())
        
        # Optionally, save the final content to artifacts based on some condition
        if "final result" in user_input.lower():
            st.session_state['artifacts'].append(response)

# Function to add artifacts manually
def add_artifact(content):
    st.session_state['artifacts'].append(content)

# Manually add an artifact
manual_artifact = st.text_input("Add Artifact Content:", "")
if st.button("Add Artifact"):
    add_artifact(manual_artifact)

# Function to save chat history
if st.button("Save Chat History"):
    save_chat_history(st.session_state['chat_history'])
    st.success("Chat history saved successfully.")

# Function to load chat history
if st.button("Load Chat History"):
    st.session_state['chat_history'] = load_chat_history("chat_history.json")
    st.success("Chat history loaded successfully.")

st.markdown(
    """
    <style>
    .stTextInput {
        margin-top: 20px;
    }
    .stButton {
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
