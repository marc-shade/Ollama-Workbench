# prompts.py
import json
import os
import streamlit as st

def get_prompts_file_path(prompt_type):
    prompts_folder = "prompts"
    if not os.path.exists(prompts_folder):
        os.makedirs(prompts_folder)
    return os.path.join(prompts_folder, f"{prompt_type}_prompts.json")

def load_prompts(prompt_type):
    file_path = get_prompts_file_path(prompt_type)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        return {}

def save_prompts(prompt_type, prompts):
    file_path = get_prompts_file_path(prompt_type)
    with open(file_path, "w") as f:
        json.dump(prompts, f, indent=4)

def get_agent_prompt():
    return load_prompts("agent")

def get_metacognitive_prompt():
    return load_prompts("metacognitive")

def manage_prompts():
    st.header("Manage Prompts")
    prompt_types = ["Agent", "Metacognitive"]
    selected_prompt_type = st.selectbox("Select Prompt Type:", prompt_types)

    if selected_prompt_type == "Agent":
        prompts = get_agent_prompt()
    else:
        prompts = get_metacognitive_prompt()

    # Use st.markdown to inject CSS for 100% width
    st.markdown("""
        <style>
        div[data-testid="stDataEditor"] {
            width: 100% !important;
        }
        </style>
    """, unsafe_allow_html=True)

    edited_prompts = st.data_editor(prompts, num_rows="dynamic", key=f"{selected_prompt_type}_prompts")

    if edited_prompts != prompts:
        if selected_prompt_type == "Agent":
            save_prompts("agent", edited_prompts)
        else:
            save_prompts("metacognitive", edited_prompts)
        st.success(f"{selected_prompt_type} prompts saved successfully!")
        st.experimental_rerun()