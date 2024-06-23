# prompts.py
import json
import os
import streamlit as st

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_prompts_file_path(prompt_type):
    prompts_folder = os.path.join(SCRIPT_DIR, "prompts")
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
        prompt_type = "agent"
    else:
        prompts = get_metacognitive_prompt()
        prompt_type = "metacognitive"

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
        save_prompts(prompt_type, edited_prompts)
        st.success(f"{selected_prompt_type} prompts saved successfully!")
        st.experimental_rerun()

    # Download prompts
    st.download_button(
        label=f"Download {selected_prompt_type} Prompts",
        data=json.dumps(edited_prompts, indent=4),
        file_name=f"{prompt_type}_prompts.json",
        mime="application/json",
    )

    # Upload prompts
    uploaded_file = st.file_uploader(f"Upload {selected_prompt_type} Prompts", type=["json"])
    if uploaded_file is not None:
        try:
            uploaded_prompts = json.load(uploaded_file)
            # Append uploaded prompts to existing prompts
            edited_prompts.update(uploaded_prompts)
            save_prompts(prompt_type, edited_prompts)
            st.success(f"{selected_prompt_type} prompts uploaded and appended successfully!")
            st.experimental_rerun()
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid prompts JSON file.")
