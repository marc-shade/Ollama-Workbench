# prompts.py (COMPLETE)
import json
import os
import streamlit as st
from global_vrm_loader import global_vrm_loader
import base64

# Directory where the script is located
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
    prompts = load_prompts("agent")
    if not prompts:
        # Default prompts if none exist
        prompts = {
            "General Assistant": {
                "prompt": "You are a helpful AI assistant focused on general conversation and tasks. You aim to be clear, informative, and engaging while maintaining a natural conversational style.",
                "model_voice": "en-US-Wavenet-A"
            },
            "Code Assistant": {
                "prompt": "You are a skilled programming assistant. When discussing code, you provide clear explanations and practical examples. You help with debugging, optimization, and best practices.",
                "model_voice": "en-US-Wavenet-B"
            },
            "Technical Writer": {
                "prompt": "You are a technical writing assistant who helps create clear, well-structured documentation and articles. You excel at explaining complex topics in an accessible way.",
                "model_voice": "en-US-Wavenet-C"
            }
        }
        save_prompts("agent", prompts)
    else:
        # Create a new dictionary with updated prompts
        updated_prompts = {}
        for key, value in prompts.items():
            if isinstance(value, dict):
                if 'model_voice' not in value:
                    value['model_voice'] = 'en-US-Wavenet-A'
                updated_prompts[key] = value
            else:
                # Convert string prompts to dict format
                updated_prompts[key] = {
                    "prompt": value,
                    "model_voice": "en-US-Wavenet-A"
                }
        prompts = updated_prompts
        save_prompts("agent", prompts)
    return prompts

def get_metacognitive_prompt():
    prompts = load_prompts("metacognitive")
    if not prompts:
        # Default prompts if none exist
        prompts = {
            "Analytical": "I approach problems systematically, breaking them down into smaller components and analyzing each part carefully.",
            "Intuitive": "I combine logical analysis with intuitive understanding, considering both practical and creative solutions.",
            "Collaborative": "I engage in a collaborative thinking process, actively involving you in the discussion and solution-finding."
        }
        save_prompts("metacognitive", prompts)
    return prompts

def get_voice_prompt():
    prompts = load_prompts("voice")
    if not prompts:
        # Default prompts if none exist
        prompts = {
            "Professional": "I maintain a clear, professional tone while being approachable and helpful.",
            "Casual": "I use a friendly, conversational tone while remaining informative and helpful.",
            "Technical": "I use precise technical language when appropriate, but can adjust my explanations to match your expertise level."
        }
        save_prompts("voice", prompts)
    return prompts

def get_identity_prompt():
    return load_prompts("identity")

def manage_prompts():
    st.title("✨ Prompts")
    prompt_types = ["Agent", "Metacognitive", "Voice", "Identity", "Model Voice"]
    selected_prompt_type = st.selectbox("Select Prompt Type:", prompt_types)

    if selected_prompt_type == "Agent":
        prompts = get_agent_prompt()
        prompt_type = "agent"
    elif selected_prompt_type == "Metacognitive":
        prompts = get_metacognitive_prompt()
        prompt_type = "metacognitive"
    elif selected_prompt_type == "Voice":
        prompts = get_voice_prompt()
        prompt_type = "voice"
    elif selected_prompt_type == "Identity":
        prompts = get_identity_prompt()
        prompt_type = "identity"
    else:
        prompts = load_prompts("model_voice")
        prompt_type = "model_voice"

    st.markdown("""
        <style>
        .stDataFrame, div[data-testid="stDataEditor"] {
            width: 100% !important;
        }
        </style>
    """, unsafe_allow_html=True)

    if selected_prompt_type == "Agent":
        for key in list(prompts.keys()):
            with st.expander(key):
                if isinstance(prompts[key], str):
                    prompts[key] = {"prompt": prompts[key], "model_voice": "en-US-Wavenet-A"}

                prompts[key]['prompt'] = st.text_area("Prompt", value=prompts[key].get('prompt', ''), key=f"{key}_prompt")
                voice_options = ["en-US-Wavenet-A", "en-US-Wavenet-B", "en-US-Wavenet-C", "en-US-Wavenet-D"]
                prompts[key]['model_voice'] = st.selectbox("🗣️ Model Voice:", voice_options, index=voice_options.index(prompts[key].get('model_voice', "en-US-Wavenet-A")), key=f"{key}_model_voice")

                # VRM Model Upload
                vrm_model_file = st.file_uploader(f"Upload VRM Model for {key}", type=["vrm"], key=f"vrm_upload_{key}")
                if vrm_model_file is not None:
                    # Save the VRM model file
                    agent_models_dir = os.path.join(SCRIPT_DIR, "agent_models")
                    if not os.path.exists(agent_models_dir):
                        os.makedirs(agent_models_dir)
                    vrm_model_path = os.path.join(agent_models_dir, vrm_model_file.name)
                    with open(vrm_model_path, "wb") as f:
                        f.write(vrm_model_file.getvalue())

                    prompts[key]['vrm_model_path'] = vrm_model_path  # Store the path
                    global_vrm_loader.load_model(key, vrm_model_path)
                    st.success("VRM model uploaded successfully!")

        if st.button("Add New Agent Prompt"):
            new_key = f"New Agent {len(prompts) + 1}"
            prompts[new_key] = {"prompt": "", "model_voice": "en-US-Wavenet-A"}
            st.success(f"Added new agent prompt: {new_key}")
            st.rerun()
    else:
        edited_prompts = st.data_editor(prompts, num_rows="dynamic", key=f"{selected_prompt_type}_prompts")

    if st.button("Save Prompts"):
        if selected_prompt_type == "Agent":
            save_prompts(prompt_type, prompts)
        else:
            save_prompts(prompt_type, edited_prompts)
        st.success(f"{selected_prompt_type} prompts saved successfully!")
        st.rerun()

if __name__ == "__main__":
    manage_prompts()