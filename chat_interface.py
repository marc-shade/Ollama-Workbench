# chat_interface.py

import streamlit as st
import os
import json
from datetime import datetime
import re
import ollama
from ollama_utils import get_available_models, get_all_models, load_api_keys, call_ollama_endpoint
from openai_utils import call_openai_api
from groq_utils import call_groq_api
from prompts import get_agent_prompt, get_metacognitive_prompt, get_voice_prompt
import tiktoken
from streamlit_extras.bottom_container import bottom
from enhanced_corpus import GraphRAGCorpus, OllamaEmbedder
from groq_utils import GROQ_MODELS
from openai_utils import OPENAI_MODELS

SETTINGS_FILE = "chat-settings.json"
RAGTEST_DIR = "ragtest"

# Load settings from JSON file
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
            for key, value in settings.items():
                if value != "None":  # Only set non-None values
                    st.session_state[key] = value
    print(f"Settings loaded: {st.session_state}")



def save_settings():
    settings = {
        "selected_model": st.session_state.selected_model,
        "agent_type": st.session_state.agent_type,
        "metacognitive_type": st.session_state.metacognitive_type,
        "voice_type": st.session_state.voice_type,
        "selected_corpus": st.session_state.selected_corpus,
        "temperature_slider_chat": st.session_state.temperature_slider_chat,
        "max_tokens_slider_chat": st.session_state.max_tokens_slider_chat,
        "presence_penalty_slider_chat": st.session_state.presence_penalty_slider_chat,
        "frequency_penalty_slider_chat": st.session_state.frequency_penalty_slider_chat,
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)
    print(f"Settings saved: {settings}")
    st.success("Settings saved successfully!")

def ai_assisted_prompt_writing():
    st.markdown("""
    <style>
    .stModal > div[data-testid="stHorizontalBlock"]:first-child {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader("AI Prompt Writer")
    
    if st.button("X", key="close_modal", help="Cancel assisted prompt writing."):
        st.session_state.show_prompt_modal = False
        st.rerun()
    
    user_need = st.text_input("What do you need help with?")
    if user_need:
        prompt_suggestion = generate_prompt_suggestion(user_need)
        if prompt_suggestion:
            st.write("Suggested prompt:")
            edited_prompt = st.text_area("Edit the prompt before using it:", value=prompt_suggestion)
            if st.button("Use this prompt"):
                st.session_state.chat_input = edited_prompt
                st.session_state.show_prompt_modal = False
                st.rerun()
        else:
            st.warning("Unable to generate a prompt suggestion. Please try again or select a different model.")

def pull_model(model_name):
    try:
        st.info(f"Pulling model '{model_name}'. This may take a while...")
        result = subprocess.run(['ollama', 'pull', model_name], capture_output=True, text=True, check=True)
        st.success(f"Successfully pulled model '{model_name}'.")
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to pull model '{model_name}'. Error: {e.stderr}")
        return False

def generate_prompt_suggestion(user_need):
    api_keys = load_api_keys()
    model = st.session_state.selected_model
    prompt = f"Create a detailed and effective prompt for an AI assistant based on this user need: {user_need}"

    try:
        if model in OPENAI_MODELS:
            response = call_openai_api(
                model,
                [{"role": "user", "content": prompt}],
                temperature=st.session_state.temperature_slider_chat,
                max_tokens=st.session_state.max_tokens_slider_chat,
                openai_api_key=api_keys.get("openai_api_key")
            )
            return response.strip()
        elif model in GROQ_MODELS:
            response = call_groq_api(
                model,
                prompt,
                temperature=st.session_state.temperature_slider_chat,
                max_tokens=st.session_state.max_tokens_slider_chat,
                groq_api_key=api_keys.get("groq_api_key")
            )
            return response.strip()
        else:
            response = ollama.generate(
                model,
                prompt,
                temperature=st.session_state.temperature_slider_chat,
                num_predict=st.session_state.max_tokens_slider_chat
            )
            return response['response'].strip()
    except ollama.ResponseError as e:
        if "model not found" in str(e).lower():
            st.error(f"Model '{model}' not found.")
            if st.button("Pull Model"):
                if pull_model(model):
                    st.rerun()
            return None
        else:
            st.error(f"An error occurred: {str(e)}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def get_graphrag_context(user_input, corpus_name):
    """Get context from GraphRAG corpus."""
    try:
        embedder = OllamaEmbedder()
        corpus = GraphRAGCorpus.load(corpus_name, embedder)
        results = corpus.query(user_input, n_results=3)
        
        context = ""
        for result in results:
            context += f"Relevant Information (Similarity: {result['similarity']:.4f}):\n{result['content']}\n\n"
        
        return context.strip() if context else None
    except Exception as e:
        st.error(f"Error querying GraphRAG corpus: {str(e)}")
        return None

def extract_content_blocks(text):
    if text is None:
        return [], []
    
    # Extract code blocks
    code_blocks = re.findall(r'```[\s\S]*?```', text)
    
    # Remove code blocks from the text
    text_without_code = re.sub(r'```[\s\S]*?```', '', text)
    
    # Extract article blocks that start with 'Title:' and include everything until the next 'Title:' or the end of the text
    article_blocks = re.findall(r'^Title:.*?(?=^Title:|\Z)', text_without_code, re.MULTILINE | re.DOTALL)
    
    return [block.strip('`').strip() for block in code_blocks], [block.strip() for block in article_blocks]

def construct_agent_prompt(agent_type, metacognitive_type, voice_type):
    prompt = ""
    
    if agent_type != "None":
        prompt += f"You are a {agent_type}. {get_agent_prompt()[agent_type]}\n\n"
    else:
        prompt += "You are a helpful AI assistant.\n\n"
    
    if metacognitive_type != "None":
        prompt += f"Use the following metacognitive approach: {get_metacognitive_prompt()[metacognitive_type]}\n\n"
    
    if voice_type != "None":
        prompt += f"Speak in the following voice: {get_voice_prompt()[voice_type]}\n\n"
    
    prompt += """When you generate code or an article, it will be automatically saved to the user's Workspace. 
    For code, use triple backticks (```) to enclose the code block. 
    For articles, start with 'Title:' followed by the article title on a new line, then the content.
    Keep the formatting clean and consistent for both code and articles.\n\n"""
    
    return prompt

def chat_interface():
    load_settings()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "workspace_items" not in st.session_state:
        st.session_state.workspace_items = []
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""
    if "suggested_prompt" not in st.session_state:
        st.session_state.suggested_prompt = ""
    if "show_prompt_modal" not in st.session_state:
        st.session_state.show_prompt_modal = False

    st.session_state.agent_type = st.session_state.get("agent_type", "None")
    st.session_state.metacognitive_type = st.session_state.get("metacognitive_type", "None")
    st.session_state.voice_type = st.session_state.get("voice_type", "None")

    if "selected_model" not in st.session_state:
        available_models = get_available_models()
        st.session_state.selected_model = st.session_state.get("selected_model", available_models[0] if available_models else None)

    st.session_state.selected_corpus = st.session_state.get("selected_corpus", "None")
    st.session_state.temperature_slider_chat = st.session_state.get("temperature_slider_chat", 0.7)
    st.session_state.max_tokens_slider_chat = st.session_state.get("max_tokens_slider_chat", 4000)
    st.session_state.presence_penalty_slider_chat = st.session_state.get("presence_penalty_slider_chat", 0.0)
    st.session_state.frequency_penalty_slider_chat = st.session_state.get("frequency_penalty_slider_chat", 0.0)

    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0

    with st.sidebar:
        with st.expander("⚙️ Chat Agent Settings", expanded=False):
            available_models = get_all_models()  # Update available models
            st.session_state.selected_model = st.selectbox(
                "📦 Model:",
                available_models,
                index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
            )
            agent_types = ["None"] + list(get_agent_prompt().keys())
            st.session_state.agent_type = st.selectbox("🧑‍🔧 Agent Type:", agent_types, index=agent_types.index(st.session_state.agent_type))
            metacognitive_types = ["None"] + list(get_metacognitive_prompt().keys())
            st.session_state.metacognitive_type = st.selectbox("🧠 Metacognitive Type:", metacognitive_types, index=metacognitive_types.index(st.session_state.metacognitive_type))
            voice_types = ["None"] + list(get_voice_prompt().keys())
            st.session_state.voice_type = st.selectbox("🗣️ Voice Type:", voice_types, index=voice_types.index(st.session_state.voice_type))
            corpus_options = ["None"] + [d for d in os.listdir(RAGTEST_DIR) if os.path.isdir(os.path.join(RAGTEST_DIR, d))]
            st.session_state.selected_corpus = st.selectbox("📚 Corpus:", corpus_options, index=corpus_options.index(st.session_state.selected_corpus) if st.session_state.selected_corpus in corpus_options else 0)
            st.button("💾 Save Settings", key="save_settings_general", on_click=save_settings)

        with st.expander("🛠️ Advanced Settings", expanded=False):
            st.session_state.temperature_slider_chat = st.slider("🌡️ Temperature", min_value=0.0, max_value=1.0, value=st.session_state.temperature_slider_chat, step=0.1)
            st.session_state.max_tokens_slider_chat = st.slider("📊 Max Tokens", min_value=1000, max_value=128000, value=st.session_state.max_tokens_slider_chat, step=1000)
            st.session_state.presence_penalty_slider_chat = st.slider("🚫 Presence Penalty", min_value=-2.0, max_value=2.0, value=st.session_state.presence_penalty_slider_chat, step=0.1)
            st.session_state.frequency_penalty_slider_chat = st.slider("🔁 Frequency Penalty", min_value=-2.0, max_value=2.0, value=st.session_state.frequency_penalty_slider_chat, step=0.1)
            st.button("💾 Save Settings", key="save_settings_advanced", on_click=save_settings)

        with st.expander("📁 Saved Chats", expanded=False):
            manage_saved_chats()

        if st.button("📥 Save Chat"):
            save_chat_and_workspace()

    chat_tab, workspace_tab = st.tabs(["💬 Chat", "📜 Workspace"])

    with chat_tab:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    if message.get("content"):  # Check if content exists and is not None
                        code_blocks = extract_code_blocks(message["content"])
                        for code_block in code_blocks:
                            st.code(code_block)
                        non_code_parts = re.split(r'```[\s\S]*?```', message["content"])
                        for part in non_code_parts:
                            st.markdown(part.strip())
                    else:
                        st.warning("This message has no content.")
                else:
                    if message.get("content"):  # Check if content exists and is not None
                        st.markdown(message["content"])
                    else:
                        st.warning("This message has no content.")

        user_input_placeholder = st.empty()
        response_placeholder = st.empty()

    with bottom():
        col1, col2 = st.columns([1, 20])
        with col1:
            if st.button("✨", key="prompt_helper", help="Need help writing a prompt?"):
                st.session_state.show_prompt_modal = True
                st.rerun()
        with col2:
            user_input = st.chat_input("What is up my person?")

    if st.session_state.get("show_prompt_modal", False):
        ai_assisted_prompt_writing()

    if st.session_state.chat_input:
        user_input = st.session_state.chat_input
        st.session_state.chat_input = ""

    if user_input:
        api_keys = load_api_keys()
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.total_tokens += count_tokens(user_input)

        agent_prompt = construct_agent_prompt(
            st.session_state.agent_type,
            st.session_state.metacognitive_type,
            st.session_state.voice_type
        )

        chat_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_history[-5:]])  # Include only the last 5 messages

        corpus_context = ""
        if st.session_state.selected_corpus != "None":
            graph_response = get_graphrag_context(user_input, st.session_state.selected_corpus)
            if graph_response:
                corpus_context = f"\nRelevant context from the knowledge base:\n{graph_response}\n"
            else:
                st.warning(f"No relevant context found in the corpus '{st.session_state.selected_corpus}'. Proceeding without additional context.")

        final_prompt = f"""
        {agent_prompt}
        
        Recent conversation history:
        {chat_history}
        
        {corpus_context}
        
        Human: {user_input}
        
        Assistant: Let me address your request based on the information provided and my capabilities.
        """

        st.session_state.total_tokens += count_tokens(final_prompt)
        st.info(f"Total Token Count: {st.session_state.total_tokens}")

        with response_placeholder.container():
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                if st.session_state.selected_model in OPENAI_MODELS:
                    full_response = call_openai_api(
                        st.session_state.selected_model,
                        [{"role": "user", "content": final_prompt}],
                        temperature=st.session_state.temperature_slider_chat,
                        max_tokens=st.session_state.max_tokens_slider_chat,
                        openai_api_key=api_keys.get("openai_api_key")
                    )
                elif st.session_state.selected_model in GROQ_MODELS:
                    full_response = call_groq_api(
                        st.session_state.selected_model,
                        final_prompt,
                        temperature=st.session_state.temperature_slider_chat,
                        max_tokens=st.session_state.max_tokens_slider_chat,
                        groq_api_key=api_keys.get("groq_api_key")
                    )
                else:
                    for response_chunk in ollama.generate(
                        st.session_state.selected_model,
                        final_prompt,
                        stream=True,
                        options={
                            "temperature": st.session_state.temperature_slider_chat,
                            "num_predict": st.session_state.max_tokens_slider_chat,
                            "presence_penalty": st.session_state.presence_penalty_slider_chat,
                            "frequency_penalty": st.session_state.frequency_penalty_slider_chat,
                        }
                    ):
                        full_response += response_chunk["response"]
                        message_placeholder.markdown(full_response + "▌")
                        st.session_state.total_tokens += count_tokens(response_chunk["response"])
                message_placeholder.markdown(full_response)

        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

        code_blocks, article_blocks = extract_content_blocks(full_response)
        
        for code_block in code_blocks:
            st.session_state.workspace_items.append({
                "type": "code",
                "content": code_block,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        for article_block in article_blocks:
            st.session_state.workspace_items.append({
                "type": "article",
                "content": article_block,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        if code_blocks or article_blocks:
            st.success(f"{len(code_blocks)} code block(s) and {len(article_blocks)} article(s) automatically saved to Workspace")

    with workspace_tab:
        for index, item in enumerate(st.session_state.workspace_items):
            with st.expander(f"Item {index + 1} - {item['timestamp']}"):
                if item['type'] == 'code':
                    st.code(item['content'])
                elif item['type'] == 'article':
                    lines = item['content'].split('\n')
                    st.subheader(lines[0].replace('Title:', '').strip())
                    st.markdown('\n'.join(lines[1:]))
                else:
                    st.write(item['content'])
                if st.button(f"Remove Item {index + 1}"):
                    st.session_state.workspace_items.pop(index)
                    st.rerun()

        new_item = st.text_area("Add a new item to the workspace:", key="new_workspace_item")
        if st.button("✚ Add to Workspace"):
            if new_item:
                st.session_state.workspace_items.append({
                    "type": "text",
                    "content": new_item,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.success("New item added to Workspace")
                st.rerun()

def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def extract_code_blocks(text):
    if text is None:
        return []  # Return an empty list if text is None
    code_blocks = re.findall(r'```[\s\S]*?```', text)
    return [block.strip('`').strip() for block in code_blocks]

def save_chat_and_workspace():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"{timestamp}"
    chat_name = st.text_input("Enter a name for the save:", value=default_filename, key="save_chat_name")
    if chat_name:
        save_data = {
            "chat_history": st.session_state.chat_history,
            "workspace_items": st.session_state.workspace_items,
            "total_tokens": st.session_state.total_tokens
        }
        sessions_folder = "sessions"
        if not os.path.exists(sessions_folder):
            os.makedirs(sessions_folder)
        file_path = os.path.join(sessions_folder, chat_name + ".json")
        with open(file_path, "w") as f:
            json.dump(save_data, f)
        st.success(f"Chat and Workspace saved to {chat_name}")

def manage_saved_chats():
    st.sidebar.subheader("Saved Chats and Workspaces")
    sessions_folder = "sessions"
    if not os.path.exists(sessions_folder):
        os.makedirs(sessions_folder)
    saved_files = [f for f in os.listdir(sessions_folder) if f.endswith(".json")]

    if "rename_file" not in st.session_state:
        st.session_state.rename_file = None

    for file in saved_files:
        col1, col2, col3 = st.sidebar.columns([3, 1, 1])
        with col1:
            file_name = os.path.splitext(file)[0]
            if st.button(file_name):
                load_chat_and_workspace(os.path.join(sessions_folder, file))
        with col2:
            if st.button("✏️", key=f"rename_{file}"):
                st.session_state.rename_file = file
                st.rerun()
        with col3:
            if st.button("🗑️", key=f"delete_{file}"):
                delete_chat_and_workspace(os.path.join(sessions_folder, file))

    if st.session_state.rename_file:
        rename_chat_and_workspace(st.session_state.rename_file, sessions_folder)

def load_chat_and_workspace(file_path):
    with open(file_path, "r") as f:
        loaded_data = json.load(f)
    st.session_state.chat_history = loaded_data.get("chat_history", [])
    st.session_state.workspace_items = loaded_data.get("workspace_items", [])
    st.session_state.total_tokens = loaded_data.get("total_tokens", 0)
    st.success(f"Loaded {os.path.basename(file_path)}")
    st.rerun()

def delete_chat_and_workspace(file_path):
    os.remove(file_path)
    st.success(f"File {os.path.basename(file_path)} deleted.")
    st.rerun()

def rename_chat_and_workspace(file_to_rename, sessions_folder):
    current_name = os.path.splitext(file_to_rename)[0]
    new_name = st.sidebar.text_input("Rename file:", value=current_name, key="rename_file_input")
    if st.sidebar.button("Confirm Rename"):
        if new_name and new_name != current_name:
            old_file_path = os.path.join(sessions_folder, file_to_rename)
            new_file_path = os.path.join(sessions_folder, new_name + ".json")
            if new_file_path != old_file_path:
                os.rename(old_file_path, new_file_path)
                st.sidebar.success(f"File renamed to {new_name}")
                st.session_state.rename_file = None
                st.rerun()
        else:
            st.sidebar.error("Please enter a new name different from the current one.")
    
    if st.sidebar.button("Cancel"):
        st.session_state.rename_file = None
        st.rerun()

if __name__ == "__main__":
    chat_interface()