# chat_interface.py
import streamlit as st
import os
import json
from datetime import datetime
import re
import ollama
from ollama_utils import get_available_models
from prompts import get_agent_prompt, get_metacognitive_prompt, get_voice_prompt
import tiktoken
from files_management import files_tab  # Import from files_management.py
from streamlit_extras.bottom_container import bottom  # Correct import

def chat_interface():
    st.header("🧑 Chat With a Custom Agent")

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "workspace_items" not in st.session_state:
        st.session_state.workspace_items = []
    if "agent_type" not in st.session_state:
        st.session_state.agent_type = "None"
    if "metacognitive_type" not in st.session_state:
        st.session_state.metacognitive_type = "None"
    if "voice_type" not in st.session_state:
        st.session_state.voice_type = "None"
    if "selected_model" not in st.session_state:
        available_models = get_available_models()
        st.session_state.selected_model = available_models[0] if available_models else None
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0

    # Create tabs for Chat, Workspace, and Files
    chat_tab, workspace_tab, files_tab_ui = st.tabs(["Chat", "Workspace", "Files"])

    with chat_tab:
        # Settings (Collapsible, open by default)
        with st.expander("⚙️ Settings", expanded=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                available_models = get_available_models()
                selected_model = st.selectbox(
                    "📦 Select a Model:",
                    available_models,
                    key="selected_model",
                    index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
                )
            with col2:
                agent_types = ["None"] + list(get_agent_prompt().keys())
                agent_type = st.selectbox("🧑‍🔧 Select Agent Type:", agent_types, key="agent_type")
            with col3:
                metacognitive_types = ["None"] + list(get_metacognitive_prompt().keys())
                metacognitive_type = st.selectbox("🧠 Select Metacognitive Type:", metacognitive_types, key="metacognitive_type")
            with col4:
                voice_types = ["None"] + list(get_voice_prompt().keys())
                voice_type = st.selectbox("🗣️ Select Voice Type:", voice_types, key="voice_type")
            with col5:
                corpus_folder = "corpus"
                if not os.path.exists(corpus_folder):
                    os.makedirs(corpus_folder)
                corpus_options = ["None"] + [f for f in os.listdir(corpus_folder) if os.path.isdir(os.path.join(corpus_folder, f))]
                selected_corpus = st.selectbox("📚 Select Corpus:", corpus_options, key="selected_corpus")

        # Advanced Settings (Collapsible, collapsed by default)
        with st.expander("🛠️ Advanced Settings", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                temperature = st.slider("🌡️ Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key="temperature_slider_chat")
            with col2:
                max_tokens = st.slider("📊 Max Tokens", min_value=100, max_value=32000, value=4000, step=100, key="max_tokens_slider_chat")
            with col3:
                presence_penalty = st.slider("🚫 Presence Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.1, key="presence_penalty_slider_chat")
            with col4:
                frequency_penalty = st.slider("🔁 Frequency Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.1, key="frequency_penalty_slider_chat")

        # Display total token count
        st.write(f"Total Token Count: {st.session_state.total_tokens}")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                code_blocks = extract_code_blocks(message["content"])
                for code_block in code_blocks:
                    st.code(code_block)
                non_code_parts = re.split(r'```[\s\S]*?```', message["content"])
                for part in non_code_parts:
                    st.markdown(part.strip())
            else:
                st.markdown(message["content"])

    # Create placeholders for user input and assistant's response
    user_input_placeholder = st.empty()
    response_placeholder = st.empty()

    # Chat input at the bottom using bottom_container
    with bottom():
        prompt = st.chat_input("🧐 What is up my person❔", key="chat_input")

    # Process the user input and generate response outside the bottom container
    if prompt:
        # Display user input
        with user_input_placeholder.container():
            with st.chat_message("user"):
                st.markdown(prompt)

        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Update token count for user input
        st.session_state.total_tokens += count_tokens(prompt)

        # Generate and display the assistant's response
        full_response = ""

        # Combine agent type, metacognitive type, and voice type prompts
        combined_prompt = ""
        if agent_type != "None":
            combined_prompt += get_agent_prompt()[agent_type] + "\n\n"
        if metacognitive_type != "None":
            combined_prompt += get_metacognitive_prompt()[metacognitive_type] + "\n\n"
        if voice_type != "None":
            combined_prompt += get_voice_prompt()[voice_type] + "\n\n"

        # Include chat history and corpus context
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
        if selected_corpus != "None":
            corpus_context = get_corpus_context_from_db(corpus_folder, selected_corpus, prompt)
            final_prompt = f"{combined_prompt}Conversation History:\n{chat_history}\n\nContext: {corpus_context}\n\nUser: {prompt}\n\n{combined_prompt}"
        else:
            final_prompt = f"{combined_prompt}Conversation History:\n{chat_history}\n\nUser: {prompt}\n\n{combined_prompt}"

        # Update token count for the entire prompt
        st.session_state.total_tokens += count_tokens(final_prompt)

        with response_placeholder.container():
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                for response_chunk in ollama.generate(st.session_state.selected_model, final_prompt, stream=True):
                    full_response += response_chunk["response"]
                    message_placeholder.markdown(full_response + "▌")
                    # Update token count for each chunk of the response
                    st.session_state.total_tokens += count_tokens(response_chunk["response"])
                message_placeholder.markdown(full_response)

        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

        # Update the total token count display
        st.write(f"Total Token Count: {st.session_state.total_tokens}")

        # Automatically detect and save code to workspace
        code_blocks = extract_code_blocks(full_response)
        for code_block in code_blocks:
            st.session_state.workspace_items.append({
                "type": "code",
                "content": code_block,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        if code_blocks:
            st.success(f"{len(code_blocks)} code block(s) automatically saved to Workspace")

    # Save chat and workspace button (outside the bottom container)
    if st.button("📥 Save Chat and Workspace"):
        save_chat_and_workspace()

    # Load/Rename/Delete chat and workspace
    manage_saved_chats()

    # Workspace tab
    with workspace_tab:
        st.subheader("📜 Workspace")
        
        # Display workspace items
        for index, item in enumerate(st.session_state.workspace_items):
            with st.expander(f"Item {index + 1} - {item['timestamp']}"):
                if item['type'] == 'code':
                    st.code(item['content'])
                else:
                    st.write(item['content'])
                if st.button(f"Remove Item {index + 1}"):
                    st.session_state.workspace_items.pop(index)
                    st.rerun()

        # Option to add a new workspace item manually
        new_item = st.text_area("Add a new item to the workspace:", key="new_workspace_item")
        if st.button("Add to Workspace"):
            if new_item:
                st.session_state.workspace_items.append({
                    "type": "text",
                    "content": new_item,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.success("New item added to Workspace")
                st.rerun()

    # Files tab
    with files_tab_ui:
        files_tab()

def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def extract_code_blocks(text):
    code_blocks = re.findall(r'```[\s\S]*?```', text)
    return [block.strip('`').strip() for block in code_blocks]

def get_corpus_context_from_db(corpus_folder, corpus_name, query):
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma
    corpus_path = os.path.join(corpus_folder, corpus_name)
    embeddings = OllamaEmbeddings()
    db = Chroma(persist_directory=corpus_path, embedding_function=embeddings)
    results = db.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in results])

def save_chat_and_workspace():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"chat_and_workspace_{timestamp}"
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
    new_name = st.text_input("Rename file:", value=current_name, key="rename_file_input")
    if new_name:
        old_file_path = os.path.join(sessions_folder, file_to_rename)
        new_file_path = os.path.join(sessions_folder, new_name + ".json")
        if new_file_path != old_file_path:
            os.rename(old_file_path, new_file_path)
            st.success(f"File renamed to {new_name}")
            st.session_state.rename_file = None
            st.cache_resource.clear()
            st.rerun()