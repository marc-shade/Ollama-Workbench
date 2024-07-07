import streamlit as st

def display_welcome_message():
    """Display the welcome message and feature overview."""
    col1, col2 = st.columns([3, 1]) # Adjust column ratios

    with col1:
        st.write("""
            ### Welcome to the Ollama Workbench!
            This application provides tools for managing, testing, and interacting with your Ollama models.

            #### 💬 Chat
            Engage in a real-time chat with a selected model, enhanced with various features:
            - 🧑‍🔧 **Agent Types:** Choose from a variety of predefined agent types, each with specific prompts to guide the model's behavior (e.g., Coder, Analyst, Creative Writer).
            - 🧠 **Metacognitive Types:** Enhance the model's reasoning abilities by selecting a metacognitive type (e.g., Visualization of Thought, Chain of Thought).
            - 🗣️ **Voice Types:** Choose a voice type to guide the tone and style of the model's responses (e.g., Friendly, Professional, Humorous).
            - 📚 **Corpus Integration:** Load a corpus of text to provide contextual information to the model, improving its responses.
            - 🛠️ **Advanced Settings:** Fine-tune the model's output by adjusting parameters like temperature, max tokens, presence penalty, and frequency penalty.
            - 📜 **Workspace:** Save and manage code snippets and other text generated during your chat sessions.
            - 💾 **Save/Load Sessions:** Save your chat history and workspace for later use, or load previously saved sessions.

            #### ⚙️ Workflow
            - 🧠 **Brainstorm:** Engage in interactive brainstorming sessions with multiple AI agents, each with unique roles and perspectives. The agents use AutoGen's 'Teachable Agent' memory feature to remember your brainstoring sessions.
            - 🚀 **Projects:** Create, edit, delete, import, and export projects and tasks. Assign AI agents to tasks and auto-generate responses.
            - ✨ **Prompts:** Create, edit, and delete custom prompts for Agent Type, Metacognitive Type, and Voice Type.

            #### 🗄️ Document
            - 🗂️ **Manage Corpus:** Create, edit, and delete corpus from files.
            - 📂 **Manage Files**: Upload, view, edit, and delete files.
            - 🕸️ **Web to Corpus File**: Convert web content into a corpus for analysis or training.
            - ✔️ **Repository Analyzer**: Analyze your Python repository, generate documentation, debug reports, or a README.md file.

            #### 🛠️ Maintain
            - 📋 **List Local Models:** View a list of all locally available models, including their size and last modified date.
            - 🦙 **Show Model Information:** Display detailed information about a selected model.
            - ⬇ **Pull a Model:** Download a new model from the Ollama library.
            - 🗑️ **Remove a Model**: Delete a selected model from the local storage.
            - 🔄 **Update Models**: Update all local models.

            #### 📊 Test
            - 🧪 **Model Feature Test**: Test a model's capability to handle JSON and function calls.
            - 🎯 **Model Comparison by Response Quality**: Compare the response quality and performance of multiple models for a given prompt.
            - 💬 **Contextual Response Test by Model**: Test how well a model maintains context across multiple prompts.
            - 👁️ **Vision Model Comparison**: Compare the performance of vision models using the same test image.
        """)
    
    with col2:
        st.html("<img src='https://2acrestudios.com/wp-content/uploads/2024/07/00010-3993212168.png' style='max-width: 200px;' />")
