# welcome.py
import streamlit as st

def display_welcome_message():
    """Display the welcome message and feature overview."""
    col1, col2 = st.columns([3, 1]) # Adjust column ratios

    with col1:
        st.write("""
            ### About Ollama Workbench
            This application provides tools for managing, testing, and interacting with your Ollama models. Explore its powerful features designed to streamline your AI workflows and enhance your productivity.

            #### 💬 Chat
            Engage in a real-time chat with a selected model, enhanced with various features:
            - 🧑‍🔧 **Agent Types:** Choose from a variety of predefined agent types, each with specific prompts to guide the model's behavior (e.g., Coder, Analyst, Creative Writer).
            - 🧠 **Metacognitive Types:** Enhance the model's reasoning abilities by selecting a metacognitive type (e.g., Visualization of Thought, Chain of Thought).
            - 🗣️ **Voice Types:** Choose a voice type to guide the tone and style of the model's responses (e.g., Friendly, Professional, Humorous).
            - 📚 **Corpus Integration:** Load a corpus of text to provide contextual information to the model, improving its responses using the power of retrieval augmented generation (RAG).
            - 🛠️ **Advanced Settings:** Fine-tune the model's output by adjusting parameters like temperature, max tokens, presence penalty, and frequency penalty.
            - 📜 **Workspace:** Save and manage code snippets and other text generated during your chat sessions.
            - 💾 **Save/Load Sessions:** Save your chat history and workspace for later use, or load previously saved sessions.

            #### 🔄 Workflow
            Ollama Workbench offers a range of tools to design and execute complex AI workflows:

            - 🔬 **Research:** Conduct in-depth research using multiple AI agents, each employing a different search approach and utilizing diverse search libraries (DuckDuckGo, Google, Bing, SerpApi, Serper). Agents generate summaries compiled into a comprehensive report with references.
            - 🧠 **Brainstorm:** Engage in interactive brainstorming sessions with multiple AI agents, each with unique roles and perspectives. The agents leverage AutoGen's 'Teachable Agent' memory feature, remembering your past brainstorming sessions for enhanced context and continuity.
            - 🚀 **Projects:** Create, edit, delete, import, and export projects and tasks. Assign AI agents to tasks and automatically generate responses, streamlining your project management workflow.
            - 🔨 **Build:** **[New!]** An autonomous multi-agent software development system that manages the entire lifecycle of software projects. Specialized agents handle tasks like project planning, code generation, testing, documentation, and quality review.
            - ✳️ **Nodes:** **[New!]** A visual workflow builder. Design and execute workflows by connecting different nodes representing AI models, inputs, and outputs. Configure each node's settings, including prompt engineering, model selection, and output formatting. Create powerful processing pipelines by chaining together multiple AI models.
            - ✨ **Prompts:** Create, edit, and delete custom prompts for Agent Type, Metacognitive Type, Voice Type, and Identity, enabling highly specialized AI behaviors and granular control over agent interactions.

            #### 🗄️ Document
            Manage your knowledge base and streamline document creation:

            - 🗂️ **Manage Corpus:** Create, edit, delete, and rename corpus from files, URLs, or text input, enhancing AI models' contextual knowledge for tasks like question answering and summarization.
            - 📂 **Manage Files**: Upload, view, edit, and delete files, providing centralized file management for use across the platform.
            - 🕸️ **Web to Corpus File**: Convert web content into a corpus for analysis or model training, allowing you to tailor AI models to specific domains using online information.
            - 🔍 **Repository Analyzer**: Analyze Python repositories and automatically generate documentation, debug reports, or a README.md file, streamlining the documentation process and maintaining code quality. 

            #### 🛠️ Maintain
            Keep your Ollama environment running smoothly:

            - 🤖 **List Local Models:** View a list of all locally available models, including their size and last modified date.
            - 🦙 **Show Model Information:** Display detailed information about a selected model.
            - ⬇ **Pull a Model:** Download a new model from the Ollama library, expanding your available AI capabilities.
            - 🗑️ **Remove a Model**: Delete a selected model from local storage to manage disk space.
            - ⤵️ **Update Models**: Update all local models to ensure you're using the latest versions.
            - ⚙️ **Server Configuration:** Configure the Ollama server settings.
            - 🖥️ **Server Monitoring:** Monitor the Ollama server's resource usage.
            - ☁️ **External Providers:** **[New!]** Configure and manage API keys for external services like OpenAI, Groq, and various search providers, giving you access to a wider array of AI models and search features. 

            #### 📊 Test
            Evaluate and compare your AI models:

            - 🧪 **Model Feature Test**: Test a model's capability to handle JSON and function calls.
            - 🎯 **Model Comparison by Response Quality**: Compare the response quality and performance of multiple models for a given prompt.
            - 💬 **Contextual Response Test by Model**: Test how well a model maintains context across multiple prompts.
            - 👁️ **Vision Model Comparison**: Compare the performance of vision models using the same test image.
        """)

    with col2:
        st.html("<img src='https://2acrestudios.com/wp-content/uploads/2024/07/00010-3993212168.png' style='max-width: 200px;' />") 