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
            
            #### 🖼️ Multimodal Chat
            **[New!]** Engage with models that can see and understand images:
            - 🖼️ **Image Upload:** Upload images in various formats including JPG, PNG, and WebP.
            - 🔍 **Image Analysis:** Have the model describe, analyze, or answer questions about images.
            - 💬 **Multimodal Conversation:** Seamlessly mix text and images in the same conversation.
            - 📊 **Chart Understanding:** Use visual models to interpret and explain charts, graphs, and diagrams.
            
            #### 🧰 Tool Playground
            **[New!]** Experiment with function calling and tool integration:
            - 🔧 **Predefined Tools:** Test models with built-in tools like calculator, weather API, search, and more.
            - 🛠️ **Custom Tools:** Create and test your own custom tools with JSON schema definitions.
            - 🔌 **MCP Tools Integration:** Leverage Model Control Protocol tools from your local system.
            - 📊 **Tool Execution:** See how models use tools to solve problems and return structured information.

            #### 🔍 Structured Output
            **[New!]** Generate structured data from text:
            - 📋 **JSON Schema:** Define the structure of the output using JSON schema.
            - 🧩 **Schema Templates:** Choose from predefined schemas for common data structures.
            - 🔄 **Custom Schemas:** Create and save your own custom schemas.
            - 📊 **Visualization:** View the generated structured data in different formats.

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
            - ☁️ **External Providers:** Configure and manage API keys for external services like OpenAI, Groq, and various search providers, giving you access to a wider array of AI models and search features.
            - 🔄 **OpenAI Compatibility:** **[New!]** Run an OpenAI-compatible API server that translates OpenAI API calls to Ollama, allowing you to use OpenAI client libraries with Ollama models.

            #### 📊 Test
            Evaluate and compare your AI models:

            - 🛠️ **Model Onboarding:** **[New!]** A comprehensive test suite for evaluating and onboarding new models.
            - 🧪 **Model Feature Test**: Test a model's capability to handle JSON and function calls.
            - 🎯 **Model Comparison by Response Quality**: Compare the response quality and performance of multiple models for a given prompt.
            - 💬 **Contextual Response Test by Model**: Test how well a model maintains context across multiple prompts.
            - 👁️ **Vision Model Comparison**: Compare the performance of vision models using the same test image.
            - 🧰 **Tool Calling:** **[New!]** Test a model's ability to use tools and function calling.
            - 🔍 **Structured Output:** **[New!]** Test a model's ability to generate structured data based on JSON schema.
            - 🔎 **Model Capabilities:** **[New!]** Discover and compare the capabilities of different models with a comprehensive testing framework.
            - 📊 **Test Visualization:** **[New!]** Visualize and analyze test results with interactive charts and tables.
        """)
        
        st.markdown("---")
        
        st.subheader("🆕 What's New")
        
        st.markdown("""
        #### Latest Features (May 2024)
        
        1. **Updated to Ollama 0.7.0+** - Compatibility with the latest Ollama API
        2. **Multimodal Support** - Chat with models that can see and understand images
        3. **Tool Calling Integration** - Use models with function calling and tool integration
        4. **MCP Tools Support** - Leverage Model Control Protocol tools from your local system
        5. **Structured Output Generation** - Generate structured JSON data from text
        6. **OpenAI Compatibility Layer** - Use OpenAI client libraries with Ollama models
        7. **Model Capabilities Discovery** - Discover and compare model capabilities
        8. **Enhanced Test Visualization** - Better visualizations for test results
        9. **Centralized Configuration** - Environment variables and configuration management
        10. **Improved Error Handling** - Better error handling and user feedback
        """)
        
        st.markdown("---")
        
        st.subheader("💡 Tips")
        
        st.markdown("""
        - **Experiment with Different Models**: Ollama supports a wide range of models. Try different ones to find the best fit for your specific tasks.
        - **Combine Workflows**: Use the Research feature to gather information, then use that information in a Brainstorming session to generate new ideas.
        - **Use Tool Calling**: For complex tasks, try using models with tool calling capabilities for more accurate and detailed results.
        - **Leverage MCP Tools**: If you have specialized tools on your local system, use the MCP tools integration to make them available to your models.
        - **Structure Your Outputs**: When you need structured data, use the Structured Output feature to generate consistent, well-formatted results.
        - **Test Before Deploying**: Use the testing tools to evaluate model performance before using them in production.
        - **Save Your Sessions**: Don't forget to save your chat sessions and workspaces for future reference.
        - **Explore the Documentation**: Check out the Help section for more detailed information on each feature.
        """)

    with col2:
        st.image("https://2acrestudios.com/wp-content/uploads/2024/07/00010-3993212168.png", width=200)
        
        st.markdown("---")
        
        st.subheader("🔗 Quick Links")
        
        st.markdown("""
        - [Chat](#/Chat)
        - [Multimodal Chat](#/Multimodal%20Chat)
        - [Tool Playground](#/Tool%20Playground)
        - [Structured Output](#/Structured%20Output)
        - [Model Capabilities](#/Model%20Capabilities)
        - [Server Configuration](#/Server%20Configuration)
        """)
        
        st.markdown("---")
        
        st.subheader("📚 Resources")
        
        st.markdown("""
        - [Ollama Documentation](https://ollama.ai/documentation)
        - [Ollama Models Library](https://ollama.ai/library)
        - [Ollama GitHub Repository](https://github.com/ollama/ollama)
        - [Ollama Workbench GitHub](https://github.com/your-username/ollama-workbench)
        """)