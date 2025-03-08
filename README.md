# Ollama Workbench

<img src="https://2acrestudios.com/wp-content/uploads/2024/06/00001-2881912941.png" style="width: 300px;" align="right" />

Ollama Workbench is a powerful and versatile platform designed to streamline the management, testing, and utilization of various AI models from the Ollama library. It transcends simple model testing, offering advanced features for crafting highly tunable AI agents, orchestrating complex workflows, and facilitating dynamic collaborative brainstorming sessions. With its intuitive interface, Ollama Workbench empowers both novice programmers and experienced developers to harness the full potential of their machine learning models in innovative and productive ways.

**Table of Contents**
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

### 💬 Chat and Advanced Agent Interaction

Engage in sophisticated interactions with AI models through a feature-rich chat interface:

- 🧑‍🔧 **Customizable Agent Types:** Choose from pre-defined agent types or create your own, each with specific prompts to guide the model's behavior (e.g., Coder, Analyst, Creative Writer).
- 🧠 **Metacognitive Enhancements:** Boost the model's reasoning abilities with selectable metacognitive types (e.g., Visualization of Thought, Chain of Thought).
- 🗣️ **Voice and Personality Customization:** Tailor the tone and style of the model's responses with customizable voice types.
- 📚 **Dynamic Corpus Integration:** Enhance model responses by loading relevant text corpora for contextual information, leveraging the power of retrieval augmented generation (RAG).
- 🛠️ **Fine-grained Control:** Adjust advanced parameters like temperature, max tokens, presence penalty, and frequency penalty for precise output tuning.
- 📜 **Integrated Workspace:** Efficiently manage and save code snippets and text generated during sessions.
- 💾 **Session Management:** Save and load entire chat histories and workspaces for seamless workflow continuity.


<img src="https://github.com/user-attachments/assets/7f55c77e-9e3e-4b96-b27f-84615b4ba21c" />


### ⚙️ Advanced Workflows and Collaboration

Ollama Workbench offers powerful tools for designing and executing complex workflows, enabling seamless collaboration between AI agents:

- 🧠 **Brainstorm Mode:**  Create and manage teams of specialized AI agents for collaborative ideation and problem-solving. This mode leverages AutoGen's 'Teachable Agent' memory feature, allowing agents to recall past brainstorming sessions for enhanced continuity and context.
- 🔨 **Build:**  A groundbreaking new feature! Build is an autonomous multi-agent software development system that manages the entire lifecycle of software projects. It uses specialized agents for tasks such as project planning, code generation, testing, documentation, and quality review.
- 🔬 **Research:** Conduct in-depth research using multiple AI agents, each specializing in a different search approach and utilizing diverse search libraries like DuckDuckGo, Google, Bing, SerpApi, and Serper. The agents generate summaries of their findings, which are then compiled into a final, comprehensive report with references.
- 🚀 **Project Management:** Develop, manage, and execute complex projects with multiple AI agents. Configure each agent with specific models, types, parameters, and even assign them individual tasks within a project.
- 🤖 **AI-Assisted Project Planning:**  Generate tasks, assign agents, and create project structures using natural language descriptions. Ollama Workbench can understand your project goals and intelligently suggest tasks and agent assignments.
- ✨ **Custom Prompt Management:** Create, edit, and manage custom prompts for various agent types, enabling highly specialized AI behaviors. This provides granular control over agent interaction and output. 
- ✳️ **Nodes:**  A visual workflow builder! Design and execute workflows by connecting different nodes representing AI models, inputs, and outputs. Configure each node's settings, including prompt engineering, model selection, and output formatting. This feature provides a powerful and flexible way to chain together AI models and create custom processing pipelines.

### 🗄️ Document and Knowledge Management

- 🗂️ **Corpus Management:** Create, edit, and curate text corpora to enhance AI model knowledge and context, improving performance in tasks like question answering and text summarization. 
- 📂 **File Management System:**  A comprehensive system for uploading, viewing, editing, and organizing files used across the platform. This centralizes file management, making it easy to access and share data between different features.
- 🕸️ **Web Content Integration:** Convert web content into usable corpora for analysis or model training. This allows you to leverage online information and tailor AI models to specific domains.
- ✔️ **Intelligent Repository Analysis:** Analyze Python repositories to generate documentation, debug reports, and README files with real-time UI feedback. This feature automates the documentation process and helps maintain code quality.

### 🛠️ Maintain

- 📋 **List Local Models:** View a list of all locally available models, including their size and last modified date.
- 🦙 **Show Model Information:** Display detailed information about a selected model, such as architecture, parameters, and training data.
- ⬇ **Pull a Model:** Download a new model from the Ollama library, expanding your available AI capabilities. 
- 🗑️ **Remove a Model**: Delete a selected model from local storage to manage disk space.
- 🔄 **Update Models**: Update all local models to ensure you're using the latest versions.
- ⚙️ **Server Configuration:** Configure the Ollama server settings, including host address, allowed origins, model directory, global keep-alive, and concurrency controls.
- 🖥️ **Server Monitoring:** Monitor the Ollama server's resource usage, including CPU, memory, and GPU utilization, as well as view live server logs for troubleshooting.
- ☁️ **External Providers:** Configure and manage API keys for external services like OpenAI, Groq, SerpApi, Serper, Google Custom Search, and Bing Search, allowing you to access a wider range of AI models and search capabilities.

### 📊 Test

- 🧪 **Model Feature Test**: Test a model's capability to handle JSON and function calls, essential for advanced AI applications.
- 🎯 **Model Comparison by Response Quality**:  Compare the response quality and performance of multiple models for a given prompt, helping you choose the best model for your needs.
- 💬 **Contextual Response Test by Model**: Test how well a model maintains context across multiple prompts, assessing its ability to understand and respond coherently in ongoing conversations.
- 👁️ **Vision Model Comparison**: Compare the performance of vision models by uploading a test image, evaluating their capabilities in image understanding and description.

## Installation

### Prerequisites

* **Python 3.11:** Ollama Workbench requires Python 3.11 specifically for optimal compatibility. Other versions (like 3.12 or 3.13) may cause dependency issues.
* **Ollama Server:** Ollama Workbench requires Ollama Server to be installed and running. The setup script will help you install it if needed.

### Simple 2-Step Installation

1. **Set up the environment:**
   ```bash
   ./setup.sh
   ```
   This automated script:
   - Finds or installs Python 3.11
   - Creates a properly configured virtual environment
   - Installs all dependencies in the correct order
   - Creates a launcher script

2. **Run Ollama Workbench:**
   ```bash
   ./run_ollama_workbench.sh
   ```
   The launcher automatically ensures everything is set up correctly before starting the application.

### What the Setup Script Does

The `setup.sh` script handles all the complexity for you:

- Finds Python 3.11 on your system or installs it if needed
- Creates a dedicated virtual environment with the exact correct Python version
- Installs all dependencies with proper version compatibility
- Installs spaCy and downloads the required language model
- Installs the Google API Client Library for search functionality
- Optimizes installation for your platform (special handling for Apple Silicon)
- Creates a launcher script that ensures everything runs correctly

### Troubleshooting

If you encounter any issues:

- **"Python 3.11 not found"**: The script will attempt to install it automatically, but you may need to install Python 3.11 manually from [python.org](https://www.python.org/downloads/) or using your system's package manager.
- **Import errors**: Run the setup script again to recreate the virtual environment.
- **Ollama not running**: The launcher will offer to install Ollama if it's not found and start the Ollama server if needed.

For Apple Silicon Mac users, please refer to [APPLE_SILICON_SETUP.md](APPLE_SILICON_SETUP.md) for additional information.

## Usage

**Launching the Application:**

* Use the `run_ollama_workbench.sh` script to launch the app:
   ```bash
   ./run_ollama_workbench.sh
   ```

**Navigating the Interface:**

* **Sidebar:**  Access all functionalities through the intuitive sidebar.
* **Main Content Area:** The main content area displays the selected feature, whether it's the chat interface, model comparison tools, or workflow management. 

**Key Sections:**

* **Chat:** Engage in conversations, experiment with agent types, and manage your workspace.
* **Workflow:** Design and execute complex workflows, collaborate with AI agents in Brainstorm mode, manage projects, and customize prompts.
* **Document:** Manage your knowledge base with corpus management, leverage the web crawler to build corpora, analyze Python repositories, and organize your files.
* **Maintain:**  View, pull, remove, and update your Ollama models, configure your server, monitor its performance, and manage API keys for external providers. 
* **Test:** Put your models through their paces with feature tests, compare performance across different models, assess contextual response, and evaluate vision capabilities.

**Additional Tips:**

* **Explore the Prompts:** The prompts provided for agent types, metacognitive types, and voice types are highly customizable. Experiment with modifying them to create even more specialized AI behaviors.
* **Stay Updated:** Use the `run_ollama_workbench.sh` script to ensure you always have the latest features and improvements.

## Contributing

Contributions are welcome!  

1. Fork the repository.
2. Create a new branch (`git checkout -b my-new-feature`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin my-new-feature`).
6. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
