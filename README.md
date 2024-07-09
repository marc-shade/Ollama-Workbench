# Ollama Workbench

<img src="https://2acrestudios.com/wp-content/uploads/2024/06/00001-2881912941.png" style="width: 300px;" align="right" />

Ollama Workbench is a powerful and versatile platform for managing, testing, and leveraging various AI models from the Ollama library. It goes beyond simple model testing, offering advanced features for creating highly tunable AI agents, orchestrating complex workflows, and facilitating collaborative brainstorming sessions. With an intuitive interface, Ollama Workbench empowers users to harness the full potential of their machine learning models in creative and productive ways.

**Table of Contents**
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

### 💬 Chat and Advanced Agent Interaction
Engage in sophisticated interactions with AI models through a feature-rich chat interface:
- 🧑‍🔧 **Customizable Agent Types:** Choose from pre-defined agent types or create your own, each with specific prompts to guide the model's behavior (e.g., Coder, Analyst, Creative Writer).
- 🧠 **Metacognitive Enhancements:** Boost the model's reasoning abilities with selectable metacognitive types (e.g., Visualization of Thought, Chain of Thought).
- 🗣️ **Voice and Personality Customization:** Tailor the tone and style of the model's responses with customizable voice types.
- 📚 **Dynamic Corpus Integration:** Enhance model responses by loading relevant text corpora for contextual information.
- 🛠️ **Fine-grained Control:** Adjust advanced parameters like temperature, max tokens, presence penalty, and frequency penalty for precise output tuning.
- 📜 **Integrated Workspace:** Efficiently manage and save code snippets and text generated during sessions.
- 💾 **Session Management:** Save and load entire chat histories and workspaces for seamless workflow continuity.

### ⚙️ Advanced Workflows and Collaboration
- 🧠 **Brainstorm Mode:** Create and manage teams of specialized AI agents for collaborative ideation and problem-solving.
- 🚀 **Project Management:** Develop, manage, and execute complex projects with multiple AI agents. Configure each agent with specific models, types, and parameters.
- 🤖 **AI-Assisted Project Planning:** Generate tasks, assign agents, and create project structures using natural language descriptions.
- ✨ **Custom Prompt Management:** Create, edit, and manage custom prompts for various agent types, enabling highly specialized AI behaviors.

### 🗄️ Document and Knowledge Management
- 🗂️ **Corpus Management:** Create, edit, and curate text corpora to enhance AI model knowledge and context.
- 📂 **File Management System:** Comprehensive system for uploading, viewing, editing, and organizing files used across the platform.
- 🕸️ **Web Content Integration:** Convert web content into usable corpora for analysis or model training.
- ✔️ **Intelligent Repository Analysis:** Analyze Python repositories to generate documentation, debug reports, and README files with real-time UI feedback.

### 🛠️ Maintain
- 📋 **List Local Models:** View a list of all locally available models, including their size and last modified date.
- 🦙 **Show Model Information:** Display detailed information about a selected model.
- ⬇ **Pull a Model:** Download a new model from the Ollama library.
- 🗑️ **Remove a Model**: Delete a selected model from the local storage.
- 🔄 **Update Models**: Update all local models.

### 📊 Test
- 🧪 **Model Feature Test**: Test a model's capability to handle JSON and function calls.
- 🎯 **Model Comparison by Response Quality**: Compare the response quality of multiple models for a given prompt.
- 💬 **Contextual Response Test by Model**: Test how well a model maintains context across multiple prompts.
- 👁️ **Vision Model Comparison**: Compare the performance of vision models using the same test image.

## Installation

Create virtual environment (optional but nice):
```bash
conda create --name ollamaworkbench python=3.11
```

```bash
conda activate ollamaworkbench
```

### Steps
1. **Clone the repository**
    ```bash
    git clone https://github.com/marc-shade/Ollama-Workbench.git
    cd Ollama-Workbench
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application (choose one)**
    ```bash
    streamlit run main.py
    ```

    ```bash
    sh run_ollama_workbench.sh
    ```

If you don't have Ollama Server installed, you can run install_ollama.sh and you'll get the best models for working with all the JSON format needed for agents and workflows to work properly.
```bash
cd Ollama-Workbench
sh install_ollama.sh
```

## Usage

### Launching the Application
After installation, you can start the application using the following command:
```bash
streamlit run main.py
```

Alternately, you can use the following sh script that runs the app with an automated update if you want to stay updated without thinking about it and always run the latest version.
```bash
sh run_ollama_workbench.sh
```

### Navigating the Interface
- Use the sidebar to access different functionalities.
- Explore various sections: Chat, Workflow, Document, Maintenance, and Testing.

### Chat
- **Chat**: Select a model and start chatting! The chat interface will automatically detect and color code blocks within the assistant's responses.
- **Custom Agent Creation:** Design specialized agents with unique combinations of agent types, metacognitive abilities, and voice characteristics.
- **Workspace**: Use the Workspace tab to save and manage code and text generated during your chat sessions. You can add new items manually or save assistant responses directly to the workspace.
- **Files**: Upload, view, edit, and delete files. These files can be used as corpus in the Chat.

### Workflow
- **Brainstorm Mode:** Create teams of AI agents for collaborative problem-solving.
- **Workflow Customization:** Create, save, and load custom workflows for repeated use.
- **Manage Projects**: Select "Manage Projects" from the Workflow section in the sidebar. You can create new projects, add tasks to projects, assign tasks to AI agents, and run the agents to generate outputs for each task.
- **Manage Prompts**: Create, edit, and delete custom prompts for Agent Type, Metacognitive Type, and Voice Type. You can also download and upload prompt JSON files for easy sharing and backup.

### Document
- **Repository Analyzer**: Select "Repository Analyzer" from the Document section in the sidebar. Enter the path to your repository and choose the task type (documentation, debug, or README). Select a model and adjust the temperature and max tokens as needed. Click "Analyze Repository" to generate a PDF report and, if applicable, a README.md file.
- **Web to Corpus**: Convert web content into a corpus for analysis or training. The generated files will be saved to the 'files' folder, accessible from both the 'Document' section and the 'Chat' section.
- **Manage Files**: Upload, view, edit, and delete files. These files can be used as corpus in the Chat.

### Maintain
- **List Local Models**: Select to view available models.
- **Show Model Information**: Choose a model to view its detailed information.
- **Pull a Model**: Enter the name of the model you wish to download.
- **Remove a Model**: Select a model and confirm removal.
- **Update Models**: Update all local models.

### Testing
- **Model Feature Test**: Test models for JSON handling and function calling capabilities.
- **Model Comparison by Response Quality**: Compare the responses of multiple models to a given prompt.
- **Contextual Response Test by Model**: Test the contextual understanding of a model through a series of prompts.
- **Vision Model Comparison**: Compare vision models by uploading an image.

<img src="https://2acrestudios.com/wp-content/uploads/2024/07/Screenshot-2024-07-07-at-10.26.14 AM-2.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/07/Screenshot-2024-07-07-at-9.41.15 AM-2-1.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/07/Screenshot-2024-07-07-at-9.42.18 AM-2.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/07/Screenshot-2024-07-07-at-10.14.03 AM-2.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/07/Screenshot-2024-07-07-at-9.42.24 AM-2.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/07/Screenshot-2024-07-07-at-9.42.45 AM-2.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/07/Screenshot-2024-07-07-at-9.42.54 AM-2.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/07/Screenshot-2024-07-07-at-9.58.22 AM-2.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/07/Screenshot-2024-07-07-at-9.58.46 AM-2.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/07/Screenshot-2024-07-07-at-9.59.18 AM-2.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/07/Screenshot-2024-07-07-at-10.00.19 AM-2.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/07/Screenshot-2024-07-07-at-10.00.53 AM-2.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/07/Screenshot-2024-07-07-at-10.03.32 AM-2.png" />


### Upcoming Features
- **Expanded Workflow Library:** Look forward to a growing collection of pre-designed workflows for various tasks and industries.
- **Prompt Packs:** Soon, you'll be able to purchase specialized prompt packs to enhance your AI agents' capabilities in specific domains.

### Contributing
Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgements
Special thanks to the contributors and the community for their support and feedback.

Enjoy using the Ollama Workbench! If you encounter any issues, feel free to open an issue or submit a pull request. Happy testing and model management!
