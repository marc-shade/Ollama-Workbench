# Ollama Workbench

<img src="https://2acrestudios.com/wp-content/uploads/2024/06/00001-2881912941.png" style="width: 300px;" align="right" />

This application is a comprehensive tool for managing and testing various models from the Ollama library. With features ranging from model comparison to vision testing, Ollama Workbench provides an intuitive interface for users to evaluate and maintain their machine learning models. It also features a chat interface for real-time interaction with selected models, including a workspace for saving and managing code and text generated during chat sessions.

**Table of Contents**
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

### 🧑 Chat
Engage in a real-time chat with a selected model, enhanced with various features:
- 🧑‍🔧 **Agent Types:** Choose from a variety of predefined agent types, each with specific prompts to guide the model's behavior (e.g., Coder, Analyst, Creative Writer). You can also create and manage your own custom agent type prompts.
- 🧠 **Metacognitive Types:** Enhance the model's reasoning abilities by selecting a metacognitive type (e.g., Visualization of Thought, Chain of Thought). You can also create and manage your own custom metacognitive type prompts.
- 🗣️ **Voice Types:** Set the tone and style of the model's responses (e.g., Sarcastic, Formal, Storyteller). You can also create and manage your own custom voice type prompts.
- 📚 **Corpus Integration:** Load a corpus of text from the 'Files' section to provide contextual information to the model, improving its responses.
- 🛠️ **Advanced Settings:** Fine-tune the model's output by adjusting parameters like temperature, max tokens, presence penalty, and frequency penalty.
- 📜 **Workspace:** Save and manage code snippets and other text generated during your chat sessions.
- 💾 **Save/Load Sessions:** Save your chat history and workspace for later use, or load previously saved sessions.

### ⚙️ Workflow
- 🚀 **Manage Projects:** Create, manage, and run projects with multiple AI agents. Each agent can be configured with a specific model, agent type, metacognitive type, voice type, corpus, temperature, and max tokens. You can add tasks to projects, assign them to agents, and run the agents to generate outputs for each task.
- ✨ **Manage Agent Prompts:** Create, edit, and delete custom prompts for Agent Type, Metacognitive Type, and Voice Type. You can also download and upload prompt JSON files for easy sharing and backup.

### 🗄️ Document
- 🗂️ **Manage Corpus:** Create, edit, and delete corpus from files.
- 📂 **Manage Files**: Upload, view, edit, and delete files. These files can be used as corpus in the Chat.
- 🕸️ **Web to Corpus File**: Convert web content into a corpus for analysis or training. The generated files will be saved to the 'files' folder, accessible from both the 'Document' section and the 'Chat' section.
- ✔️ **Repository Analyzer**: Analyze your Python repository, generate documentation, debug reports, or a README.md file. The output will stream in real-time in the UI.

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

### Prerequisites
- [Python 3.7+](https://www.python.org/downloads/)
- [Streamlit](https://streamlit.io/)
- [requests](https://pypi.org/project/requests/)
- [pandas](https://pandas.pydata.org/)
- [ollama](https://pypi.org/project/ollama/)

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

3. **Run the application**
    ```bash
    streamlit run main.py
    ```

## Usage

### Launching the Application
After installation, you can start the application using the following command:
```bash
streamlit run main.py
```
### Using the Sidebar
- Navigate through different functionalities using the sidebar.
- Select the desired test, chat, document, workflow, or maintenance function.

### Chat
- **Chat**: Select a model and start chatting! The chat interface will automatically detect and color code blocks within the assistant's responses.
- **Workspace**: Use the Workspace tab to save and manage code and text generated during your chat sessions. You can add new items manually or save assistant responses directly to the workspace.
- **Files**: Upload, view, edit, and delete files. These files can be used as corpus in the Chat.

### Workflow
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

<img src="https://2acrestudios.com/wp-content/uploads/2024/06/Screenshot-2024-06-26-at-1.12.39 PM-2.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/06/Screenshot-2024-06-26-at-1.12.54 PM-2.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/06/Screenshot-2024-06-26-at-1.12.59 PM-2.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/06/Screenshot-2024-06-26-at-1.13.14 PM-2.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/06/Screenshot-2024-06-26-at-1.13.22 PM-2.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/06/Screenshot-2024-06-26-at-1.13.29 PM-2.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/06/Screenshot-2024-06-26-at-1.13.34 PM-2.png" />
<img src="https://2acrestudios.com/wp-content/uploads/2024/06/Screenshot-2024-06-26-at-1.15.41 PM-2.png" />

### Contributing
Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

For detailed guidelines, refer to [CONTRIBUTING.md](CONTRIBUTING.md).

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgements
Special thanks to the contributors and the community for their support and feedback.

Enjoy using the Ollama Workbench! If you encounter any issues, feel free to open an issue or submit a pull request. Happy testing and model management!
