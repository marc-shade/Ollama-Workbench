# Ollama Workbench

<img src="https://2acrestudios.com/wp-content/uploads/2024/06/00001-2881912941.png" style="width: 300px;" align="right" />

Welcome to the Ollama Workbench! This application is a comprehensive tool for managing and testing various models from the Ollama library. With features ranging from model comparison to vision testing, Ollama Workbench provides an intuitive interface for users to evaluate and maintain their machine learning models.

**Table of Contents**
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

### Model Management
- **List Local Models**: View a list of all locally available models, including their size and last modified date.
- **Show Model Information**: Display detailed information about a selected model.
- **Pull a Model**: Download a new model from the Ollama library.
- **Remove a Model**: Delete a selected model from the local storage.

### Testing
- **Model Feature Test**: Test a model's capability to handle JSON and function calls.
- **Model Comparison by Response Quality**: Compare the response quality of multiple models for a given prompt.
- **Contextual Response Test by Model**: Test how well a model maintains context across multiple prompts.
- **Vision Model Comparison**: Compare the performance of vision models using the same test image.

<img src="https://2acrestudios.com/wp-content/uploads/2024/06/Screenshot-2024-06-18-at-12.27.34 PM.png" />

## Installation

### Prerequisites
- [Python 3.7+](https://www.python.org/downloads/)
- [Streamlit](https://streamlit.io/)
- [requests](https://pypi.org/project/requests/)
- [pandas](https://pandas.pydata.org/)

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
    streamlit run main-workbench.py
    ```

## Usage

### Launching the Application
After installation, you can start the application using the following command:
```bash
streamlit run main-workbench.py
```

### Using the Sidebar
- Navigate through different functionalities using the sidebar.
- Select the desired test or maintenance function.

### Model Management
- **List Local Models**: Select to view available models.
- **Show Model Information**: Choose a model to view its detailed information.
- **Pull a Model**: Enter the name of the model you wish to download.
- **Remove a Model**: Select a model and confirm removal.

### Testing
- **Model Feature Test**: Test models for JSON handling and function calling capabilities.
- **Model Comparison by Response Quality**: Compare the responses of multiple models to a given prompt.
- **Contextual Response Test by Model**: Test the contextual understanding of a model through a series of prompts.
- **Vision Model Comparison**: Compare vision models by uploading an image.

## Contributing

Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

For detailed guidelines, refer to [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to the contributors and the community for their support and feedback.

---

Enjoy using the Ollama Workbench! If you encounter any issues, feel free to open an issue or submit a pull request. Happy testing and model management!

