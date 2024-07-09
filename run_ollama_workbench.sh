#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python; then
    echo "Python is not installed. Please install Python and try again."
    exit 1
fi

# Check if Git is installed
if ! command_exists git; then
    echo "Git is not installed. Please install Git and try again."
    exit 1
fi

# Set the repository URL and local directory
REPO_URL="https://github.com/marc-shade/Ollama-Workbench.git"
LOCAL_DIR="$HOME/ollama_workbench"

# Clone or update the repository
if [ ! -d "$LOCAL_DIR" ]; then
    git clone "$REPO_URL" "$LOCAL_DIR"
else
    cd "$LOCAL_DIR"
    git pull
fi

# Create and activate virtual environment
if [ ! -d "$LOCAL_DIR/venv" ]; then
    python -m venv "$LOCAL_DIR/venv"
fi
source "$LOCAL_DIR/venv/bin/activate"

# Install or update requirements
pip install -r "$LOCAL_DIR/requirements.txt"

# Install or update Ollama server (optional, assuming user has Ollama installed)
if [ -f "$LOCAL_DIR/install_ollama.sh" ]; then
    bash "$LOCAL_DIR/install_ollama.sh"
else
    echo "Ollama installation script not found. Skipping Ollama installation."
fi

# Run the Streamlit app
streamlit run "$LOCAL_DIR/main.py"