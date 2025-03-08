#!/bin/bash

# Set directory paths
LOCAL_DIR="$(dirname "$0")"
VENV_DIR="$LOCAL_DIR/venv"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv --prompt venv
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install required packages from requirements.txt
echo "Installing required packages from requirements.txt..."
"$VENV_DIR/bin/pip" install -r "$LOCAL_DIR/requirements.txt"

# Run the application
echo "Running Ollama Workbench..."
cd "$LOCAL_DIR"
"$VENV_DIR/bin/python" -m streamlit run "$LOCAL_DIR/main.py"