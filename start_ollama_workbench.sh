#!/bin/bash

# Set directory paths
LOCAL_DIR="$(dirname "$0")"
VENV_DIR="$LOCAL_DIR/venv"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Please run setup.sh or run_ollama_workbench.sh first."
    exit 1
fi

# Check if streamlit is installed
if ! $VENV_DIR/bin/pip show streamlit >/dev/null 2>&1; then
    echo "Streamlit not found. Please run setup.sh or run_ollama_workbench.sh first."
    exit 1
fi

# Check if streamlit_option_menu is installed
if ! $VENV_DIR/bin/pip show streamlit-option-menu >/dev/null 2>&1; then
    echo "Installing streamlit-option-menu..."
    $VENV_DIR/bin/pip install streamlit-option-menu==0.3.13
fi

# Run the application
echo "Starting Ollama Workbench..."
cd "$LOCAL_DIR"
$VENV_DIR/bin/streamlit run "$LOCAL_DIR/main.py"