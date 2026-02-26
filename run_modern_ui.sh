#!/bin/bash

# Run the modern UI test for Ollama Workbench
echo "Starting Ollama Workbench with Modern UI..."
echo "Press Ctrl+C to stop the application"

# Activate the virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create necessary directories if they don't exist
mkdir -p sessions
mkdir -p workspaces
mkdir -p ragtest

# Run the modern UI test
streamlit run run_modernui.py --server.port=8501

# Deactivate the virtual environment if it was activated
if [ -d "venv" ]; then
    deactivate
fi