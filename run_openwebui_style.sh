#!/bin/bash

# Run Ollama Workbench with Open WebUI-style interface
echo "Starting Ollama Workbench with Open WebUI-style interface..."
echo "Press Ctrl+C to stop the application"

# Activate the virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create necessary directories if they don't exist
mkdir -p sessions
mkdir -p workspaces
mkdir -p ragtest
mkdir -p uploads
mkdir -p tmp

# Make sure enhanced_corpus.py is accessible
if [ ! -f "enhanced_corpus.py" ]; then
    echo "Error: enhanced_corpus.py not found!"
    echo "Using a simple RAG implementation instead."
fi

# Run the application with Streamlit
# Use --server.maxMessageSize to allow larger message passing
# Use --server.enableXsrfProtection=false to avoid CSRF issues
streamlit run main.py --server.maxMessageSize=200

# Deactivate the virtual environment if it was activated
if [ -d "venv" ]; then
    deactivate
fi