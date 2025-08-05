#!/bin/bash
echo "Starting Ollama Workbench..."
cd "/Volumes/FILES/code/Ollama-Workbench"
source venv/bin/activate

# Start Ollama server if not running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 3
fi

# Start Streamlit
echo "Starting Streamlit interface..."
streamlit run main.py
