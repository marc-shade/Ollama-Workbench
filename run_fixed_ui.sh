#!/bin/bash

# Run fixed version of Ollama Workbench that properly handles form elements
echo "Starting Ollama Workbench with Fixed UI..."
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

# Make sure settings file exists
if [ ! -f "chat-settings.json" ]; then
    echo "Creating default settings file..."
    echo '{"selected_model": "llama2", "agent_type": "None", "metacognitive_type": "None", "voice_type": "None", "selected_corpus": "None", "temperature_slider_chat": 0.7, "max_tokens_slider_chat": 4000, "presence_penalty_slider_chat": 0.0, "frequency_penalty_slider_chat": 0.0, "episodic_memory_enabled": false, "advanced_thinking_enabled": false, "thinking_steps": ["1. Analyzing the problem", "2. Breaking down into subtasks", "3. Exploring potential solutions", "4. Evaluating approaches", "5. Formulating a comprehensive answer"], "instance_adaptive_cot_enabled": false, "cot_strategy": "IAP-ss", "cot_threshold": 0.5, "cot_top_n": 3}' > chat-settings.json
fi

# Run the fixed main script
# Use important Streamlit flags to help with the form elements:
# --server.maxMessageSize=200 - Allows larger messages for complex UIs
# --client.showErrorDetails=false - Hide detailed error messages
# --client.toolbarMode=minimal - Minimize UI distractions
echo "Starting Streamlit with fixed UI..."
streamlit run fixed_main.py --server.maxMessageSize=200 --client.showErrorDetails=false --client.toolbarMode=minimal

# Deactivate the virtual environment if it was activated
if [ -d "venv" ]; then
    deactivate
fi