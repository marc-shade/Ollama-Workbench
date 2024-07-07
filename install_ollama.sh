#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if curl is installed
if ! command_exists curl; then
    echo "Error: curl is not installed. Please install curl and try again."
    exit 1
fi

# Install Ollama
echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Check if Ollama was installed successfully
if ! command_exists ollama; then
    echo "Error: Ollama installation failed. Please check the output above for any error messages."
    exit 1
fi

echo "Ollama installed successfully."

# Start Ollama service
echo "Starting Ollama service..."
ollama serve &

# Wait for Ollama service to start
echo "Waiting for Ollama service to start..."
sleep 10

# Pull Mistral models
echo "Pulling mistral:instruct model..."
ollama pull mistral:instruct

echo "Pulling mistral:7b-instruct-v0.2-q8_0 model..."
ollama pull mistral:7b-instruct-v0.2-q8_0

echo "Installation and model pulling completed successfully!"