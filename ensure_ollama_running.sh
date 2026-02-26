#!/bin/bash

# Script to ensure Ollama server is running
# This script checks if Ollama is running and starts it if needed

echo "==== Checking Ollama Server Status ===="
echo "CHECKPOINT: Verifying Ollama server is running"

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "CHECKPOINT: Ollama is not running. Starting Ollama..."
    
    # Try to start Ollama
    ollama serve &
    
    # Wait for Ollama to start
    echo "CHECKPOINT: Waiting for Ollama server to initialize..."
    sleep 5
    
    # Verify Ollama started successfully
    if pgrep -x "ollama" > /dev/null; then
        echo "CHECKPOINT: Ollama server started successfully"
    else
        echo "CHECKPOINT: ERROR - Failed to start Ollama server"
        echo "CHECKPOINT: Please start Ollama manually with 'ollama serve'"
        exit 1
    fi
else
    echo "CHECKPOINT: Ollama server is already running"
fi

# Test Ollama API connection
echo "CHECKPOINT: Testing Ollama API connection..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "CHECKPOINT: Ollama API is responding"
else
    echo "CHECKPOINT: WARNING - Ollama API is not responding"
    echo "CHECKPOINT: Please check if Ollama is properly installed and configured"
fi

echo "CHECKPOINT: Ollama server check complete"
