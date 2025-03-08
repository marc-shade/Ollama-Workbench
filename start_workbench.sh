#!/bin/bash

# Function to check if a port is in use
port_in_use() {
    lsof -i :$1 >/dev/null 2>&1
}

# Function to wait for a service to be ready
wait_for_service() {
    local port=$1
    local service=$2
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for $service to be ready..."
    while ! curl -s http://localhost:$port >/dev/null; do
        if [ $attempt -ge $max_attempts ]; then
            echo "$service failed to start"
            return 1
        fi
        sleep 1
        ((attempt++))
    done
    echo "$service is ready!"
}

# Kill any existing processes
if port_in_use 5000; then
    echo "Cleaning up existing TTS server..."
    kill $(lsof -t -i:5000) 2>/dev/null
fi

# Start the TTS server in the background
echo "Starting TTS server..."
./tts_server/start_tts_server.sh &
TTS_PID=$!

# Wait for TTS server to be ready
sleep 2

# Start Ollama if it's not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 2  # Give Ollama time to start
fi

# Start Ollama Workbench
echo "Starting Ollama Workbench..."
streamlit run main.py

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    kill $TTS_PID 2>/dev/null
    if [ ! -z "$OLLAMA_PID" ]; then
        kill $OLLAMA_PID 2>/dev/null
    fi
    exit 0
}

# Set up cleanup on script exit
trap cleanup EXIT INT TERM
