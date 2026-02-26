#!/bin/bash

# Check if there's a TTS server running on port 8000
TTS_PID=$(lsof -Pi :8000 -sTCP:LISTEN -t 2>/dev/null)

if [ -n "$TTS_PID" ]; then
    echo "Found TTS server running on port 8000 (PID: $TTS_PID)"
    echo "Shutting down TTS server..."
    kill $TTS_PID
    sleep 1
    
    # Verify it's gone
    if ! lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null; then
        echo "TTS server successfully shut down."
    else
        echo "TTS server is still running, trying force kill..."
        kill -9 $TTS_PID
        sleep 1
        if ! lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null; then
            echo "TTS server successfully force killed."
        else
            echo "ERROR: Failed to kill TTS server. Please check process manually."
        fi
    fi
else
    echo "No TTS server found running on port 8000."
fi