#!/bin/bash
# Script to start the TTS server

# Navigate to the script directory
cd "$(dirname "$0")"

# Create logs directory if it doesn't exist
mkdir -p ../logs

# Check if the server is already running
if pgrep -f "python app.py" > /dev/null; then
    echo "TTS server is already running."
    exit 0
fi

# Start the server
echo "Starting TTS server..."
python app.py > ../logs/tts_server.out 2>&1 &

# Wait a moment for the server to start
sleep 2

# Check if the server started successfully
if pgrep -f "python app.py" > /dev/null; then
    echo "TTS server started successfully on port 8000."
else
    echo "Failed to start TTS server. Check the logs for details."
    exit 1
fi
