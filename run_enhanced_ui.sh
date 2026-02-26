#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Ensure the full path is used
ENHANCED_UI_PATH="$SCRIPT_DIR/enhanced_chat_interface.py"
TTS_SERVER_PATH="$SCRIPT_DIR/tts_server/app.py"

# Check if TTS server is already running on port 8000
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "TTS server already running on port 8000, skipping server start"
    START_TTS=false
else
    START_TTS=true
    # Start the TTS server in the background
    echo "Starting TTS server..."
    cd tts_server
    python app.py --port 8000 &
    TTS_PID=$!
    cd ..
    
    # Wait for TTS server to start
    sleep 2
fi

# Check if enhanced_chat_interface.py exists
if [ ! -f "$ENHANCED_UI_PATH" ]; then
    echo "Error: $ENHANCED_UI_PATH not found!"
    if [ "$START_TTS" = true ] && [ -n "$TTS_PID" ]; then
        echo "Cleaning up TTS server..."
        kill $TTS_PID 2>/dev/null
    fi
    exit 1
fi

# Check for and activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ] && [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$SCRIPT_DIR/venv/bin/activate"
    
    # Make sure pygame is installed in the virtual environment
    echo "Installing required packages in virtual environment..."
    pip install pygame streamlit-extras gtts groq > /dev/null
fi

# Run the enhanced chat interface
echo "Starting enhanced chat interface..."
streamlit run "$ENHANCED_UI_PATH"

# Cleanup - kill the TTS server only if we started it
if [ "$START_TTS" = true ] && [ -n "$TTS_PID" ]; then
    echo "Cleaning up TTS server..."
    kill $TTS_PID 2>/dev/null
fi