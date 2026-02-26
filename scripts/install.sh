#!/bin/bash

# Text colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Ollama Workbench Installer ===${NC}"
echo "This script will install Ollama Workbench and all its dependencies."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo -e "${RED}Homebrew is not installed. Installing Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Check if Python 3.11 is installed
if ! command -v python3.11 &> /dev/null; then
    echo -e "${BLUE}Installing Python 3.11...${NC}"
    brew install python@3.11
fi

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${BLUE}Installing Miniconda...${NC}"
    brew install --cask miniconda
    conda init "$(basename "${SHELL}")"
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${BLUE}Installing Ollama...${NC}"
    curl https://ollama.ai/install.sh | sh
fi

 # Install Zonos TTS
 echo -e "${BLUE}Installing Zonos TTS...${NC}"
  if [ -d "/Volumes/FILES/code/Zonos" ]; then
         rm -rf /Volumes/FILES/code/Zonos
     fi
     if [ ! -d "/Volumes/FILES/code/Zonos" ]; then
      git clone https://github.com/Zyphra/Zonos.git /Volumes/FILES/code/Zonos
      if [ -f "/Volumes/FILES/code/Zonos/requirements.txt" ]; then
          if command -v uv &> /dev/null; then
                 echo -e "${BLUE}Installing Zonos Python dependencies with uv...${NC}"
                 uv pip install -r /Volumes/FILES/code/Zonos/requirements.txt
             else
                 echo -e "${BLUE}Installing Zonos Python dependencies with pip...${NC}"
                 pip install -r /Volumes/FILES/code/Zonos/requirements.txt
             fi
          echo -e "${GREEN}Zonos TTS dependencies installed successfully.${NC}"
      else
          echo -e "${RED}Zonos TTS requirements.txt not found. Zonos TTS might not be installed correctly.${NC}"
      fi
     fi

     # Add instructions to run Zonos server if applicable
     echo -e "${BLUE}To run Zonos TTS (if it has a server component), follow the instructions in the Zonos repository.${NC}"
     echo -e "${BLUE}To run Zonos TTS in Docker, navigate to the /Volumes/FILES/code/Zonos directory and run 'docker compose up' or follow the instructions in the Zonos repository.${NC}"


# Create and activate Conda environment (Skipped)
# echo -e "${BLUE}Creating Conda environment...${NC}"
# conda env create -f environment.yml
# conda activate ollamaworkbench

# Install additional system dependencies
echo -e "${BLUE}Installing system dependencies...${NC}"
brew install portaudio

# Install Python packages
echo -e "${BLUE}Installing Python packages...${NC}"
if command -v uv &> /dev/null; then
        echo -e "${BLUE}Installing Python packages with uv...${NC}"
        uv pip install -r requirements.txt
    else
        echo -e "${BLUE}Installing Python packages with pip...${NC}"
        pip install -r requirements.txt
    fi

# Install and start the TTS server
echo -e "${BLUE}Setting up TTS server...${NC}"
pip install flask flask-cors gtts
mkdir -p tts_server
cat > tts_server/app.py << 'EOL'
from flask import Flask, request, jsonify
from gtts import gTTS
import base64
import io
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/v1/audio/speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.json
        text = data.get('input', '')
        voice = data.get('voice', 'en-US-Wavenet-A')
        
        # Extract language from voice (e.g., 'en-US-Wavenet-A' -> 'en')
        lang = voice.split('-')[0] if '-' in voice else 'en'
        
        # Create a bytes buffer for the audio
        audio_buffer = io.BytesIO()
        
        # Generate speech
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(audio_buffer)
        
        # Get the audio data and encode it
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        return jsonify({'audio': audio_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
EOL

# Create a launch script for the TTS server
cat > tts_server/start_tts_server.sh << 'EOL'
#!/bin/bash
cd "$(dirname "$0")"
python app.py
EOL

chmod +x tts_server/start_tts_server.sh

# Add TTS requirements to requirements.txt if they don't exist
if ! grep -q "flask-cors" requirements.txt; then
    echo "flask-cors" >> requirements.txt
fi
if ! grep -q "gtts" requirements.txt; then
    echo "gtts" >> requirements.txt
fi

# Create a launch script for Ollama Workbench
cat > start_workbench.sh << 'EOL'
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
EOL

chmod +x start_workbench.sh

echo -e "${GREEN}Installation complete!${NC}"
echo -e "${BLUE}To start Ollama Workbench, run:${NC}"
echo "./start_workbench.sh"
