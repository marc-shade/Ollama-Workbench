#!/bin/bash

# Automated setup and launch script for Ollama Workbench
# This script will handle all environment, dependency, and runtime issues for the user.
# It will log all steps and provide checkpoint-style, user-friendly output.

LOG_FILE="setup_workbench.log"
echo "==== Ollama Workbench Setup Started ====" | tee $LOG_FILE
echo "$(date) - Starting setup" >> $LOG_FILE

log_checkpoint() {
    echo "$1"
    echo "$(date) - $1" >> $LOG_FILE
}

check_ollama() {
    log_checkpoint "CHECKPOINT: Checking if Ollama is installed and running"
    if ! command -v ollama &> /dev/null; then
        log_checkpoint "CHECKPOINT: Ollama is not installed. Please install Ollama first."
        log_checkpoint "CHECKPOINT: Visit https://ollama.com/download for installation instructions."
        exit 1
    fi
    
    # Check if Ollama server is running
    if ! curl -s http://localhost:11434/api/tags > /dev/null; then
        log_checkpoint "CHECKPOINT: Ollama server is not running. Starting it now..."
        ollama serve > /dev/null 2>&1 &
        sleep 2
        log_checkpoint "CHECKPOINT: Ollama server started"
    else
        log_checkpoint "CHECKPOINT: Ollama server is already running"
    fi
}

# Check if Ollama is installed and running
check_ollama

# 1. Activate or create Python venv
if [ -d "venv" ]; then
    log_checkpoint "CHECKPOINT: Activating existing virtual environment."
    source venv/bin/activate
elif command -v conda &>/dev/null && conda info --envs | grep -q 'ollama-workbench'; then
    log_checkpoint "CHECKPOINT: Activating existing conda environment."
    conda activate ollama-workbench
else
    log_checkpoint "CHECKPOINT: Creating new Python virtual environment."
    python3 -m venv venv
    source venv/bin/activate
fi

# 2. Uninstall conflicting packages from all locations
log_checkpoint "CHECKPOINT: Uninstalling all NumPy, torch, torchvision, torchaudio."
pip uninstall -y numpy torch torchvision torchaudio || true
python3 -m pip uninstall -y numpy torch torchvision torchaudio || true

# 3. Remove .pyc and __pycache__
log_checkpoint "CHECKPOINT: Cleaning up .pyc and __pycache__ files."
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# 4. Upgrade pip
log_checkpoint "CHECKPOINT: Upgrading pip."
pip install --upgrade pip

# 5. Install compatible versions of NumPy, PyTorch, Streamlit
log_checkpoint "CHECKPOINT: Installing compatible versions of NumPy, PyTorch, Streamlit."
pip install numpy==1.23.5
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install --force-reinstall --no-cache-dir streamlit==1.27.2

# 6. Install remaining dependencies
log_checkpoint "CHECKPOINT: Installing all other dependencies."
if [ -f "integrated_requirements.txt" ]; then
    pip install -r integrated_requirements.txt
else
    pip install -r requirements.txt
fi

# 7. Install additional dependencies for robust Ollama utilities
log_checkpoint "CHECKPOINT: Installing additional dependencies for robust Ollama utilities."
pip install psutil requests importlib-metadata streamlit-option-menu

# 8. Verify installation
log_checkpoint "CHECKPOINT: Verifying installed versions."
echo -n "NumPy version: "
python -c "import numpy; print(numpy.__version__)"
echo -n "PyTorch version: "
python -c "import torch; print(torch.__version__)"
echo -n "Streamlit version: "
python -c "import streamlit; print(streamlit.__version__)"
echo -n "Ollama package: "
python -c "import importlib.util; print('Available' if importlib.util.find_spec('ollama') else 'Not available')"

# 9. Create ensure_ollama_running.sh if it doesn't exist
if [ ! -f "ensure_ollama_running.sh" ]; then
    log_checkpoint "CHECKPOINT: Creating ensure_ollama_running.sh script."
    cat > ensure_ollama_running.sh << 'EOF'
#!/bin/bash

# Check if Ollama server is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Ollama server is not running. Starting it now..."
    ollama serve > /dev/null 2>&1 &
    sleep 2
    echo "Ollama server started"
else
    echo "Ollama server is already running"
fi
EOF
    chmod +x ensure_ollama_running.sh
fi

# 10. Ensure Ollama server is running
log_checkpoint "CHECKPOINT: Ensuring Ollama server is running."
./ensure_ollama_running.sh 2>&1 | tee -a $LOG_FILE

# 11. Launch Ollama Workbench
log_checkpoint "CHECKPOINT: Launching Ollama Workbench."
streamlit run integrated_main.py 2>&1 | tee -a $LOG_FILE

# End
log_checkpoint "CHECKPOINT: Ollama Workbench setup and launch complete."
echo "==== Ollama Workbench is running! ===="
echo "If you encounter issues, see $LOG_FILE for details."
