#!/bin/bash

echo "==== Fixing dependencies for Ollama Workbench ===="
echo "CHECKPOINT: Starting dependency fix process"

# Create a log file
LOG_FILE="dependency_fix.log"
echo "$(date) - Starting dependency fix" > $LOG_FILE

# Function to log messages
log_message() {
    echo "$1"
    echo "$(date) - $1" >> $LOG_FILE
}

# Ensure we have the virtual environment activated if it exists
if [ -d "venv" ]; then
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    fi
    log_message "CHECKPOINT: Virtual environment activated"
fi

# Update pip
log_message "CHECKPOINT: Upgrading pip"
pip install --upgrade pip

# Fix NumPy version compatibility issue
log_message "CHECKPOINT: Fixing NumPy version compatibility issues"
pip uninstall -y numpy
log_message "CHECKPOINT: Installing numpy==1.23.5 for compatibility"
pip install numpy==1.23.5

# Fix PyTorch issues
log_message "CHECKPOINT: Fixing PyTorch issues"
pip uninstall -y torch torchvision torchaudio
log_message "CHECKPOINT: Installing PyTorch with compatible versions"
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Fix streamlit asyncio issue
log_message "CHECKPOINT: Installing compatible streamlit version"
pip install streamlit==1.27.2

# Install critical dependencies first
log_message "CHECKPOINT: Installing critical dependencies"
pip install typing-extensions>=4.5.0
pip install "ollama>=0.4.8,!=0.7.0"
pip install json-schema-for-humans==0.44.2

# Now install everything else from the integrated requirements
log_message "CHECKPOINT: Installing remaining dependencies from integrated_requirements.txt"
if [ -f "integrated_requirements.txt" ]; then
    pip install -r integrated_requirements.txt
else
    log_message "CHECKPOINT: integrated_requirements.txt not found, using regular requirements.txt"
    pip install -r requirements.txt
fi

# Fix possible version conflicts
log_message "CHECKPOINT: Fixing potential version conflicts"
pip install --force-reinstall "ollama>=0.4.8,!=0.7.0"
pip install --force-reinstall "numpy==1.23.5"
pip install --force-reinstall "torch==2.0.1"

# Double-check for json-schema-for-humans
python -c "import json_schema_for_humans" &>/dev/null || {
    log_message "CHECKPOINT: Installing json-schema-for-humans with specific version"
    pip install -U "json-schema-for-humans==0.44.2"
}

# Verify fixed dependencies
log_message "CHECKPOINT: Verifying numpy installation"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" 2>&1 | tee -a $LOG_FILE

log_message "CHECKPOINT: Verifying torch installation"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>&1 | tee -a $LOG_FILE

log_message "CHECKPOINT: Dependencies fixed! You can now run ./run_integrated_workbench.sh"
echo "If you encounter any other issues, please check the dependency_fix.log file for details"