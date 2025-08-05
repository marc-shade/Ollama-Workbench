#!/bin/bash

# Run Integrated Ollama Workbench
# This script runs the integrated version of Ollama Workbench with the modern chat interface

echo "==== Starting Integrated Ollama Workbench ===="
echo "CHECKPOINT: Initializing environment"

# Create a log file for detailed debugging
LOG_FILE="ollama_workbench_install.log"
echo "$(date) - Starting Ollama Workbench installation" > $LOG_FILE

# Function to log messages both to console and log file
log_message() {
    echo "$1"
    echo "$(date) - $1" >> $LOG_FILE
}

# Function to check and install Python packages with more robust error handling
check_and_install_package() {
    package_name=$1
    pip_package=${2:-$package_name}
    import_name=${3:-$package_name}
    
    echo "CHECKPOINT: Checking for $package_name..."
    if ! python -c "import $import_name" 2>/dev/null; then
        echo "CHECKPOINT: Installing $package_name..."
        pip install $pip_package
        # Verify installation was successful
        if ! python -c "import $import_name" 2>/dev/null; then
            echo "CHECKPOINT: WARNING - Failed to import $import_name after installation"
            echo "CHECKPOINT: Trying alternative installation method..."
            pip install --upgrade $pip_package
            
            # Check again after upgrade attempt
            if ! python -c "import $import_name" 2>/dev/null; then
                echo "CHECKPOINT: ERROR - Failed to install $package_name properly"
                echo "CHECKPOINT: Please try installing manually with: pip install $pip_package"
            else
                echo "CHECKPOINT: $package_name installed successfully with upgrade"
            fi
        else
            echo "CHECKPOINT: $package_name installed successfully"
        fi
    else
        echo "CHECKPOINT: $package_name is already installed"
    fi
}

# Ensure Ollama server is running using the dedicated script
log_message "CHECKPOINT: Ensuring Ollama server is running"
./ensure_ollama_running.sh 2>&1 | tee -a $LOG_FILE

# Check the exit code of the ensure_ollama_running.sh script
if [ $? -ne 0 ]; then
    log_message "CHECKPOINT: WARNING - Ollama server check failed"
    log_message "CHECKPOINT: Some features may not work correctly without Ollama"
else
    log_message "CHECKPOINT: Ollama server check completed successfully"
fi

# Kill any existing Streamlit processes
echo "CHECKPOINT: Stopping any existing Streamlit processes..."
pkill -f "streamlit run"
sleep 2
echo "CHECKPOINT: Streamlit processes stopped"

# Kill any existing TTS server processes
echo "CHECKPOINT: Stopping any existing TTS server processes..."
pkill -f "python -m TTS.server.server"
sleep 2
echo "CHECKPOINT: TTS server processes stopped"

# Check for required dependencies
echo "CHECKPOINT: Verifying all required dependencies"

# Core dependencies
check_and_install_package streamlit
check_and_install_package tiktoken
check_and_install_package numpy
check_and_install_package flask
check_and_install_package "streamlit_option_menu" "streamlit-option-menu"

# Web automation dependencies
echo "CHECKPOINT: Checking for selenium (required for UI automation)..."
check_and_install_package selenium
check_and_install_package "webdriver_manager" "webdriver-manager"
check_and_install_package playwright

# LangChain dependencies
echo "CHECKPOINT: Checking for langchain and langchain_community (required for RAG)..."
check_and_install_package "langchain" "langchain==0.2.15"
check_and_install_package "langchain_community" "langchain-community==0.2.15"

# Special dependencies with different import and package names
echo "CHECKPOINT: Installing autogen (critical dependency)..."
check_and_install_package "autogen" "pyautogen" "autogen"

# Try alternative installation method for autogen if needed
if ! python -c "import autogen" 2>/dev/null; then
    echo "CHECKPOINT: Trying direct installation of pyautogen..."
    pip uninstall -y pyautogen
    pip install --no-cache-dir pyautogen
    
    # Check if installation was successful
    if python -c "import autogen" 2>/dev/null; then
        echo "CHECKPOINT: Successfully installed autogen with alternative method"
    else
        echo "CHECKPOINT: WARNING - autogen installation still failed"
        echo "CHECKPOINT: Some features like Brainstorm may not work properly"
    fi
fi

# TTS dependencies
check_and_install_package TTS
check_and_install_package "soundfile" "soundfile"
check_and_install_package "sounddevice" "sounddevice"

# Start the TTS server in the background
echo "CHECKPOINT: Starting TTS server..."
python -m TTS.server.server --model_name tts_models/en/ljspeech/tacotron2-DDC &
sleep 5  # Give TTS server time to start
echo "CHECKPOINT: TTS server started"

# Install all required dependencies from the integrated requirements file
log_message "CHECKPOINT: Installing all dependencies from integrated_requirements.txt"
pip install -r integrated_requirements.txt 2>&1 | tee -a $LOG_FILE
log_message "CHECKPOINT: All dependencies installed from requirements file"

# Verify core dependencies are installed
log_message "CHECKPOINT: Verifying core dependencies for Ollama Workbench"

# Core dependencies array
CORE_DEPS=("streamlit" "numpy" "requests" "psutil" "ollama" "flask")

# Check each core dependency
for dep in "${CORE_DEPS[@]}"; do
    if ! python -c "import $dep" 2>/dev/null; then
        log_message "CHECKPOINT: CRITICAL - $dep is missing, attempting emergency install"
        pip install $dep --no-cache-dir 2>&1 | tee -a $LOG_FILE
        
        # Verify installation
        if ! python -c "import $dep" 2>/dev/null; then
            log_message "CHECKPOINT: CRITICAL ERROR - Failed to install $dep"
        else
            log_message "CHECKPOINT: Successfully installed $dep"
        fi
    else
        log_message "CHECKPOINT: $dep is available"
    fi
done

# Check for config.py which is critical for Ollama Workbench
if [ ! -f "config.py" ]; then
    log_message "CHECKPOINT: CRITICAL - config.py is missing"
    
    # Create a basic config.py if it doesn't exist
    log_message "CHECKPOINT: Creating basic config.py file"
    cat > config.py << 'EOF'
# Basic configuration for Ollama Workbench

CONFIG = {
    "OLLAMA_HOST": "http://localhost:11434",
    "DEFAULT_MODEL": "llama3",
    "DEBUG": True
}

def get_config():
    return CONFIG

def update_config(new_config):
    global CONFIG
    CONFIG.update(new_config)
    return CONFIG

def set_api_key(service, key):
    CONFIG[f"{service.upper()}_API_KEY"] = key

def get_api_key(service):
    return CONFIG.get(f"{service.upper()}_API_KEY", "")
EOF
    log_message "CHECKPOINT: Created basic config.py file"
fi

# Double-check critical dependencies
echo "CHECKPOINT: Verifying critical dependencies after installation"

# Verify autogen installation
echo "CHECKPOINT: Checking autogen installation..."
if ! python -c "import autogen" 2>/dev/null; then
    echo "CHECKPOINT: CRITICAL ERROR - autogen still not available after installation"
    echo "CHECKPOINT: Attempting emergency fix for autogen..."
    
    # Try multiple installation methods
    pip uninstall -y pyautogen autogen
    pip install --no-cache-dir pyautogen
    pip install --no-cache-dir "pyautogen==0.9.0"
    
    # Check if any method worked
    if python -c "import autogen" 2>/dev/null; then
        echo "CHECKPOINT: Emergency fix for autogen successful!"
    else
        echo "CHECKPOINT: WARNING - Could not install autogen. Some features will be unavailable."
        echo "CHECKPOINT: Please try manually with: pip install pyautogen==0.9.0"
    fi
else
    echo "CHECKPOINT: autogen is available"
fi

# Verify pdfkit installation
echo "CHECKPOINT: Checking pdfkit installation..."
if ! python -c "import pdfkit" 2>/dev/null; then
    echo "CHECKPOINT: pdfkit not found, installing..."
    pip install pdfkit
    
    # Check if installation was successful
    if ! python -c "import pdfkit" 2>/dev/null; then
        echo "CHECKPOINT: WARNING - pdfkit installation may have failed"
        
        # Check if wkhtmltopdf is installed (required by pdfkit)
        echo "CHECKPOINT: Checking for wkhtmltopdf (required by pdfkit)..."
        if ! command -v wkhtmltopdf &>/dev/null; then
            echo "CHECKPOINT: wkhtmltopdf not found - this is required by pdfkit"
            echo "CHECKPOINT: On macOS, install with: brew install wkhtmltopdf"
            echo "CHECKPOINT: On Ubuntu/Debian, install with: sudo apt-get install wkhtmltopdf"
            echo "CHECKPOINT: On CentOS/RHEL, install with: sudo yum install wkhtmltopdf"
        else
            echo "CHECKPOINT: wkhtmltopdf is installed"
        fi
    else
        echo "CHECKPOINT: pdfkit installed successfully"
    fi
else
    echo "CHECKPOINT: pdfkit is available"
fi

# Verify json-schema-for-humans installation
echo "CHECKPOINT: Checking json-schema-for-humans installation..."
if ! python -c "import json_schema_for_humans" 2>/dev/null; then
    echo "CHECKPOINT: json-schema-for-humans not found, installing..."
    pip install json-schema-for-humans
    
    # Check if installation was successful
    if python -c "import json_schema_for_humans" 2>/dev/null; then
        echo "CHECKPOINT: json-schema-for-humans installed successfully"
    else
        echo "CHECKPOINT: WARNING - json-schema-for-humans installation failed"
    fi
else
    echo "CHECKPOINT: json-schema-for-humans is available"
fi

# Fix deprecated Streamlit functions
echo "CHECKPOINT: Fixing deprecated Streamlit functions..."

# Create a backup of the files before modifying them
for file in $(grep -l "st\.experimental_get_query_params" *.py); do
    cp "$file" "${file}.bak"
    echo "CHECKPOINT: Created backup of $file as ${file}.bak"
    
    # Replace deprecated function with the correct syntax
    sed -i '' 's/st\.experimental_get_query_params/dict(st\.query_params)/g' "$file"
    echo "CHECKPOINT: Updated $file to use dict(st.query_params) instead of st.experimental_get_query_params"
 done

# Run the integrated application
echo "CHECKPOINT: Starting Integrated Ollama Workbench with modern UI..."
streamlit run integrated_main.py

echo "CHECKPOINT: Integrated Ollama Workbench is now running."
