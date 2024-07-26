#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python; then
    echo "Python is not installed. Please install Python and try again."
    exit 1
fi

# Check if Git is installed
if ! command_exists git; then
    echo "Git is not installed. Please install Git and try again."
    exit 1
fi

# Set the repository URL and local directory
REPO_URL="https://github.com/marc-shade/Ollama-Workbench.git"
LOCAL_DIR="$HOME/Ollama-Workbench"

# Create a temporary script for installation commands
TEMP_SCRIPT=$(mktemp)

# Write installation commands to the temporary script
cat << EOF > "$TEMP_SCRIPT"
#!/bin/bash
set -e

# Clone or update the repository
if [ ! -d "$LOCAL_DIR" ]; then
    git clone "$REPO_URL" "$LOCAL_DIR"
else
    cd "$LOCAL_DIR"
    git pull
fi

# Install or update requirements
pip install -U -r "$LOCAL_DIR/requirements.txt"
python -m spacy download en_core_web_sm

# Install or update Ollama server (optional, assuming user has Ollama installed)
if [ -f "$LOCAL_DIR/install_ollama.sh" ]; then
    bash "$LOCAL_DIR/install_ollama.sh"
else
    echo "Ollama installation script not found. Skipping Ollama installation."
fi
EOF

# Make the temporary script executable
chmod +x "$TEMP_SCRIPT"

# Run the loading screen with the installation script
python "$LOCAL_DIR/loading_screen.py" bash "$TEMP_SCRIPT"

# Remove the temporary script
rm "$TEMP_SCRIPT"

# Run the Streamlit app
streamlit run "$LOCAL_DIR/main.py"