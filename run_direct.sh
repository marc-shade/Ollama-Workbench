#!/bin/bash

# Set directory paths
LOCAL_DIR="$(dirname "$0")"

# Text formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Ollama Workbench Direct Runner ===${NC}"

# First, install streamlit_option_menu in the current Python environment
echo -e "${YELLOW}Installing streamlit_option_menu in the current Python environment...${NC}"
python3 "$LOCAL_DIR/install_streamlit_option_menu.py"

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install streamlit_option_menu. Trying alternative method...${NC}"
    pip install streamlit-option-menu==0.3.13
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install streamlit_option_menu. Please install it manually:${NC}"
        echo "pip install streamlit-option-menu==0.3.13"
        exit 1
    fi
fi

# Check if streamlit is installed
if ! python3 -c "import streamlit" &>/dev/null; then
    echo -e "${YELLOW}Installing streamlit...${NC}"
    pip install streamlit
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install streamlit. Please install it manually:${NC}"
        echo "pip install streamlit"
        exit 1
    fi
fi

# Run the application
echo -e "${GREEN}Starting Ollama Workbench...${NC}"
cd "$LOCAL_DIR"
streamlit run "$LOCAL_DIR/main.py"