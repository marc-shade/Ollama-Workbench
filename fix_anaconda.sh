#!/bin/bash

# Text formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Fixing streamlit_option_menu in Anaconda ===${NC}"

# Try to find the Anaconda pip executable
ANACONDA_PIP="/opt/anaconda3/bin/pip"

if [ ! -f "$ANACONDA_PIP" ]; then
    echo -e "${YELLOW}Anaconda pip not found at $ANACONDA_PIP${NC}"
    echo -e "${YELLOW}Trying to find pip in Anaconda directory...${NC}"
    
    # Try to find pip in common Anaconda locations
    for pip_path in "/opt/anaconda3/bin/pip" "/Users/marc/anaconda3/bin/pip" "/Users/marc/opt/anaconda3/bin/pip" "/anaconda3/bin/pip" "/opt/anaconda/bin/pip"; do
        if [ -f "$pip_path" ]; then
            ANACONDA_PIP="$pip_path"
            echo -e "${GREEN}Found Anaconda pip at $ANACONDA_PIP${NC}"
            break
        fi
    done
    
    if [ ! -f "$ANACONDA_PIP" ]; then
        echo -e "${RED}Could not find Anaconda pip. Please enter the path to your Anaconda pip:${NC}"
        read -p "Anaconda pip path: " ANACONDA_PIP
        
        if [ ! -f "$ANACONDA_PIP" ]; then
            echo -e "${RED}Invalid path. Exiting.${NC}"
            exit 1
        fi
    fi
fi

# List of essential packages to install
PACKAGES=(
    "streamlit-option-menu==0.3.13"
    "openai==1.43.0"
    "streamlit==1.38.0"
    "streamlit-extras==0.4.7"
    "streamlit-flow==0.1.0"
    "streamlit-javascript==0.1.5"
    "langchain==0.2.15"
    "langchain-community==0.2.15"
    "ollama==0.4.2"
    "groq==0.10.0"
    "mistralai"
    "tiktoken==0.7.0"
    "pydantic==2.10.2"
    "requests==2.32.3"
    "flask==3.0.3"
    "flask-cors==5.0.0"
    "sentence-transformers==2.5.0"
    "torch==2.2.0"
    "numpy>=1.24.3,<2.0.0"
    "gtts==2.5.4"
    "pygame==2.6.1"
    "autogen==0.2.35"
    "pyautogen==0.2.35"
    "fake-useragent==1.5.1"
)

# Function to install a package
install_package() {
    local package=$1
    local pip_cmd=$2
    
    echo -e "${YELLOW}Installing $package using $pip_cmd...${NC}"
    "$pip_cmd" install $package
    
    return $?
}

# Install each package
for package in "${PACKAGES[@]}"; do
    install_package "$package" "$ANACONDA_PIP"
    
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Trying alternative method for $package...${NC}"
        
        # Try using the python executable directly
        ANACONDA_PYTHON="${ANACONDA_PIP%/*}/python"
        if [ -f "$ANACONDA_PYTHON" ]; then
            echo -e "${YELLOW}Using $ANACONDA_PYTHON to install $package...${NC}"
            "$ANACONDA_PYTHON" -m pip install $package
            
            if [ $? -ne 0 ]; then
                echo -e "${RED}Failed to install $package using $ANACONDA_PYTHON${NC}"
                # Continue with next package instead of exiting
                continue
            fi
        else
            echo -e "${RED}Could not find Anaconda Python. Skipping $package.${NC}"
            continue
        fi
    fi
done

echo -e "${GREEN}Installation complete. Try running the application now.${NC}"
echo -e "${YELLOW}If you still encounter issues, try running:${NC}"
echo -e "conda install -c conda-forge streamlit-option-menu"
echo -e "or"
echo -e "pip install streamlit-option-menu==0.3.13 --user"