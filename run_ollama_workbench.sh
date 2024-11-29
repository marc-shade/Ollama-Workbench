#!/bin/bash

# Text formatting
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if Ollama is running
is_ollama_running() {
    lsof -i :11434 >/dev/null 2>&1
}

# Function to check if a Python package is installed
is_package_installed() {
    $VENV_DIR/bin/pip show "$1" >/dev/null 2>&1
}

# Function to display the header
show_header() {
    clear
    echo -e "${BOLD}╔════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║       Ollama Workbench Manager         ║${NC}"
    echo -e "${BOLD}╚════════════════════════════════════════╝${NC}"
    echo
}

# Function to setup system dependencies
setup_system_deps() {
    echo -e "${BLUE}Checking system dependencies...${NC}"
    
    # Check if Homebrew is installed
    if ! command_exists brew; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install/Update required system packages
    echo "Installing/Updating system packages..."
    brew install gcc gfortran meson pkg-config cmake openblas
    brew link --force openblas
}

# Function to check and setup Python environment
setup_python_env() {
    echo -e "${BLUE}Checking Python environment...${NC}"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment..."
        python -m venv "$VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip and essential tools
    echo "Upgrading pip and essential tools..."
    $VENV_DIR/bin/pip install --upgrade pip wheel setuptools build
}

# Function to install/update dependencies
install_dependencies() {
    echo -e "${BLUE}Installing Python dependencies...${NC}"
    
    # Install numpy with openblas
    echo "Installing numpy with openblas..."
    OPENBLAS="$(brew --prefix openblas)"
    CFLAGS="-falign-functions=8 ${CFLAGS}" ATLAS=None BLAS="${OPENBLAS}/lib/libblas.dylib" LAPACK="${OPENBLAS}/lib/liblapack.dylib" $VENV_DIR/bin/pip install --no-cache-dir numpy
    
    # Install scipy with openblas
    echo "Installing scipy..."
    CFLAGS="-falign-functions=8 ${CFLAGS}" ATLAS=None BLAS="${OPENBLAS}/lib/libblas.dylib" LAPACK="${OPENBLAS}/lib/liblapack.dylib" $VENV_DIR/bin/pip install --no-cache-dir scipy
    
    # Install other core dependencies first
    echo "Installing core dependencies..."
    $VENV_DIR/bin/pip install --no-cache-dir scikit-learn pandas
    
    # Install streamlit and its requirements
    echo "Installing Streamlit and related packages..."
    $VENV_DIR/bin/pip install --no-cache-dir streamlit watchdog
    
    # Install remaining requirements
    echo "Installing remaining requirements..."
    $VENV_DIR/bin/pip install -r "$LOCAL_DIR/requirements.txt"
    
    # Install spaCy and its model
    echo "Installing spaCy and language model..."
    $VENV_DIR/bin/pip install --no-cache-dir spacy
    $VENV_DIR/bin/python -m spacy download en_core_web_sm
}

# Function to check and install Ollama
setup_ollama() {
    echo -e "${BLUE}Checking Ollama installation...${NC}"
    
    if ! command_exists ollama; then
        echo "Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        echo "Ollama is already installed"
    fi
    
    if ! is_ollama_running; then
        echo "Starting Ollama service..."
        ollama serve &
        sleep 5  # Give Ollama time to start
    else
        echo "Ollama is already running"
    fi
}

# Function to run the application
run_app() {
    echo -e "${BLUE}Launching Ollama Workbench...${NC}"
    cd "$LOCAL_DIR"
    $VENV_DIR/bin/streamlit run main.py
}

# Set directory paths
LOCAL_DIR="$HOME/Ollama-Workbench"
VENV_DIR="$LOCAL_DIR/venv"

# Main menu function
show_menu() {
    show_header
    echo -e "1) ${GREEN}Install/Update${NC} - Fresh install or update existing installation"
    echo -e "2) ${GREEN}Run Application${NC} - Start Ollama Workbench"
    echo -e "3) ${GREEN}Check Dependencies${NC} - Verify all dependencies are installed"
    echo -e "4) ${GREEN}Exit${NC}"
    echo
    read -p "Please select an option (1-4): " choice
    
    case $choice in
        1)
            show_header
            echo "Installing/Updating Ollama Workbench..."
            setup_system_deps
            setup_python_env
            install_dependencies
            setup_ollama
            echo -e "\n${GREEN}Installation/Update complete!${NC}"
            read -p "Press Enter to return to main menu..."
            show_menu
            ;;
        2)
            show_header
            setup_python_env
            setup_ollama
            run_app
            ;;
        3)
            show_header
            setup_system_deps
            setup_python_env
            echo "Checking dependencies..."
            install_dependencies
            echo -e "\n${GREEN}Dependency check complete!${NC}"
            read -p "Press Enter to return to main menu..."
            show_menu
            ;;
        4)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option. Please try again.${NC}"
            sleep 2
            show_menu
            ;;
    esac
}

# Check if Python is installed
if ! command_exists python; then
    echo -e "${RED}Error: Python is not installed. Please install Python and try again.${NC}"
    exit 1
fi

# Start the menu
show_menu
