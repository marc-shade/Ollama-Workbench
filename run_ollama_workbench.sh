#!/bin/bash

# Text formatting
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Set directory paths
LOCAL_DIR="$HOME/Ollama-Workbench"
VENV_DIR="$LOCAL_DIR/venv"
LOG_DIR="$LOCAL_DIR/logs"
LOG_FILE="$LOG_DIR/setup.log"

# Function to log messages
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "[$timestamp] [$level] $message" >> "$LOG_FILE"
    case $level in
        "INFO") echo -e "${BLUE}$message${NC}" ;;
        "SUCCESS") echo -e "${GREEN}$message${NC}" ;;
        "WARNING") echo -e "${YELLOW}$message${NC}" ;;
        "ERROR") echo -e "${RED}$message${NC}" ;;
    esac
}

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

# Function to check system requirements
check_system_requirements() {
    log_message "INFO" "Checking system requirements..."
    
    # Check Python version
    if command_exists python3; then
        local python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        local major_version=$(echo "$python_version" | cut -d. -f1)
        local minor_version=$(echo "$python_version" | cut -d. -f2)
        
        if [ "$major_version" -ge 3 ] && [ "$minor_version" -ge 8 ]; then
            log_message "SUCCESS" "Python $python_version found"
        else
            log_message "ERROR" "Python version must be 3.8 or higher (found $python_version)"
            return 1
        fi
    else
        log_message "ERROR" "Python 3 not found"
        return 1
    fi
    
    # Check disk space (convert to integer before comparison)
    local free_space=$(df -k "$HOME" | awk 'NR==2 {print int($4/1024/1024)}')
    if [ "$free_space" -lt 10 ]; then
        log_message "WARNING" "Low disk space: ${free_space}GB available. Recommend at least 10GB"
    fi
    
    # Check memory (convert to GB and compare as integer)
    local total_mem=$(sysctl -n hw.memsize)
    local total_mem_gb=$((total_mem/1024/1024/1024))
    if [ "$total_mem_gb" -lt 8 ]; then
        log_message "WARNING" "Low memory: ${total_mem_gb}GB RAM. Recommend at least 8GB"
    fi
    
    return 0
}

# Function to setup system dependencies
setup_system_deps() {
    log_message "INFO" "Setting up system dependencies..."
    
    # Check if Homebrew is installed
    if ! command_exists brew; then
        log_message "INFO" "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Function to install formula if not installed
    install_formula() {
        local formula=$1
        if ! brew list --formula | grep -q "^${formula}\$"; then
            log_message "INFO" "Installing ${formula}..."
            brew install --formula "$formula"
        else
            log_message "INFO" "${formula} is already installed"
        fi
    }
    
    # Install/Update required system packages
    log_message "INFO" "Installing/Updating system packages..."
    install_formula "gcc"
    install_formula "gfortran"
    install_formula "meson"
    install_formula "pkg-config"
    install_formula "cmake"
    install_formula "openblas"
    
    # Link OpenBLAS
    brew link --force openblas >/dev/null 2>&1
    
    # Install git if not present
    if ! command_exists git; then
        log_message "INFO" "Installing git..."
        install_formula "git"
    fi
}

# Function to setup Python environment
setup_python_env() {
    log_message "INFO" "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        log_message "INFO" "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip and essential tools
    log_message "INFO" "Upgrading pip and essential tools..."
    $VENV_DIR/bin/pip install --upgrade pip wheel setuptools build
}

# Function to install/update dependencies
install_dependencies() {
    log_message "INFO" "Installing Python dependencies..."
    
    # Install numpy with openblas first
    log_message "INFO" "Installing numpy with openblas..."
    OPENBLAS="$(brew --prefix openblas)"
    export OPENBLAS=$OPENBLAS
    export CFLAGS="-falign-functions=8 ${CFLAGS:-}"
    export ATLAS=None
    export BLAS="${OPENBLAS}/lib/libblas.dylib"
    export LAPACK="${OPENBLAS}/lib/liblapack.dylib"
    
    # First uninstall numpy and scipy if they exist
    $VENV_DIR/bin/pip uninstall -y numpy scipy
    
    # Install numpy first
    log_message "INFO" "Installing numpy 1.24.3..."
    $VENV_DIR/bin/pip install --no-cache-dir numpy==1.24.3
    
    # Install scipy next
    log_message "INFO" "Installing scipy 1.11.3..."
    $VENV_DIR/bin/pip install --no-cache-dir scipy==1.11.3
    
    # Install core dependencies first
    log_message "INFO" "Installing core dependencies..."
    $VENV_DIR/bin/pip install --no-cache-dir streamlit streamlit-option-menu streamlit-extras streamlit-javascript
    
    # Install machine learning dependencies
    log_message "INFO" "Installing machine learning packages..."
    $VENV_DIR/bin/pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    $VENV_DIR/bin/pip install --no-cache-dir transformers sentence-transformers spacy tiktoken
    
    # Install AI and LLM packages
    log_message "INFO" "Installing AI and LLM packages..."
    $VENV_DIR/bin/pip install --no-cache-dir ollama openai langchain langchain_community groq autogen pyautogen
    
    # Install web and API packages
    log_message "INFO" "Installing web and API packages..."
    $VENV_DIR/bin/pip install --no-cache-dir requests httpx beautifulsoup4 fake_useragent flask \
        duckduckgo_search google-api-python-client google_search_results serpapi selenium webdriver-manager playwright
    
    # Install utility packages
    log_message "INFO" "Installing utility packages..."
    $VENV_DIR/bin/pip install --no-cache-dir psutil GPUtil rich tqdm humanize
    
    # Install document processing packages
    log_message "INFO" "Installing document processing packages..."
    $VENV_DIR/bin/pip install --no-cache-dir PyPDF2 fpdf pdfkit reportlab mdutils
    
    # Install development packages
    log_message "INFO" "Installing development packages..."
    $VENV_DIR/bin/pip install --no-cache-dir pytest pytest-html flake8 radon ruff Pygments PyYAML
    
    # Install database packages
    log_message "INFO" "Installing database packages..."
    $VENV_DIR/bin/pip install --no-cache-dir chromadb
    
    # Install spaCy model
    log_message "INFO" "Installing spaCy language model..."
    $VENV_DIR/bin/python -m spacy download en_core_web_sm
    
    # Verify all packages installed correctly
    local missing_packages=()
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^#.*$ ]] && continue
        [[ -z "$line" ]] && continue
        
        # Extract package name (remove version specifier)
        local package_name=$(echo "$line" | cut -d'>' -f1 | cut -d'=' -f1 | tr -d ' ')
        if [ -n "$package_name" ] && ! is_package_installed "$package_name"; then
            missing_packages+=("$package_name")
        fi
    done < "$LOCAL_DIR/requirements.txt"
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        log_message "WARNING" "Some packages failed to install. Retrying..."
        for package in "${missing_packages[@]}"; do
            log_message "INFO" "Retrying installation of $package..."
            $VENV_DIR/bin/pip install --no-cache-dir "$package"
        done
    fi
    
    log_message "SUCCESS" "Package installation complete"
}

# Function to check and install Ollama
setup_ollama() {
    log_message "INFO" "Checking Ollama installation..."
    
    if ! command_exists ollama; then
        log_message "INFO" "Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        log_message "SUCCESS" "Ollama is already installed"
    fi
    
    if ! is_ollama_running; then
        log_message "INFO" "Starting Ollama service..."
        ollama serve &
        sleep 5  # Give Ollama time to start
    else
        log_message "SUCCESS" "Ollama is already running"
    fi
}

# Function to run health checks
run_health_checks() {
    log_message "INFO" "Running health checks..."
    
    # Check if critical services are running
    if ! is_ollama_running; then
        log_message "ERROR" "Ollama service is not running"
        return 1
    fi
    
    # Check if critical packages are installed
    local missing_packages=()
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^#.*$ ]] && continue
        [[ -z "$line" ]] && continue
        
        # Extract package name (remove version specifier)
        local package_name=$(echo "$line" | cut -d'>' -f1 | cut -d'=' -f1 | tr -d ' ')
        if [ -n "$package_name" ] && ! is_package_installed "$package_name"; then
            missing_packages+=("$package_name")
        fi
    done < "$LOCAL_DIR/requirements.txt"
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        log_message "ERROR" "Missing packages: ${missing_packages[*]}"
        return 1
    fi
    
    log_message "SUCCESS" "All health checks passed"
    return 0
}

# Function to run the application
run_app() {
    log_message "INFO" "Launching Ollama Workbench..."
    cd "$LOCAL_DIR"
    
    # Run health checks before starting
    if ! run_health_checks; then
        log_message "ERROR" "Health checks failed. Please run the installation/update option first."
        return 1
    fi
    
    $VENV_DIR/bin/streamlit run main.py
}

# Function to clean installation
clean_install() {
    log_message "INFO" "Cleaning installation..."
    
    # Remove virtual environment
    if [ -d "$VENV_DIR" ]; then
        rm -rf "$VENV_DIR"
    fi
    
    # Remove logs
    if [ -d "$LOG_DIR" ]; then
        rm -rf "$LOG_DIR"
    fi
    
    log_message "SUCCESS" "Clean-up complete"
}

# Main menu function
show_menu() {
    show_header
    echo -e "1) ${GREEN}Install/Update${NC} - Fresh install or update existing installation"
    echo -e "2) ${GREEN}Run Application${NC} - Start Ollama Workbench"
    echo -e "3) ${GREEN}Health Check${NC} - Run system and dependency checks"
    echo -e "4) ${GREEN}Clean Install${NC} - Remove and reinstall everything"
    echo -e "5) ${GREEN}Exit${NC}"
    echo
    read -p "Please select an option (1-5): " choice
    
    case $choice in
        1)
            show_header
            log_message "INFO" "Starting installation/update process..."
            if check_system_requirements; then
                setup_system_deps
                setup_python_env
                install_dependencies
                setup_ollama
                log_message "SUCCESS" "Installation/Update complete!"
            else
                log_message "ERROR" "System requirements not met"
            fi
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
            if check_system_requirements && run_health_checks; then
                log_message "SUCCESS" "All checks passed!"
            else
                log_message "WARNING" "Some checks failed. Please check the logs."
            fi
            read -p "Press Enter to return to main menu..."
            show_menu
            ;;
        4)
            show_header
            read -p "This will remove all existing installations. Are you sure? (y/N): " confirm
            if [[ $confirm == [yY] ]]; then
                clean_install
                show_menu
            else
                show_menu
            fi
            ;;
        5)
            log_message "INFO" "Exiting..."
            exit 0
            ;;
        *)
            log_message "ERROR" "Invalid option"
            sleep 2
            show_menu
            ;;
    esac
}

# Create necessary directories
mkdir -p "$LOG_DIR"

# Start the script
log_message "INFO" "Starting Ollama Workbench Manager..."

# Check if Python is installed
if ! command_exists python3; then
    log_message "ERROR" "Python 3 is not installed. Please install Python and try again."
    exit 1
fi

# Start the menu
show_menu
