#!/bin/bash

# Text formatting
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Set directory paths
LOCAL_DIR="$(dirname "$0")"
VENV_DIR="$LOCAL_DIR/venv"
LOG_DIR="$LOCAL_DIR/logs"
LOG_FILE="$LOG_DIR/setup.log"

# Create necessary directories
mkdir -p "$LOG_DIR"

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
    
    # Install required packages from requirements.txt
    $VENV_DIR/bin/pip install -r "$LOCAL_DIR/requirements.txt"
    
    # Install Flask explicitly
    if ! is_package_installed "Flask"; then
        log_message "INFO" "Installing Flask..."
        $VENV_DIR/bin/pip install Flask
    fi

    # Install Ollama explicitly
    if ! is_package_installed "ollama"; then
        log_message "INFO" "Installing Ollama..."
        $VENV_DIR/bin/pip install ollama
    fi

    # Install psutil explicitly
    if ! is_package_installed "psutil"; then
        log_message "INFO" "Installing psutil..."
        $VENV_DIR/bin/pip install psutil
    fi

    # Install openai explicitly
    if ! is_package_installed "openai"; then
        log_message "INFO" "Installing openai..."
        $VENV_DIR/bin/pip install openai
    fi

    # Install groq explicitly
    if ! is_package_installed "groq"; then
        log_message "INFO" "Installing groq..."
        $VENV_DIR/bin/pip install groq
    fi

    # Install sentence_transformers explicitly
    if ! is_package_installed "sentence_transformers"; then
        log_message "INFO" "Installing sentence_transformers..."
        $VENV_DIR/bin/pip install sentence_transformers
    fi

    # Install mistralai explicitly
    if ! is_package_installed "mistralai"; then
        log_message "INFO" "Installing mistralai..."
        $VENV_DIR/bin/pip install mistralai
    fi

    # Install tiktoken explicitly
    if ! is_package_installed "tiktoken"; then
        log_message "INFO" "Installing tiktoken..."
        $VENV_DIR/bin/pip install tiktoken
    fi

    # Install PyPDF2 explicitly
    if ! is_package_installed "PyPDF2"; then
        log_message "INFO" "Installing  PyPDF2..."
        $VENV_DIR/bin/pip install PyPDF2
    fi

    # Install gtts explicitly
    if ! is_package_installed "gtts"; then
        log_message "INFO" "Installing  gtts..."
        $VENV_DIR/bin/pip install gtts
    fi

    # Install pygame explicitly
    if ! is_package_installed "pygame"; then
        log_message "INFO" "Installing  pygame..."
        $VENV_DIR/bin/pip install pygame
    fi

    # Install autogen explicitly
    if ! is_package_installed "autogen"; then
        log_message "INFO" "Installing  autogen..."
        $VENV_DIR/bin/pip install autogen
    fi

    # Install pdfkit explicitly
    if ! is_package_installed "pdfkit"; then
        log_message "INFO" "Installing  pdfkit..."
        $VENV_DIR/bin/pip install pdfkit
    fi

    # Install selenium explicitly
    if ! is_package_installed "selenium"; then
        log_message "INFO" "Installing  selenium..."
        $VENV_DIR/bin/pip install selenium
    fi

    # Install webdriver-manager explicitly
    if ! is_package_installed "webdriver-manager"; then
        log_message "INFO" "Installing  webdriver-manager..."
        $VENV_DIR/bin/pip install webdriver-manager
    fi

    # Install fake-useragent explicitly
    if ! is_package_installed "fake-useragent"; then
        log_message "INFO" "Installing  fake-useragent..."
        $VENV_DIR/bin/pip install fake-useragent
    fi

    # Install humanize explicitly
    if ! is_package_installed "humanize"; then
        log_message "INFO" "Installing  humanize..."
        $VENV_DIR/bin/pip install humanize
    fi

    # Install fpdf explicitly
    if ! is_package_installed "fpdf"; then
        log_message "INFO" "Installing  fpdf..."
        $VENV_DIR/bin/pip install fpdf
    fi

    # Install radon explicitly
    if ! is_package_installed "radon"; then
        log_message "INFO" "Installing  radon..."
        $VENV_DIR/bin/pip install radon
    fi

    # Install flake8 explicitly
    if ! is_package_installed "flake8"; then
        log_message "INFO" "Installing  flake8..."
        $VENV_DIR/bin/pip install flake8
    fi

    # Install langchain-community explicitly
    if ! is_package_installed "langchain-community"; then
        log_message "INFO" "Installing  langchain-community..."
        $VENV_DIR/bin/pip install langchain-community
    fi

    # Install duckduckgo-search explicitly
    if ! is_package_installed "duckduckgo-search"; then
        log_message "INFO" "Installing  duckduckgo-search..."
        $VENV_DIR/bin/pip install duckduckgo-search
    fi

    # Install googleapiclient explicitly
    if ! is_package_installed "googleapiclient"; then
        log_message "INFO" "Installing googleapiclient..."
        $VENV_DIR/bin/pip install google-api-python-client
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
    
    # Function to check if Ollama is running
    is_ollama_running() {
        lsof -i :11434 >/dev/null 2>&1
    }
    
    # Check if Ollama is running before starting it
    if ! is_ollama_running; then
        log_message "INFO" "Starting Ollama service..."
        # Command to start the Ollama service
        # (Add the actual command to start the service here)
    else
        log_message "INFO" "Ollama is already running"
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

    $VENV_DIR/bin/streamlit run "$LOCAL_DIR/main.py"
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
    echo -e ""
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
                python "$LOCAL_DIR/setup.py"
                log_message "SUCCESS" "Installation/Update complete!"
                read -p "Press Enter to return to main menu..."
                show_menu
            else
                log_message "ERROR" "System requirements not met"
            fi
            ;;
        2)
            show_header
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

# Start the script
log_message "INFO" "Starting Ollama Workbench Manager..."

# Check if Python is installed
if ! command_exists python3; then
    log_message "ERROR" "Python 3 is not installed. Please install Python and try again."
    exit 1
fi

# Start the menu
show_menu