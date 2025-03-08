#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

# Print styled messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info "Activating Python 3.11 virtual environment..."
source "$SCRIPT_DIR/venv/bin/activate"

# Verify Python version
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" != "3.11" ]]; then
    print_error "Wrong Python version: $PYTHON_VERSION (expected 3.11)"
    print_info "Running setup script to fix environment..."
    bash "$SCRIPT_DIR/setup.sh"
    source "$SCRIPT_DIR/venv/bin/activate"
    
    # Check again
    PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "$PYTHON_VERSION" != "3.11" ]]; then
        print_error "Could not set up Python 3.11 environment"
        exit 1
    fi
fi

print_success "Using Python $PYTHON_VERSION"

# Check for critical module imports
MISSING_MODULES=()
CRITICAL_MODULES=(
    "flask" "streamlit" "numpy" "ollama" "openai" "psutil" "langchain"
    "sklearn" "sentence_transformers" "PyPDF2" "bs4" "requests" "selenium"
    "humanize" "chromadb" "googleapiclient" "spacy" "pygame" "gtts" "pydub"
)

for module in "${CRITICAL_MODULES[@]}"; do
    # Adjust module name for import check
    import_name="$module"
    if [[ "$module" == "beautifulsoup4" ]]; then
        import_name="bs4"
    elif [[ "$module" == "google_search_results" ]]; then
        import_name="serpapi"
    elif [[ "$module" == "gputil" ]]; then
        import_name="GPUtil"
    fi
    
    if ! python -c "import $import_name" &> /dev/null; then
        print_warning "$import_name import failed! Attempting to install..."
        
        # Map module name to package name if they differ
        package_name="$module"
        if [[ "$module" == "sklearn" ]]; then
            package_name="scikit-learn"
        elif [[ "$module" == "sentence_transformers" ]]; then
            package_name="sentence-transformers"
        elif [[ "$module" == "bs4" ]]; then
            package_name="beautifulsoup4"
        elif [[ "$module" == "googleapiclient" ]]; then
            package_name="google-api-python-client"
        fi
        
        pip install $package_name
        MISSING_MODULES+=("$module")
    fi
done

if [ ${#MISSING_MODULES[@]} -gt 0 ]; then
    print_warning "Had to install missing modules: ${MISSING_MODULES[*]}"
    print_info "You may want to run setup.sh again to ensure all dependencies are properly installed"
fi

# Check for spaCy language model
print_info "Checking for spaCy language model (en_core_web_sm)..."
if ! python -c "import spacy; spacy.load('en_core_web_sm')" &> /dev/null; then
    print_warning "spaCy language model 'en_core_web_sm' not found! Installing..."
    python -m spacy download en_core_web_sm
    
    if ! python -c "import spacy; spacy.load('en_core_web_sm')" &> /dev/null; then
        print_error "Failed to install spaCy language model. Some features may not work."
    else
        print_success "spaCy language model installed successfully"
    fi
fi

# Check if Ollama is installed and running
if ! command -v ollama &> /dev/null; then
    print_warning "Ollama not found"
    echo "Would you like to install Ollama now? (y/N)"
    read -p "" install_ollama
    if [[ $install_ollama =~ ^[Yy]$ ]]; then
        print_info "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
    else
        print_warning "Proceeding without Ollama. Some features may not work."
    fi
fi

# Check if Ollama is running
if command -v ollama &> /dev/null; then
    if ! curl -s http://localhost:11434/api/version &> /dev/null; then
        print_info "Starting Ollama server..."
        ollama serve &
        # Wait for it to start
        sleep 2
    else
        print_success "Ollama server is already running"
    fi
fi

# Run the application
print_info "Starting Ollama Workbench..."
streamlit run "$SCRIPT_DIR/main.py"
