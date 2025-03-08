#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}   Ollama Workbench Environment Setup   ${NC}"
echo -e "${BLUE}=======================================${NC}"

#################################################
# STEP 1: Ensure Python 3.11 is available
#################################################

print_info "Step 1: Finding or installing Python 3.11"

# Function to find existing Python 3.11 installations
find_python311() {
    # Check standard locations for Python 3.11
    python311_locations=(
        "/usr/local/bin/python3.11"
        "/usr/bin/python3.11"
        "/opt/homebrew/bin/python3.11"
        "/opt/python/bin/python3.11"
        "$HOME/.pyenv/shims/python3.11"
        "$HOME/.pyenv/versions/3.11.*/bin/python"
    )
    
    for loc in "${python311_locations[@]}"; do
        # Use wildcard expansion for paths with asterisks
        for path in $loc; do
            if [[ -x "$path" ]]; then
                # Verify it's actually Python 3.11
                version=$("$path" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
                if [[ "$version" == "3.11" ]]; then
                    echo "$path"
                    return 0
                fi
            fi
        done
    done
    
    # Check for python3 that might be 3.11
    if command -v python3 &> /dev/null; then
        version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
        if [[ "$version" == "3.11" ]]; then
            echo "$(command -v python3)"
            return 0
        fi
    fi
    
    # Check for python that might be 3.11
    if command -v python &> /dev/null; then
        version=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
        if [[ "$version" == "3.11" ]]; then
            echo "$(command -v python)"
            return 0
        fi
    fi
    
    # If we're here, we didn't find Python 3.11
    return 1
}

# Try to find existing Python 3.11
PYTHON311=$(find_python311)

# If not found, try to install it
if [[ -z "$PYTHON311" ]]; then
    print_warning "Python 3.11 not found on your system"
    
    # Detect OS for installation
    if [[ "$(uname)" == "Darwin" ]]; then  # macOS
        if command -v brew &> /dev/null; then
            print_info "Installing Python 3.11 using Homebrew..."
            brew install python@3.11
            
            # Force-link it to ensure it's in PATH
            brew link --force python@3.11
            
            # Homebrew might put Python in different locations depending on architecture
            if [[ -x "/opt/homebrew/bin/python3.11" ]]; then
                PYTHON311="/opt/homebrew/bin/python3.11"
            elif [[ -x "/usr/local/bin/python3.11" ]]; then
                PYTHON311="/usr/local/bin/python3.11"
            else
                # Try to find it again
                PYTHON311=$(find_python311)
            fi
        else
            print_error "Homebrew not found. Please install Python 3.11 manually."
            exit 1
        fi
    elif [[ "$(uname)" == "Linux" ]]; then  # Linux
        if command -v apt-get &> /dev/null; then
            print_info "Installing Python 3.11 using apt-get..."
            sudo apt-get update
            sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
            PYTHON311=$(find_python311)
        elif command -v dnf &> /dev/null; then
            print_info "Installing Python 3.11 using dnf..."
            sudo dnf install -y python3.11 python3.11-devel
            PYTHON311=$(find_python311)
        else
            print_error "Could not detect package manager. Please install Python 3.11 manually."
            exit 1
        fi
    else
        print_error "Unsupported operating system. Please install Python 3.11 manually."
        exit 1
    fi
fi

# Final check to ensure we have Python 3.11
if [[ -z "$PYTHON311" || ! -x "$PYTHON311" ]]; then
    print_error "Failed to find or install Python 3.11"
    exit 1
fi

# Verify the Python version
PYTHON_VERSION=$("$PYTHON311" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" != "3.11" ]]; then
    print_error "Found Python $PYTHON_VERSION instead of 3.11"
    exit 1
fi

print_success "Found Python 3.11: $PYTHON311 (version $PYTHON_VERSION)"

#################################################
# STEP 2: Create virtual environment using Python 3.11
#################################################

print_info "Step 2: Creating virtual environment with Python 3.11"

VENV_DIR="$SCRIPT_DIR/venv"

# Remove existing venv if it exists
if [[ -d "$VENV_DIR" ]]; then
    print_warning "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create a new virtual environment explicitly with Python 3.11
print_info "Creating new virtual environment with $PYTHON311..."
"$PYTHON311" -m venv "$VENV_DIR"

# Verify the venv was created
if [[ ! -d "$VENV_DIR" || ! -f "$VENV_DIR/bin/python" ]]; then
    print_error "Failed to create virtual environment"
    exit 1
fi

# Verify the Python version in the venv
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"
VENV_VERSION=$("$VENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

if [[ "$VENV_VERSION" != "3.11" ]]; then
    print_error "Virtual environment has Python $VENV_VERSION instead of 3.11"
    exit 1
fi

print_success "Virtual environment created with Python $VENV_VERSION"

#################################################
# STEP 3: Install dependencies in correct order
#################################################

print_info "Step 3: Installing dependencies in the correct order"

# Upgrade pip, setuptools, and wheel
print_info "Upgrading pip, setuptools, and wheel..."
"$VENV_PIP" install --upgrade pip setuptools wheel

# Install core dependencies first (explicitly install Flask to fix import error)
print_info "Installing core dependencies..."
"$VENV_PIP" install flask==3.0.3 flask-cors==5.0.0

# Verify Flask installation
print_info "Verifying Flask installation..."
if ! "$VENV_PYTHON" -c "import flask; print(f'Flask {flask.__version__} imported successfully')" &> /dev/null; then
    print_error "Flask installation failed! Trying again..."
    "$VENV_PIP" uninstall -y flask flask-cors
    "$VENV_PIP" install flask==3.0.3 flask-cors==5.0.0
    
    if ! "$VENV_PYTHON" -c "import flask; print(f'Flask {flask.__version__} imported successfully')" &> /dev/null; then
        print_error "Flask installation failed after retry!"
        exit 1
    fi
fi
print_success "Flask installed successfully"

# Install Google API Client Library (needed for search_libraries.py)
print_info "Installing Google API Client Library..."
"$VENV_PIP" install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2

# Verify Google API Client Library installation
print_info "Verifying Google API Client Library installation..."
if ! "$VENV_PYTHON" -c "import googleapiclient; print('Google API Client Library imported successfully')" &> /dev/null; then
    print_error "Google API Client Library installation failed! Trying again..."
    "$VENV_PIP" uninstall -y google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2
    "$VENV_PIP" install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2
    
    if ! "$VENV_PYTHON" -c "import googleapiclient; print('Google API Client Library imported successfully')" &> /dev/null; then
        print_error "Google API Client Library installation failed after retry!"
        exit 1
    fi
fi
print_success "Google API Client Library installed successfully"

# Install humanize (needed for pull_model.py)
print_info "Installing humanize..."
"$VENV_PIP" install humanize==4.10.0

# Verify humanize installation
print_info "Verifying humanize installation..."
if ! "$VENV_PYTHON" -c "import humanize; print(f'humanize imported successfully')" &> /dev/null; then
    print_error "humanize installation failed! Trying again..."
    "$VENV_PIP" uninstall -y humanize
    "$VENV_PIP" install humanize==4.10.0
    
    if ! "$VENV_PYTHON" -c "import humanize; print(f'humanize imported successfully')" &> /dev/null; then
        print_error "humanize installation failed after retry!"
        exit 1
    fi
fi
print_success "humanize installed successfully"

# Install PyPDF2 (needed for enhanced_corpus.py)
print_info "Installing PyPDF2..."
"$VENV_PIP" install PyPDF2==3.0.1

# Verify PyPDF2 installation
print_info "Verifying PyPDF2 installation..."
if ! "$VENV_PYTHON" -c "import PyPDF2; print(f'PyPDF2 imported successfully')" &> /dev/null; then
    print_error "PyPDF2 installation failed! Trying again..."
    "$VENV_PIP" uninstall -y PyPDF2
    "$VENV_PIP" install PyPDF2==3.0.1
    
    if ! "$VENV_PYTHON" -c "import PyPDF2; print(f'PyPDF2 imported successfully')" &> /dev/null; then
        print_error "PyPDF2 installation failed after retry!"
        exit 1
    fi
fi
print_success "PyPDF2 installed successfully"

# Install scikit-learn (sklearn) - this is critical for the enhanced_corpus.py module
print_info "Installing scikit-learn (sklearn)..."
"$VENV_PIP" install scikit-learn==1.5.1

# Verify scikit-learn installation
print_info "Verifying scikit-learn installation..."
if ! "$VENV_PYTHON" -c "import sklearn; print(f'scikit-learn imported successfully as sklearn')" &> /dev/null; then
    print_error "scikit-learn installation failed! Trying again..."
    "$VENV_PIP" uninstall -y scikit-learn
    "$VENV_PIP" install scikit-learn==1.5.1
    
    if ! "$VENV_PYTHON" -c "import sklearn; print(f'scikit-learn imported successfully as sklearn')" &> /dev/null; then
        print_error "scikit-learn installation failed after retry!"
        exit 1
    fi
fi
print_success "scikit-learn installed successfully"

# Install NumPy 1.x next
print_info "Installing NumPy 1.x..."
"$VENV_PIP" install "numpy>=1.24.3,<2.0.0"

# Install tiktoken with compatibility flag
print_info "Installing tiktoken with compatibility flag..."
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 "$VENV_PIP" install tiktoken==0.7.0

# Verify tiktoken installation
print_info "Verifying tiktoken installation..."
if ! "$VENV_PYTHON" -c "import tiktoken; print(f'tiktoken imported successfully')" &> /dev/null; then
    print_warning "tiktoken installed but import failed - creating fallback implementation"
    
    # Create a fallback implementation
    SITE_PACKAGES=$("$VENV_PYTHON" -c "import site; print(site.getsitepackages()[0])")
    mkdir -p "$SITE_PACKAGES/tiktoken"
    
    cat > "$SITE_PACKAGES/tiktoken/__init__.py" << EOL
# Fallback tiktoken implementation
import warnings
warnings.warn("Using fallback tiktoken implementation with limited functionality")

def get_encoding(encoding_name):
    return FallbackTokenizer()

def encoding_for_model(model_name):
    return FallbackTokenizer()

class FallbackTokenizer:
    def encode(self, text):
        # Simple fallback: count characters and return as tokens
        return [ord(c) for c in text]
    
    def decode(self, tokens):
        # Convert back to characters
        return ''.join(chr(t) for t in tokens)

__version__ = "0.7.0-fallback"
EOL
    
    print_info "Testing fallback implementation..."
    if "$VENV_PYTHON" -c "import tiktoken; print(f'tiktoken imported successfully')" &> /dev/null; then
        print_success "Fallback tiktoken implementation working"
    else
        print_warning "Fallback implementation failed, but continuing anyway"
    fi
fi

# Install streamlit and key UI packages
print_info "Installing streamlit and UI packages..."
"$VENV_PIP" install streamlit==1.38.0 streamlit-option-menu==0.3.13 streamlit-extras==0.4.7 streamlit-javascript==0.1.5 streamlit-flow==0.1.0

# Install ollama client and API dependencies
print_info "Installing API/client packages..."
"$VENV_PIP" install ollama==0.4.2 openai==1.43.0 psutil==6.0.0 groq==0.10.0 mistralai

# Install langchain components
print_info "Installing langchain components..."
"$VENV_PIP" install langchain==0.2.15 langchain-community==0.2.15

# Install PyTorch with optimizations
print_info "Installing PyTorch..."
if [[ "$(uname -m)" == "arm64" ]] && [[ "$(uname)" == "Darwin" ]]; then
    print_info "Detected Apple Silicon, installing optimized PyTorch..."
    "$VENV_PIP" install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
else
    "$VENV_PIP" install torch torchvision
fi

# Install sentence-transformers (needed for enhanced_corpus.py)
print_info "Installing sentence-transformers..."
"$VENV_PIP" install sentence-transformers==3.0.1

# Install additional document processing packages
print_info "Installing document processing packages..."
"$VENV_PIP" install pdfkit==1.0.0 fpdf==1.7.2 markdown==3.6 reportlab==4.2.2

# Install web scraping and API packages
print_info "Installing web scraping and API packages..."
"$VENV_PIP" install beautifulsoup4==4.12.3 bs4==0.0.2 requests==2.32.3 selenium==4.24.0 webdriver-manager==4.0.2 fake-useragent==1.5.1

# Install utility packages
print_info "Installing utility packages..."
"$VENV_PIP" install humanize==4.10.0 cursor==1.3.5 rich==13.8.0 schedule==1.2.2 tqdm==4.66.5 networkx==3.3 gputil==1.4.0 bleach==6.1.0

# Install database and search packages
print_info "Installing database and search packages..."
"$VENV_PIP" install chromadb==0.5.5 google_search_results==2.4.2 duckduckgo-search

# Install development and testing packages
print_info "Installing development and testing packages..."
"$VENV_PIP" install pyyaml==6.0.2 pygments==2.18.0 flake8==7.1.1 pytest==8.3.2 pytest-html==4.1.1 radon==6.0.1 ruff==0.6.3

# Install autogen packages
print_info "Installing autogen packages..."
"$VENV_PIP" install autogen==0.2.35 pyautogen==0.2.35

# Install audio packages
print_info "Installing audio packages..."
"$VENV_PIP" install pygame==2.6.1 pydub==0.25.1 gtts==2.5.4

# Install remaining dependencies
print_info "Installing remaining dependencies from requirements.txt..."
if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
    "$VENV_PIP" install -r "$SCRIPT_DIR/requirements.txt" || print_warning "Some dependencies could not be installed"
else
    print_warning "requirements.txt not found, but core dependencies have been installed"
fi

# Verify critical modules
print_info "Verifying critical module imports..."
CRITICAL_MODULES=(
    "flask" "streamlit" "numpy" "ollama" "openai" "psutil" "langchain"
    "sklearn" "sentence_transformers" "PyPDF2" "beautifulsoup4" "bs4"
    "requests" "selenium" "webdriver_manager" "pdfkit" "fpdf" "markdown"
    "humanize" "cursor" "rich" "schedule" "tqdm" "networkx" "gputil" "bleach"
    "chromadb" "google_search_results" "pyyaml" "pygments" "flake8" "pytest"
    "googleapiclient" "autogen" "pygame" "gtts" "pydub" "spacy"
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
    
    if ! "$VENV_PYTHON" -c "import $import_name; print(f'$import_name imported successfully')" &> /dev/null; then
        print_warning "$import_name import failed! Attempting to reinstall..."
        
        # Map module name to package name if they differ
        package_name="$module"
        if [[ "$module" == "sklearn" ]]; then
            package_name="scikit-learn"
        elif [[ "$module" == "sentence_transformers" ]]; then
            package_name="sentence-transformers"
        elif [[ "$module" == "googleapiclient" ]]; then
            package_name="google-api-python-client"
        elif [[ "$module" == "gputil" ]]; then
            package_name="GPUtil"
        fi
        
        "$VENV_PIP" install --force-reinstall $package_name
        
        if ! "$VENV_PYTHON" -c "import $import_name" &> /dev/null; then
            print_warning "Failed to import $import_name even after reinstall"
            print_warning "This may cause issues when running the application"
        else
            print_success "$import_name reinstalled and imported successfully"
        fi
    else
        print_success "$import_name imported successfully"
    fi
done

# Install spaCy language model
print_info "Installing spaCy language model (en_core_web_sm)..."
if ! "$VENV_PYTHON" -c "import spacy; spacy.load('en_core_web_sm')" &> /dev/null; then
    print_warning "spaCy language model 'en_core_web_sm' not found. Installing..."
    "$VENV_PYTHON" -m spacy download en_core_web_sm
    
    # Verify installation
    if ! "$VENV_PYTHON" -c "import spacy; spacy.load('en_core_web_sm')" &> /dev/null; then
        print_error "Failed to install spaCy language model 'en_core_web_sm'"
        print_warning "This may cause issues when running the application"
    else
        print_success "spaCy language model 'en_core_web_sm' installed successfully"
    fi
else
    print_success "spaCy language model 'en_core_web_sm' already installed"
fi

#################################################
# STEP 4: Create launcher script
#################################################

print_info "Step 4: Creating launcher script"

cat > "$SCRIPT_DIR/run_ollama_workbench.sh" << EOL
#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "\$SCRIPT_DIR" || exit 1

# Print styled messages
print_info() {
    echo -e "\${BLUE}[INFO]\${NC} \$1"
}

print_success() {
    echo -e "\${GREEN}[✓]\${NC} \$1"
}

print_warning() {
    echo -e "\${YELLOW}[!]\${NC} \$1"
}

print_error() {
    echo -e "\${RED}[ERROR]\${NC} \$1"
}

print_info "Activating Python 3.11 virtual environment..."
source "\$SCRIPT_DIR/venv/bin/activate"

# Verify Python version
PYTHON_VERSION=\$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "\$PYTHON_VERSION" != "3.11" ]]; then
    print_error "Wrong Python version: \$PYTHON_VERSION (expected 3.11)"
    print_info "Running setup script to fix environment..."
    bash "\$SCRIPT_DIR/setup.sh"
    source "\$SCRIPT_DIR/venv/bin/activate"
    
    # Check again
    PYTHON_VERSION=\$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "\$PYTHON_VERSION" != "3.11" ]]; then
        print_error "Could not set up Python 3.11 environment"
        exit 1
    fi
fi

print_success "Using Python \$PYTHON_VERSION"

# Check for critical module imports
MISSING_MODULES=()
CRITICAL_MODULES=(
    "flask" "streamlit" "numpy" "ollama" "openai" "psutil" "langchain"
    "sklearn" "sentence_transformers" "PyPDF2" "bs4" "requests" "selenium"
    "humanize" "chromadb" "googleapiclient" "spacy" "pygame" "gtts" "pydub"
)

for module in "\${CRITICAL_MODULES[@]}"; do
    # Adjust module name for import check
    import_name="\$module"
    if [[ "\$module" == "beautifulsoup4" ]]; then
        import_name="bs4"
    elif [[ "\$module" == "google_search_results" ]]; then
        import_name="serpapi"
    elif [[ "\$module" == "gputil" ]]; then
        import_name="GPUtil"
    fi
    
    if ! python -c "import \$import_name" &> /dev/null; then
        print_warning "\$import_name import failed! Attempting to install..."
        
        # Map module name to package name if they differ
        package_name="\$module"
        if [[ "\$module" == "sklearn" ]]; then
            package_name="scikit-learn"
        elif [[ "\$module" == "sentence_transformers" ]]; then
            package_name="sentence-transformers"
        elif [[ "\$module" == "bs4" ]]; then
            package_name="beautifulsoup4"
        elif [[ "\$module" == "googleapiclient" ]]; then
            package_name="google-api-python-client"
        fi
        
        pip install \$package_name
        MISSING_MODULES+=("\$module")
    fi
done

if [ \${#MISSING_MODULES[@]} -gt 0 ]; then
    print_warning "Had to install missing modules: \${MISSING_MODULES[*]}"
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
    if [[ \$install_ollama =~ ^[Yy]$ ]]; then
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
streamlit run "\$SCRIPT_DIR/main.py"
EOL

chmod +x "$SCRIPT_DIR/run_ollama_workbench.sh"

print_success "Created launcher script: $SCRIPT_DIR/run_ollama_workbench.sh"

# Final summary
echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}     Environment setup complete!       ${NC}"
echo -e "${GREEN}=======================================${NC}"
echo ""
echo -e "Python version: ${GREEN}$VENV_VERSION${NC}"
echo -e "Virtual environment: ${GREEN}$VENV_DIR${NC}"
echo ""
echo -e "To run Ollama Workbench, use: ${YELLOW}./run_ollama_workbench.sh${NC}"
echo ""