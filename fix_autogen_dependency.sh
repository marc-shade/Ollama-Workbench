#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

print_header() {
    echo -e "${CYAN}=======================================${NC}"
    echo -e "${CYAN}   $1   ${NC}"
    echo -e "${CYAN}=======================================${NC}"
}

# Check if requirements_fixed.txt exists
if [ ! -f "requirements_fixed.txt" ]; then
    print_error "requirements_fixed.txt not found. Please run the script in the correct directory."
    exit 1
fi

print_header "Fixing Autogen Dependency Issues"

# Create a backup of the original requirements.txt
if [ -f "requirements.txt" ]; then
    print_info "Creating backup of original requirements.txt"
    cp requirements.txt requirements.txt.bak
    print_success "Backup created as requirements.txt.bak"
fi

# Replace requirements.txt with the fixed version
print_info "Replacing requirements.txt with fixed version"
cp requirements_fixed.txt requirements.txt
print_success "requirements.txt updated"

# Create a compatibility layer for autogen imports
print_info "Creating compatibility layer for autogen imports"

mkdir -p autogen_compat
cat > autogen_compat/__init__.py << 'EOL'
"""
Compatibility layer for autogen imports.
This module redirects imports from 'autogen' to 'pyautogen'.
"""

import sys
import importlib
import warnings

# Show a warning about the compatibility layer
warnings.warn(
    "Using autogen_compat layer: 'autogen' package is not available, redirecting to 'pyautogen'",
    ImportWarning
)

# Try to import pyautogen
try:
    import pyautogen
except ImportError:
    raise ImportError(
        "Neither 'autogen' nor 'pyautogen' package is installed. "
        "Please install pyautogen with: pip install pyautogen>=0.2.0"
    )

# Add the module to sys.modules
sys.modules['autogen'] = pyautogen

# Also make submodules available
for submodule_name in [
    'agentchat', 'cache', 'coding', 'oai', 'token_count_utils',
    'browser_utils', 'code_utils', 'exception_utils', 'formatting_utils',
    'function_utils', 'graph_utils', 'retrieve_utils', 'runtime_logging', 'types'
]:
    try:
        submodule = importlib.import_module(f'pyautogen.{submodule_name}')
        sys.modules[f'autogen.{submodule_name}'] = submodule
    except ImportError:
        # If the submodule doesn't exist in pyautogen, just skip it
        pass
EOL

print_success "Created autogen_compat/__init__.py"

# Create a script to modify PYTHONPATH
cat > use_autogen_compat.sh << 'EOL'
#!/bin/bash

# Add the current directory to PYTHONPATH to enable the compatibility layer
export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "Added $(pwd) to PYTHONPATH for autogen compatibility layer"

# Run the command with the modified PYTHONPATH
"$@"
EOL

chmod +x use_autogen_compat.sh
print_success "Created use_autogen_compat.sh"

# Install the fixed dependencies
print_header "Installing Fixed Dependencies"
print_info "Installing dependencies from requirements.txt"
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    print_success "Dependencies installed successfully"
else
    print_error "Failed to install dependencies"
    exit 1
fi

print_header "Fix Complete"
print_info "To run your application with the compatibility layer, use:"
echo -e "${YELLOW}./use_autogen_compat.sh streamlit run main.py${NC}"
print_info "This will ensure that 'import autogen' statements are redirected to 'pyautogen'"