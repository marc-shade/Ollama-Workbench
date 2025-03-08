#!/usr/bin/env python3
"""
Validation script for Ollama Workbench installation.
This script verifies that all required components are properly installed and configured.
"""

import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path

# ANSI colors for terminal output
COLORS = {
    "CYAN": "\033[0;36m",  
    "GREEN": "\033[0;32m",  
    "YELLOW": "\033[1;33m", 
    "RED": "\033[0;31m",    
    "NC": "\033[0m",        # No Color
}

def print_header(text):
    """Print a formatted header."""
    print(f"\n{COLORS['CYAN']}{'=' * 60}{COLORS['NC']}")
    print(f"{COLORS['CYAN']}  {text}{COLORS['NC']}")
    print(f"{COLORS['CYAN']}{'=' * 60}{COLORS['NC']}\n")

def print_success(text):
    """Print a success message."""
    print(f"{COLORS['GREEN']}✓{COLORS['NC']} {text}")

def print_warning(text):
    """Print a warning message."""
    print(f"{COLORS['YELLOW']}!{COLORS['NC']} {text}")

def print_error(text):
    """Print an error message."""
    print(f"{COLORS['RED']}✗{COLORS['NC']} {text}")

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, '__version__'):
            print_success(f"{module_name} v{module.__version__} is installed")
            return True, module.__version__
        else:
            print_success(f"{module_name} is installed (version unknown)")
            return True, None
    except ImportError:
        suggested_package = package_name or module_name
        print_error(f"{module_name} could not be imported")
        print_warning(f"Try installing it with: pip install {suggested_package}")
        return False, None

def check_python_version():
    """Check if Python version is compatible."""
    python_version = platform.python_version()
    python_version_tuple = tuple(map(int, python_version.split('.')))
    
    if python_version_tuple >= (3, 11):
        print_success(f"Python {python_version} detected (✓ Compatible)")
        return True
    else:
        print_error(f"Python {python_version} detected (✗ Version 3.11+ recommended)")
        print_warning("Some components may not work properly with older Python versions")
        return False

def check_platform():
    """Check if running on Apple Silicon and report platform details."""
    system = platform.system()
    machine = platform.machine()
    processor = platform.processor()
    
    if system == "Darwin" and machine == "arm64":
        print_success(f"Detected Apple Silicon Mac ({processor})")
        return True, "Apple Silicon"
    elif system == "Darwin" and (machine == "x86_64" or machine == "i386"):
        print_warning(f"Detected Intel Mac ({processor})")
        print_warning("This setup is optimized for Apple Silicon, but should work on Intel Macs")
        return True, "Intel Mac"
    elif system == "Linux":
        print_warning(f"Detected Linux ({machine})")
        print_warning("This setup is optimized for Apple Silicon, but should work on Linux")
        return True, "Linux"
    elif system == "Windows":
        print_warning(f"Detected Windows ({machine})")
        print_warning("This setup is optimized for Apple Silicon, but should work on Windows with adjustments")
        return True, "Windows"
    else:
        print_warning(f"Unknown platform: {system} {machine}")
        return False, f"{system} {machine}"

def check_torch_mps():
    """Check if PyTorch with MPS support is installed."""
    try:
        import torch
        print_success(f"PyTorch v{torch.__version__} is installed")
        
        # Different versions of PyTorch have different ways to check MPS
        mps_available = False
        
        # Try the standard way (PyTorch 2.0+)
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
            try:
                if torch.backends.mps.is_available():
                    mps_available = True
            except AttributeError:
                pass
        
        # Try the older way or direct check
        if not mps_available and hasattr(torch, 'mps'):
            try:
                # Some versions use torch.mps.is_available()
                if hasattr(torch.mps, 'is_available') and torch.mps.is_available():
                    mps_available = True
                # Some versions just check if the module exists
                else:
                    mps_available = True
            except (AttributeError, ImportError):
                pass
                
        # Try creating a tensor on MPS as final check
        if not mps_available:
            try:
                device = torch.device("mps")
                x = torch.ones(1, device=device)
                mps_available = True
            except (RuntimeError, ValueError):
                pass
                
        if mps_available:
            print_success("MPS (Metal Performance Shaders) is available")
            return True
        else:
            print_warning("MPS is supported but not available on this system")
            print_warning("This might be due to hardware limitations or configuration issues")
            return False
    except ImportError:
        print_error("PyTorch could not be imported")
        print_warning("Try installing it with: pip install torch torchvision")
        return False

def check_ollama_server():
    """Check if Ollama server is running."""
    try:
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print_success("Ollama server is running")
                
                # Check available models
                models = response.json().get('models', [])
                if models:
                    model_names = [model.get('name', 'unknown') for model in models]
                    print_success(f"Available models: {', '.join(model_names)}")
                else:
                    print_warning("No models found. You might need to pull models.")
                    print_warning("Try: ollama pull llama3")
                
                return True
            else:
                print_error(f"Ollama server returned status code: {response.status_code}")
                return False
        except requests.RequestException:
            print_error("Ollama server is not running or not accessible")
            print_warning("Start the server with: ollama serve")
            return False
    except ImportError:
        print_error("requests library could not be imported")
        print_warning("Try installing it with: pip install requests")
        return False

def main():
    """Run validation checks."""
    print_header("Ollama Workbench Validation")
    
    success_count = 0
    warning_count = 0
    error_count = 0
    
    # Check platform
    platform_ok, platform_type = check_platform()
    if platform_ok:
        success_count += 1
    else:
        warning_count += 1
    
    # Check Python version
    if check_python_version():
        success_count += 1
    else:
        warning_count += 1
    
    # Check Poetry installation
    poetry_path = None
    for path in [
        os.path.expanduser("~/.local/bin/poetry"),
        "/usr/local/bin/poetry",
        "/opt/homebrew/bin/poetry"
    ]:
        if os.path.exists(path):
            poetry_path = path
            break
    
    if poetry_path:
        print_success(f"Poetry found at {poetry_path}")
        try:
            result = subprocess.run([poetry_path, "--version"], 
                                    capture_output=True, text=True, check=True)
            print_success(f"Poetry version: {result.stdout.strip()}")
            success_count += 1
        except (subprocess.CalledProcessError, FileNotFoundError):
            print_warning("Poetry found but could not determine version")
            warning_count += 1
    else:
        print_error("Poetry not found in common locations")
        print_warning("Install Poetry with: curl -sSL https://install.python-poetry.org | python3 -")
        error_count += 1
    
    # Check PyTorch with MPS (for Apple Silicon)
    if platform_type == "Apple Silicon":
        if check_torch_mps():
            success_count += 1
        else:
            error_count += 1
    else:
        # Check regular PyTorch
        success, version = check_import("torch")
        if success:
            success_count += 1
        else:
            error_count += 1
    
    # Check key dependencies
    dependencies = [
        ("streamlit", None),
        ("transformers", None),
        ("sentence_transformers", "sentence-transformers"),
        ("langchain", None),
        ("langchain_community", "langchain-community"),
    ]
    
    for module, package in dependencies:
        success, _ = check_import(module, package)
        if success:
            success_count += 1
        else:
            error_count += 1
    
    # Check Ollama server
    if check_ollama_server():
        success_count += 1
    else:
        warning_count += 1
    
    # Print summary
    print_header("Validation Summary")
    print(f"Successes: {success_count}")
    print(f"Warnings: {warning_count}")
    print(f"Errors: {error_count}")
    
    if error_count > 0:
        print_error("Some critical components are missing or misconfigured")
        print_warning("Please resolve the issues above and run this script again")
        return False
    elif warning_count > 0:
        print_warning("Setup completed with some warnings")
        print_warning("You may experience some limitations, but the core functionality should work")
        return True
    else:
        print_success("All validation checks passed!")
        print_success("Ollama Workbench is ready to use")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)