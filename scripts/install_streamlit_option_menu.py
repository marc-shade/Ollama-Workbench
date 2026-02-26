#!/usr/bin/env python3
"""
Script to install streamlit_option_menu in the current Python environment.
This is useful when the application is running from a different environment
than the one set up by the setup scripts.
"""

import sys
import subprocess
import importlib.util

def check_package_installed(package_name):
    """Check if a package is installed in the current Python environment."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name, version=None):
    """Install a package in the current Python environment."""
    package_spec = package_name
    if version:
        package_spec = f"{package_name}=={version}"
    
    print(f"Installing {package_spec}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        print(f"Successfully installed {package_spec}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package_spec}: {e}")
        return False

def main():
    """Main function."""
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Check if streamlit_option_menu is already installed
    if check_package_installed("streamlit_option_menu"):
        print("streamlit_option_menu is already installed")
        return True
    
    # Try to install streamlit_option_menu
    success = install_package("streamlit-option-menu", "0.3.13")
    
    if success:
        # Verify installation
        if check_package_installed("streamlit_option_menu"):
            print("Verified: streamlit_option_menu is now installed")
            return True
        else:
            print("Error: streamlit_option_menu was not installed correctly")
            return False
    else:
        print("Failed to install streamlit_option_menu")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)