#!/usr/bin/env python3
"""
Diagnostic script to check streamlit_option_menu installation.
"""

import sys
import subprocess
import importlib.util

def check_package_installed(package_name):
    """Check if a package is installed in the current Python environment."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def get_package_version(package_name):
    """Get the installed version of a package."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
        return None
    except Exception as e:
        print(f"Error checking package version: {e}")
        return None

def main():
    """Run the diagnostic checks."""
    print("Checking streamlit_option_menu installation...")
    
    # Check if the package is installed
    is_installed = check_package_installed("streamlit_option_menu")
    print(f"streamlit_option_menu is installed: {is_installed}")
    
    # Check package version
    version = get_package_version("streamlit_option_menu")
    if version:
        print(f"Installed version: {version}")
    else:
        print("Could not determine installed version")
    
    # Check pip list for both hyphenated and underscore versions
    print("\nChecking pip list for package names:")
    for package_name in ["streamlit-option-menu", "streamlit_option_menu"]:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            import json
            packages = json.loads(result.stdout)
            found = False
            for pkg in packages:
                if pkg["name"].lower() == package_name.lower():
                    print(f"Found in pip list: {pkg['name']} (version: {pkg['version']})")
                    found = True
                    break
            if not found:
                print(f"Not found in pip list: {package_name}")
        else:
            print(f"Error checking pip list: {result.stderr}")
    
    # Suggest solutions
    print("\nPossible solutions:")
    print("1. Install the package manually:")
    print("   pip install streamlit-option-menu==0.3.13")
    print("2. Add the package to pyproject.toml:")
    print('   streamlit-option-menu = "^0.3.13"')
    print("3. Fix the setup.sh script to ensure requirements.txt is properly processed")

if __name__ == "__main__":
    main()