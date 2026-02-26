#!/usr/bin/env python
"""
Script to find all imports in Python files and check if they're installed.
"""

import os
import re
import sys
import importlib
import subprocess
from collections import defaultdict

def find_python_files(directory):
    """Find all Python files in the given directory."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def extract_imports(file_path):
    """Extract all import statements from a Python file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find all import statements
    import_pattern = re.compile(r'^(?:from\s+([.\w]+)\s+import\s+.*|import\s+([.\w,\s]+))', re.MULTILINE)
    matches = import_pattern.findall(content)
    
    imports = []
    for match in matches:
        if match[0]:  # from X import Y
            module = match[0].split('.')[0]
            if module:
                imports.append(module)
        else:  # import X, Y, Z
            modules = match[1].split(',')
            for module in modules:
                module = module.strip().split('.')[0]
                if module:
                    imports.append(module)
    
    return imports

def check_import(module_name):
    """Check if a module can be imported."""
    if not module_name:
        return True  # Skip empty module names
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def map_module_to_package(module_name):
    """Map module name to package name for pip install."""
    mapping = {
        'sklearn': 'scikit-learn',
        'bs4': 'beautifulsoup4',
        'PIL': 'pillow',
        'yaml': 'pyyaml',
        'googleapiclient': 'google-api-python-client',
        'cv2': 'opencv-python',
        'dotenv': 'python-dotenv',
        'matplotlib.pyplot': 'matplotlib',
        'IPython': 'ipython',
        'google.auth': 'google-auth',
        'google.oauth2': 'google-auth-oauthlib',
        'google_auth_oauthlib': 'google-auth-oauthlib',
        'google.cloud': 'google-cloud-storage',
        'jwt': 'pyjwt',
        'yaml': 'pyyaml',
        'GPUtil': 'gputil',
    }
    return mapping.get(module_name, module_name)

def main():
    """Main function."""
    directory = os.getcwd()
    python_files = find_python_files(directory)
    
    all_imports = set()
    file_imports = defaultdict(set)
    
    print(f"Scanning {len(python_files)} Python files for imports...")
    
    for file_path in python_files:
        try:
            imports = extract_imports(file_path)
            all_imports.update(imports)
            file_imports[os.path.basename(file_path)].update(imports)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Filter out standard library modules
    standard_libs = set(sys.builtin_module_names)
    try:
        standard_libs.update(os.listdir(os.path.dirname(os.__file__)))
    except (PermissionError, FileNotFoundError):
        pass
    
    # Add some common standard library modules that might not be in the above list
    standard_libs.update(['os', 'sys', 're', 'json', 'time', 'datetime', 'math', 'random', 'collections', 'typing'])
    
    third_party_imports = all_imports - standard_libs
    
    # Check which imports are missing
    missing_imports = []
    for module in third_party_imports:
        if module and not check_import(module):
            missing_imports.append(module)
    
    # Generate installation commands
    if missing_imports:
        print("\nMissing imports found:")
        for module in sorted(missing_imports):
            package = map_module_to_package(module)
            print(f"  {module} -> {package}")
        
        print("\nInstallation commands:")
        for module in sorted(missing_imports):
            package = map_module_to_package(module)
            print(f"pip install {package}")
        
        print("\nFiles with missing imports:")
        for file, imports in file_imports.items():
            missing_in_file = [imp for imp in imports if imp in missing_imports]
            if missing_in_file:
                print(f"  {file}: {', '.join(missing_in_file)}")
    else:
        print("All imports are satisfied!")

if __name__ == "__main__":
    main()