#!/usr/bin/env python3
"""
Dependency checker for Ollama Workbench.
This script scans Python files for imports and checks if they're installed.
"""

import os
import sys
import re
import importlib.util
import subprocess
from collections import defaultdict

# ANSI colors for terminal output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

def print_colored(color, message):
    """Print a colored message."""
    print(f"{color}{message}{NC}")

def find_python_files(directory):
    """Find all Python files in the directory."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def extract_imports(file_path):
    """Extract import statements from a Python file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Regular expressions to match different import patterns
    import_patterns = [
        r'^import\s+([\w\.]+)',  # import module
        r'^from\s+([\w\.]+)\s+import',  # from module import ...
        r'^\s+import\s+([\w\.]+)',  # indented import module
        r'^\s+from\s+([\w\.]+)\s+import',  # indented from module import ...
    ]
    
    imports = set()
    for pattern in import_patterns:
        matches = re.findall(pattern, content, re.MULTILINE)
        imports.update(matches)
    
    # Extract the top-level module name from each import
    top_level_modules = set()
    for imp in imports:
        top_level = imp.split('.')[0]
        if top_level not in ['__future__', 'os', 'sys', 'io', 're', 'json', 'time', 'datetime', 'math', 'random', 'collections', 'itertools', 'functools', 'typing', 'pathlib', 'subprocess', 'tempfile', 'shutil', 'glob', 'argparse', 'logging', 'threading', 'multiprocessing', 'concurrent', 'urllib', 'http', 'socket', 'ssl', 'email', 'smtplib', 'ftplib', 'telnetlib', 'uuid', 'hashlib', 'hmac', 'base64', 'pickle', 'csv', 'xml', 'html', 'unittest', 'doctest', 'pdb', 'traceback', 'warnings', 'contextlib', 'abc', 'ast', 'inspect', 'importlib', 'zipfile', 'tarfile', 'gzip', 'bz2', 'lzma', 'zlib', 'struct', 'codecs', 'unicodedata', 'stringprep', 'readline', 'rlcompleter', 'stat', 'filecmp', 'fnmatch', 'linecache', 'tokenize', 'tabnanny', 'pyclbr', 'py_compile', 'compileall', 'dis', 'pickletools', 'distutils', 'ensurepip', 'venv', 'zipapp', 'platform', 'errno', 'ctypes', 'site', 'code', 'codeop', 'timeit', 'trace', 'tracemalloc', 'gc', 'inspect', 'site', 'user', 'builtins', 'copy', 'pprint', 'reprlib', 'enum', 'numbers', 'cmath', 'decimal', 'fractions', 'statistics', 'array', 'dataclasses', 'heapq', 'bisect', 'weakref', 'types', 'copy_reg', 'copyreg', 'operator', 'reprlib', 'keyword', 'parser', 'symbol', 'token', 'keyword', 'tokenize', 'tabnanny', 'pyclbr', 'py_compile', 'compileall', 'dis', 'pickletools', 'formatter', 'msilib', 'msvcrt', 'winreg', 'winsound', 'posix', 'pwd', 'spwd', 'grp', 'crypt', 'termios', 'tty', 'pty', 'fcntl', 'pipes', 'resource', 'nis', 'syslog', 'optparse', 'imp']:
            top_level_modules.add(top_level)
    
    return top_level_modules

def check_package_installed(package_name):
    """Check if a package is installed."""
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
    except Exception:
        return None

def map_import_to_package(import_name):
    """Map import name to package name."""
    # Common mappings where import name differs from package name
    mappings = {
        'PIL': 'pillow',
        'bs4': 'beautifulsoup4',
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python',
        'streamlit_option_menu': 'streamlit-option-menu',
        'streamlit_extras': 'streamlit-extras',
        'streamlit_flow': 'streamlit-flow',
        'streamlit_javascript': 'streamlit-javascript',
        'PyPDF2': 'PyPDF2',  # Ensure case-sensitive mapping
        'openai': 'openai',  # Ensure openai is mapped correctly
        'langchain_community': 'langchain-community',  # Handle underscore to hyphen conversion
        'pdfkit': 'pdfkit',  # Ensure pdfkit is mapped correctly
        'requests': 'requests',  # Ensure requests is mapped correctly
        'selenium': 'selenium',  # Ensure selenium is mapped correctly
        'webdriver_manager': 'webdriver-manager',  # Handle underscore to hyphen conversion
    }
    return mappings.get(import_name, import_name)

def suggest_installation_command(package_name):
    """Suggest installation command for a package."""
    return f"pip install {package_name}"

def main():
    """Main function."""
    print_colored(GREEN, "=== Ollama Workbench Dependency Checker ===")
    
    # Find all Python files in the current directory
    python_files = find_python_files('.')
    print_colored(GREEN, f"Found {len(python_files)} Python files")
    
    # Extract imports from each file
    file_imports = {}
    all_imports = set()
    for file_path in python_files:
        imports = extract_imports(file_path)
        file_imports[file_path] = imports
        all_imports.update(imports)
    
    print_colored(GREEN, f"Found {len(all_imports)} unique imports")
    
    # Check if each import is installed
    installed = []
    missing = []
    for import_name in sorted(all_imports):
        package_name = map_import_to_package(import_name)
        is_installed = check_package_installed(import_name)
        version = get_package_version(package_name) if is_installed else None
        
        if is_installed:
            installed.append((import_name, version))
        else:
            missing.append(import_name)
    
    # Print results
    print_colored(GREEN, "\n=== Installed Packages ===")
    for import_name, version in installed:
        version_str = f"v{version}" if version else "unknown version"
        print(f"✓ {import_name} ({version_str})")
    
    print_colored(YELLOW if missing else GREEN, f"\n=== Missing Packages ({len(missing)}) ===")
    if missing:
        for import_name in missing:
            package_name = map_import_to_package(import_name)
            print(f"✗ {import_name}")
            print(f"  Suggested: {suggest_installation_command(package_name)}")
    else:
        print("No missing packages found!")
    
    # Print imports by file
    print_colored(GREEN, "\n=== Imports by File ===")
    for file_path, imports in file_imports.items():
        if imports:
            missing_in_file = [imp for imp in imports if imp in missing]
            status = f"({len(missing_in_file)} missing)" if missing_in_file else "(all installed)"
            print(f"{file_path} {status}")
            if missing_in_file:
                for imp in missing_in_file:
                    print(f"  ✗ {imp}")
    
    # Print summary
    print_colored(GREEN, "\n=== Summary ===")
    print(f"Total imports: {len(all_imports)}")
    print(f"Installed: {len(installed)}")
    print(f"Missing: {len(missing)}")
    
    if missing:
        print_colored(YELLOW, "\n=== Installation Commands ===")
        print("Run the following commands to install all missing packages:")
        for import_name in missing:
            package_name = map_import_to_package(import_name)
            print(suggest_installation_command(package_name))
    
    return len(missing) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)