#!/usr/bin/env python3
"""
Test script to verify all imports in the application
"""

import sys
import importlib.util
import os

print("Python version:", sys.version)
print("\n=== Testing Application Imports ===")

# List of files to check
files_to_check = [
    "main.py",
    "ui_elements.py",
    "chat_interface.py",
    "brainstorm.py",
    "enhanced_corpus.py"
]

# Function to extract imports from a file
def extract_imports(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Simple regex to match import statements
    import_lines = []
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('import ') or line.startswith('from '):
            # Skip comments
            if '#' in line:
                line = line[:line.index('#')].strip()
            import_lines.append(line)
    
    return import_lines

# Function to extract module names from import statements
def extract_modules(import_lines):
    modules = set()
    for line in import_lines:
        if line.startswith('import '):
            # Handle multiple imports on one line (import x, y, z)
            parts = line[7:].split(',')
            for part in parts:
                # Handle "import x as y"
                module = part.strip().split(' as ')[0].strip()
                modules.add(module)
        elif line.startswith('from '):
            # Handle "from x import y"
            module = line[5:].split(' import ')[0].strip()
            modules.add(module)
    
    return modules

# Process each file
all_modules = set()
for file_path in files_to_check:
    if os.path.exists(file_path):
        print(f"\nExtracting imports from {file_path}:")
        import_lines = extract_imports(file_path)
        modules = extract_modules(import_lines)
        
        print(f"Found {len(modules)} modules in {file_path}:")
        for module in sorted(modules):
            print(f"  - {module}")
        
        all_modules.update(modules)
    else:
        print(f"File not found: {file_path}")

# Try to import each module
print("\n=== Testing imports for all modules ===")
successful_imports = 0
failed_imports = 0

for module in sorted(all_modules):
    # Skip relative imports and built-in modules
    if module.startswith('.') or module in sys.builtin_module_names:
        continue
    
    # Get the top-level module name
    top_module = module.split('.')[0]
    
    try:
        importlib.import_module(top_module)
        print(f"✅ Successfully imported {top_module}")
        successful_imports += 1
    except ImportError as e:
        print(f"❌ Failed to import {top_module}: {e}")
        failed_imports += 1

print(f"\nImport test complete: {successful_imports} successful, {failed_imports} failed")