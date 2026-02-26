#!/usr/bin/env python3
"""
Comprehensive test script to verify all critical dependencies
"""

import sys
import subprocess
import importlib.util

print("Python version:", sys.version)
print("\n=== Testing Critical Dependencies ===")

# List of critical dependencies to check
critical_deps = [
    "openai",
    "PyPDF2",
    "streamlit_extras",
    "streamlit_javascript",
    "streamlit_option_menu",
    "ollama",
    "langchain",
    "langchain_community",
    "flask",
    "psutil",
    "groq",
    "mistralai",
    "matplotlib",
    "tiktoken",
    "sklearn",
    "numpy",
    "pandas",
    "chromadb",
    "pdfkit",
    "requests",
    "bs4",
    "selenium",
    "webdriver_manager"
]

# Try to import each dependency
for dep in critical_deps:
    try:
        module = importlib.import_module(dep)
        print(f"✅ Successfully imported {dep}")
        version = getattr(module, "__version__", "unknown")
        path = getattr(module, "__file__", "unknown")
        print(f"   Version: {version}")
        print(f"   Path: {path}")
    except ImportError as e:
        print(f"❌ Failed to import {dep}: {e}")

# Check what's installed with pip
print("\n=== Checking pip installation ===")
for package in [
    "openai",
    "PyPDF2",
    "streamlit-extras",
    "streamlit-javascript",
    "streamlit-option-menu",
    "ollama",
    "langchain",
    "langchain-community",
    "flask",
    "psutil",
    "groq",
    "mistralai",
    "matplotlib",
    "tiktoken",
    "scikit-learn",
    "numpy",
    "pandas",
    "chromadb",
    "pdfkit",
    "requests",
    "beautifulsoup4",
    "bs4",
    "selenium",
    "webdriver-manager"
]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print(f"✅ {package} is installed via pip:")
            for line in result.stdout.splitlines():
                if line.startswith("Name:") or line.startswith("Version:"):
                    print("  ", line)
        else:
            print(f"❌ {package} is NOT installed via pip")
    except Exception as e:
        print(f"Error checking {package} installation:", e)

# Check if spaCy language model is installed
print("\n=== Checking spaCy language model ===")
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        print(f"✅ Successfully loaded spaCy language model 'en_core_web_sm'")
        print(f"   Version: {getattr(nlp.meta, 'version', 'unknown')}")
    except Exception as e:
        print(f"❌ Failed to load spaCy language model 'en_core_web_sm': {e}")
        print("   Try installing it with: python -m spacy download en_core_web_sm")
except ImportError:
    print("❌ spaCy not installed, skipping language model check")

print("\n=== Test complete ===")