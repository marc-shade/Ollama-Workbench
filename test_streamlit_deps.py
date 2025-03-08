#!/usr/bin/env python3
"""
Test script to verify streamlit dependencies
"""

import sys
import subprocess
import importlib.util

print("Python version:", sys.version)
print("\n=== Testing streamlit dependencies ===")

# Try to import streamlit_extras
try:
    import streamlit_extras
    print("✅ Successfully imported streamlit_extras")
    print("streamlit_extras version:", getattr(streamlit_extras, "__version__", "unknown"))
    print("streamlit_extras path:", streamlit_extras.__file__)
except ImportError as e:
    print("❌ Failed to import streamlit_extras:", e)

# Try to import streamlit_javascript
try:
    import streamlit_javascript
    print("✅ Successfully imported streamlit_javascript")
    print("streamlit_javascript version:", getattr(streamlit_javascript, "__version__", "unknown"))
    print("streamlit_javascript path:", streamlit_javascript.__file__)
except ImportError as e:
    print("❌ Failed to import streamlit_javascript:", e)

# Check what's installed with pip
print("\n=== Checking pip installation ===")
for package in ["streamlit-extras", "streamlit-javascript"]:
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
                if line.startswith("Name:") or line.startswith("Version:") or line.startswith("Location:"):
                    print("  ", line)
        else:
            print(f"❌ {package} is NOT installed via pip")
    except Exception as e:
        print(f"Error checking {package} installation:", e)

print("\n=== Test complete ===")