#!/usr/bin/env python3
"""
Test script to verify openai import
"""

import sys
import subprocess
import importlib.util

print("Python version:", sys.version)
print("\n=== Testing openai import ===")

# Try to import openai
try:
    import openai
    print("✅ Successfully imported openai")
    print("openai version:", getattr(openai, "__version__", "unknown"))
    print("openai path:", openai.__file__)
except ImportError as e:
    print("❌ Failed to import openai:", e)

# Check what's installed with pip
print("\n=== Checking pip installation ===")
try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "openai"],
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode == 0:
        print("✅ openai is installed via pip:")
        for line in result.stdout.splitlines():
            if line.startswith("Name:") or line.startswith("Version:") or line.startswith("Location:"):
                print("  ", line)
    else:
        print("❌ openai is NOT installed via pip")
except Exception as e:
    print("Error checking openai installation:", e)

print("\n=== Test complete ===")