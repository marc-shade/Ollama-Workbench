#!/usr/bin/env python3
"""
Test script to check if the humanize package is installed and can be imported.
"""

import sys
import subprocess

print("Python version:", sys.version)
print("\n=== Testing humanize package ===")

# Try to import humanize
try:
    import humanize
    print(f"✅ Successfully imported humanize")
    print(f"   Version: {getattr(humanize, '__version__', 'unknown')}")
    print(f"   Path: {getattr(humanize, '__file__', 'unknown')}")
    
    # Test some functions
    print("\n=== Testing humanize functions ===")
    print(f"naturalsize(1024): {humanize.naturalsize(1024)}")
    print(f"naturaldelta(3600): {humanize.naturaldelta(3600)}")
except ImportError as e:
    print(f"❌ Failed to import humanize: {e}")

# Check what's installed with pip
print("\n=== Checking pip installation ===")
try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "humanize"],
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode == 0:
        print(f"✅ humanize is installed via pip:")
        for line in result.stdout.splitlines():
            if line.startswith("Name:") or line.startswith("Version:") or line.startswith("Location:"):
                print("  ", line)
    else:
        print(f"❌ humanize is NOT installed via pip")
        
        # Try to install it
        print("\n=== Attempting to install humanize ===")
        install_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "humanize"],
            capture_output=True,
            text=True,
            check=False
        )
        if install_result.returncode == 0:
            print(f"✅ Successfully installed humanize")
            print(install_result.stdout)
        else:
            print(f"❌ Failed to install humanize")
            print(install_result.stderr)
except Exception as e:
    print(f"Error checking humanize installation:", e)

print("\n=== Test complete ===")