#!/usr/bin/env python3
"""
Test script to verify pdfkit import
"""

import sys
import subprocess
import importlib.util

print("Python version:", sys.version)
print("\n=== Testing pdfkit import ===")

# Try to import pdfkit
try:
    import pdfkit
    print("✅ Successfully imported pdfkit")
    print("pdfkit version:", getattr(pdfkit, "__version__", "unknown"))
    print("pdfkit path:", pdfkit.__file__)
    
    # Check if wkhtmltopdf is installed (required by pdfkit)
    try:
        path = pdfkit.configuration().wkhtmltopdf
        if path:
            print(f"✅ wkhtmltopdf found at: {path}")
            # Try to get version
            try:
                version_output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT).decode('utf-8').strip()
                print(f"wkhtmltopdf version: {version_output}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error getting wkhtmltopdf version: {e}")
        else:
            print("❌ wkhtmltopdf path not found in configuration")
    except Exception as e:
        print(f"❌ Error checking wkhtmltopdf: {e}")
        
except ImportError as e:
    print("❌ Failed to import pdfkit:", e)

# Check what's installed with pip
print("\n=== Checking pip installation ===")
try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "pdfkit"],
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode == 0:
        print("✅ pdfkit is installed via pip:")
        for line in result.stdout.splitlines():
            if line.startswith("Name:") or line.startswith("Version:") or line.startswith("Location:"):
                print("  ", line)
    else:
        print("❌ pdfkit is NOT installed via pip")
except Exception as e:
    print("Error checking pdfkit installation:", e)

print("\n=== Test complete ===")