#!/usr/bin/env python3
"""
Validation script for UI fix modules in Ollama Workbench.

This script performs syntax validation on all fix scripts to ensure
they're properly formatted without executing them.
"""

import os
import sys
import glob
import traceback

def validate_script(script_path):
    """Validate a Python script by checking its syntax."""
    try:
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Compile the script to check syntax
        compile(script_content, script_path, 'exec')
        print(f"✅ {os.path.basename(script_path)}: Syntax OK")
        return True
    except Exception as e:
        print(f"❌ {os.path.basename(script_path)}: Syntax Error")
        traceback.print_exc()
        return False

def main():
    """Main function to validate all fix scripts."""
    print("Validating Ollama Workbench fix scripts...")
    
    # Get all fix scripts
    fix_scripts = glob.glob("fix_*.py")
    
    if not fix_scripts:
        print("No fix scripts found!")
        return
    
    # Validate each script
    valid_count = 0
    for script in fix_scripts:
        if validate_script(script):
            valid_count += 1
    
    # Print summary
    print("\nValidation Summary:")
    print(f"- Total fix scripts: {len(fix_scripts)}")
    print(f"- Valid scripts: {valid_count}")
    print(f"- Invalid scripts: {len(fix_scripts) - valid_count}")
    
    if valid_count == len(fix_scripts):
        print("\n✅ All fix scripts are syntactically valid!")
    else:
        print("\n⚠️ Some fix scripts have syntax errors. Please fix them before running.")

if __name__ == "__main__":
    main()