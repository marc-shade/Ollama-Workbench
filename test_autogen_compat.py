#!/usr/bin/env python3
"""
Test script to verify the autogen compatibility layer works correctly.
"""

import sys
import os

def test_autogen_import():
    """Test importing autogen modules through the compatibility layer."""
    try:
        print("Attempting to import autogen...")
        import autogen
        print(f"✓ Successfully imported autogen (actual package: {autogen.__name__})")
        
        # Try to import some common submodules
        print("\nTesting submodule imports:")
        
        try:
            from autogen import ConversableAgent
            print(f"✓ Successfully imported ConversableAgent")
        except ImportError as e:
            print(f"✗ Failed to import ConversableAgent: {e}")
        
        try:
            from autogen import UserProxyAgent
            print(f"✓ Successfully imported UserProxyAgent")
        except ImportError as e:
            print(f"✗ Failed to import UserProxyAgent: {e}")
        
        try:
            from autogen.oai import openai_utils
            print(f"✓ Successfully imported autogen.oai.openai_utils")
        except ImportError as e:
            print(f"✗ Failed to import autogen.oai.openai_utils: {e}")
        
        # Print version information
        if hasattr(autogen, '__version__'):
            print(f"\nAutogen version: {autogen.__version__}")
        else:
            print("\nAutogen version information not available")
        
        return True
    except ImportError as e:
        print(f"✗ Failed to import autogen: {e}")
        return False

def check_environment():
    """Check if the environment is set up correctly."""
    print("Environment Information:")
    print(f"Python version: {sys.version}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    # Check if ag2 is installed
    try:
        import ag2
        print(f"✓ ag2 is installed (version: {ag2.__version__ if hasattr(ag2, '__version__') else 'unknown'})")
    except ImportError:
        print("✗ ag2 is not installed")
    
    # Check if the compatibility layer directory is in the path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir in sys.path or '' in sys.path:
        print("✓ Current directory is in sys.path")
    else:
        print("✗ Current directory is not in sys.path")
    
    # Check if the compatibility layer exists
    compat_path = os.path.join(current_dir, 'autogen_compat', '__init__.py')
    if os.path.exists(compat_path):
        print(f"✓ Compatibility layer found at {compat_path}")
    else:
        print(f"✗ Compatibility layer not found at {compat_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("AUTOGEN COMPATIBILITY LAYER TEST")
    print("=" * 60)
    
    check_environment()
    
    print("\n" + "=" * 60)
    print("TESTING AUTOGEN IMPORT")
    print("=" * 60)
    
    success = test_autogen_import()
    
    print("\n" + "=" * 60)
    print("TEST RESULT: " + ("SUCCESS" if success else "FAILURE"))
    print("=" * 60)
    
    if not success:
        print("\nTo fix this issue:")
        print("1. Make sure ag2 is installed: pip install ag2>=0.2.0")
        print("2. Run the script with the compatibility layer: ./use_autogen_compat.sh python test_autogen_compat.py")
    
    sys.exit(0 if success else 1)