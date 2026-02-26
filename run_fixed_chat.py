#!/usr/bin/env python3
"""
Run script for fixed chat interface

This script runs the fixed chat interface with detailed logging to ensure
all features work correctly and to help troubleshoot any issues.
"""

import os
import sys
import logging
import time
from datetime import datetime

# Set up detailed logging with checkpoints for troubleshooting
logging.basicConfig(
    filename='fixed_chat.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if the environment is set up correctly"""
    logger.info("CHECKPOINT: Checking environment")
    print("Checking environment...")
    
    # Check if required modules are installed
    required_modules = ["streamlit", "ollama", "numpy", "tiktoken"]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"Module {module} is installed")
        except ImportError:
            missing_modules.append(module)
            logger.error(f"Module {module} is not installed")
    
    if missing_modules:
        print(f"Missing required modules: {', '.join(missing_modules)}")
        print("Installing missing modules...")
        
        for module in missing_modules:
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", module])
                print(f"Installed {module}")
                logger.info(f"Installed {module}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {module}: {e}")
                logger.error(f"Failed to install {module}: {e}")
                return False
    
    # Check if Ollama is running
    try:
        import ollama
        models = ollama.list()
        logger.info(f"Ollama is running with {len(models['models'])} models")
        print(f"Ollama is running with {len(models['models'])} models")
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {e}")
        print(f"Error: Could not connect to Ollama: {e}")
        print("Please make sure Ollama is running")
        return False
    
    logger.info("CHECKPOINT: Environment check passed")
    print("Environment check passed!")
    return True

def run_fixed_chat():
    """Run the fixed chat interface"""
    logger.info("CHECKPOINT: Running fixed chat interface")
    print("Running fixed chat interface...")
    
    try:
        import subprocess
        
        # Run streamlit with the fixed chat interface
        cmd = [sys.executable, "-m", "streamlit", "run", "fixed_chat_interface.py"]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Log stdout and stderr in real-time
        for line in process.stdout:
            logger.info(f"STDOUT: {line.strip()}")
            print(line.strip())
        
        # Check for errors
        for line in process.stderr:
            logger.error(f"STDERR: {line.strip()}")
            print(f"ERROR: {line.strip()}")
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Process exited with code {process.returncode}")
            print(f"Process exited with code {process.returncode}")
            return False
        
        logger.info("CHECKPOINT: Fixed chat interface ran successfully")
        print("Fixed chat interface ran successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error running fixed chat interface: {e}")
        print(f"Error running fixed chat interface: {e}")
        return False

def main():
    """Main function"""
    print("Ollama-Workbench Fixed Chat Interface")
    print("====================================")
    print("This script runs the fixed chat interface with detailed logging")
    print("to ensure all features work correctly and to help troubleshoot any issues.")
    print()
    
    logger.info("=" * 80)
    logger.info("Starting fixed chat interface")
    logger.info("=" * 80)
    
    # Check environment
    if not check_environment():
        print("Environment check failed. Please fix the issues and try again.")
        return
    
    # Run fixed chat interface
    run_fixed_chat()
    
    logger.info("=" * 80)
    logger.info("Finished fixed chat interface")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
