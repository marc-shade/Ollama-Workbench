#!/usr/bin/env python3
"""
Launch script for the fixed chat interface

This script launches the fixed chat interface with detailed logging and
environment checks to ensure all features work correctly.
"""

import os
import sys
import logging
import subprocess
from datetime import datetime

# Set up detailed logging with checkpoints for troubleshooting
logging.basicConfig(
    filename='launch_fixed_chat.log',
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
                subprocess.check_call([sys.executable, "-m", "pip", "install", module])
                print(f"Installed {module}")
                logger.info(f"Installed {module}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {module}: {e}")
                logger.error(f"Failed to install {module}: {e}")
                return False
    
    # Check if required directories exist
    required_dirs = ["sessions"]
    for directory in required_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created directory: {directory}")
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                print(f"Failed to create directory {directory}: {e}")
                logger.error(f"Failed to create directory {directory}: {e}")
                return False
    
    # Check if Ollama is running
    try:
        import ollama
        models = ollama.list()
        logger.info(f"CHECKPOINT: Ollama is running with {len(models['models'])} models")
        print(f"Ollama is running with {len(models['models'])} models")
    except Exception as e:
        logger.error(f"CHECKPOINT: Error connecting to Ollama: {e}")
        print(f"Error: Could not connect to Ollama: {e}")
        print("Please make sure Ollama is running")
        return False
    
    logger.info("CHECKPOINT: Environment check passed")
    print("Environment check passed!")
    return True

def launch_fixed_chat():
    """Launch the fixed chat interface"""
    logger.info("CHECKPOINT: Launching fixed chat interface")
    print("Launching fixed chat interface...")
    
    try:
        # Run streamlit with the fixed chat interface
        cmd = [sys.executable, "-m", "streamlit", "run", "fixed_chat_interface.py"]
        
        logger.info(f"CHECKPOINT: Running command: {' '.join(cmd)}")
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the command
        process = subprocess.Popen(cmd)
        
        logger.info("CHECKPOINT: Fixed chat interface launched")
        print("Fixed chat interface launched!")
        
        # Wait for the process to complete
        process.wait()
        
        return True
    
    except Exception as e:
        logger.error(f"CHECKPOINT: Error launching fixed chat interface: {e}")
        print(f"Error launching fixed chat interface: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("Ollama-Workbench Fixed Chat Interface")
    print("=" * 60)
    print("This script launches the fixed chat interface with all features")
    print("including model settings, agent features, and advanced functionalities.")
    print()
    
    logger.info("=" * 80)
    logger.info("CHECKPOINT: Starting fixed chat interface")
    logger.info("=" * 80)
    
    # Check environment
    if not check_environment():
        print("Environment check failed. Please fix the issues and try again.")
        return
    
    # Launch fixed chat interface
    launch_fixed_chat()
    
    logger.info("=" * 80)
    logger.info("CHECKPOINT: Finished fixed chat interface")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
