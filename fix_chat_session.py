#!/usr/bin/env python3
"""
Fix for chat session handling in Ollama-Workbench

This script analyzes and fixes session handling issues in the chat interfaces,
ensuring that session state is properly maintained between different implementations.
"""

import os
import sys
import re
import logging
import shutil
from datetime import datetime

# Set up logging with detailed information for troubleshooting
logging.basicConfig(
    filename='fix_chat_session.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def create_backup(file_path):
    """Create a backup of a file before modifying it"""
    logger.info(f"Creating backup of {file_path}")
    backup_path = f"{file_path}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        shutil.copy2(file_path, backup_path)
        logger.info(f"Backup created at {backup_path}")
        print(f"Created backup: {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create backup of {file_path}: {e}")
        print(f"Error creating backup: {e}")
        return False

def fix_modern_chat_interface():
    """Fix session handling in modern_chat_interface.py"""
    file_path = "modern_chat_interface.py"
    logger.info(f"Fixing session handling in {file_path}")
    print(f"Analyzing {file_path}...")
    
    # Create backup
    if not create_backup(file_path):
        return False
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        # Check if initialize_session_state needs to be updated
        init_function = re.search(r"def initialize_session_state\(\):(.*?)def", content, re.DOTALL)
        if init_function:
            init_code = init_function.group(1)
            
            # Check if session state synchronization is missing
            if "current_model" in init_code and "selected_model" not in init_code:
                logger.info("Adding session state synchronization to initialize_session_state")
                
                # Add synchronization code
                new_init_code = init_code.replace(
                    'if "current_model" not in st.session_state:',
                    '# Ensure compatibility with other interfaces\n'
                    '    if "selected_model" in st.session_state and st.session_state.selected_model:\n'
                    '        # If selected_model exists, use it for current_model\n'
                    '        st.session_state.current_model = st.session_state.selected_model\n'
                    '    elif "current_model" not in st.session_state:'
                )
                
                # Update content
                content = content.replace(init_code, new_init_code)
                logger.info("Session state synchronization added")
                print("Added session state synchronization to initialize_session_state")
        
        # Check if save_chat_session needs to be updated
        save_function = re.search(r"def save_chat_session\(\):(.*?)def", content, re.DOTALL)
        if save_function:
            save_code = save_function.group(1)
            
            # Check if model synchronization is missing
            if "current_model" in save_code and "selected_model" not in save_code:
                logger.info("Adding model synchronization to save_chat_session")
                
                # Add synchronization code
                new_save_code = save_code.replace(
                    'session_data = {',
                    '# Ensure both model variables are synchronized\n'
                    '    if "selected_model" in st.session_state:\n'
                    '        st.session_state.current_model = st.session_state.selected_model\n'
                    '    \n'
                    '    session_data = {'
                )
                
                # Update content
                content = content.replace(save_code, new_save_code)
                logger.info("Model synchronization added to save_chat_session")
                print("Added model synchronization to save_chat_session")
        
        # Write updated content
        with open(file_path, "w") as f:
            f.write(content)
        
        logger.info(f"Successfully updated {file_path}")
        print(f"Successfully updated {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        print(f"Error fixing {file_path}: {e}")
        return False

def fix_chat_interface():
    """Fix session handling in chat_interface.py"""
    file_path = "chat_interface.py"
    logger.info(f"Fixing session handling in {file_path}")
    print(f"Analyzing {file_path}...")
    
    # Create backup
    if not create_backup(file_path):
        return False
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        # Check if chat_interface function needs to be updated
        chat_function = re.search(r"def chat_interface\(\):(.*?)if __name__ ==", content, re.DOTALL)
        if chat_function:
            chat_code = chat_function.group(1)
            
            # Check if session state initialization is missing or incomplete
            if "if \"chat_history\" not in st.session_state:" in chat_code:
                # Check if model synchronization is missing
                if "selected_model" in chat_code and "current_model" in chat_code:
                    if "st.session_state.current_model = st.session_state.selected_model" not in chat_code:
                        logger.info("Adding model synchronization to chat_interface")
                        
                        # Add synchronization code after chat_history initialization
                        new_chat_code = chat_code.replace(
                            'if "chat_history" not in st.session_state:\n        st.session_state.chat_history = []',
                            'if "chat_history" not in st.session_state:\n'
                            '        st.session_state.chat_history = []\n'
                            '    \n'
                            '    # Ensure compatibility with other interfaces\n'
                            '    if "selected_model" in st.session_state and "current_model" not in st.session_state:\n'
                            '        st.session_state.current_model = st.session_state.selected_model\n'
                            '    elif "current_model" in st.session_state and "selected_model" not in st.session_state:\n'
                            '        st.session_state.selected_model = st.session_state.current_model'
                        )
                        
                        # Update content
                        content = content.replace(chat_code, new_chat_code)
                        logger.info("Model synchronization added to chat_interface")
                        print("Added model synchronization to chat_interface")
            
            # Check if rerun handling needs improvement
            if "st.rerun()" in chat_code:
                # Add try-except block around rerun to handle exceptions
                if "try:\n            st.rerun()" not in chat_code:
                    logger.info("Improving rerun handling in chat_interface")
                    
                    # Add try-except block
                    new_chat_code = chat_code.replace(
                        'st.rerun()',
                        'try:\n                st.rerun()\n'
                        '            except Exception as e:\n'
                        '                logger.error(f"Error during rerun: {e}")\n'
                        '                # Continue without rerun'
                    )
                    
                    # Update content
                    content = content.replace(chat_code, new_chat_code)
                    logger.info("Improved rerun handling in chat_interface")
                    print("Improved rerun handling in chat_interface")
        
        # Write updated content
        with open(file_path, "w") as f:
            f.write(content)
        
        logger.info(f"Successfully updated {file_path}")
        print(f"Successfully updated {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        print(f"Error fixing {file_path}: {e}")
        return False

def fix_simple_modern_interface():
    """Fix session handling in simple_modern_interface.py"""
    file_path = "simple_modern_interface.py"
    logger.info(f"Fixing session handling in {file_path}")
    print(f"Analyzing {file_path}...")
    
    # Create backup
    if not create_backup(file_path):
        return False
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        # Check if simple_modern_interface function needs to be updated
        interface_function = re.search(r"def simple_modern_interface\(\):(.*?)if __name__ ==", content, re.DOTALL)
        if interface_function:
            interface_code = interface_function.group(1)
            
            # Check if model synchronization is missing
            if "selected_model" in interface_code and "current_model" not in interface_code:
                logger.info("Adding model synchronization to simple_modern_interface")
                
                # Add synchronization code after session state initialization
                new_interface_code = interface_code.replace(
                    'if "selected_model" not in st.session_state or not st.session_state.selected_model:',
                    '# Ensure compatibility with other interfaces\n'
                    '    if "current_model" in st.session_state:\n'
                    '        st.session_state.selected_model = st.session_state.current_model\n'
                    '    \n'
                    '    if "selected_model" not in st.session_state or not st.session_state.selected_model:'
                )
                
                # Update content
                content = content.replace(interface_code, new_interface_code)
                logger.info("Model synchronization added to simple_modern_interface")
                print("Added model synchronization to simple_modern_interface")
        
        # Write updated content
        with open(file_path, "w") as f:
            f.write(content)
        
        logger.info(f"Successfully updated {file_path}")
        print(f"Successfully updated {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        print(f"Error fixing {file_path}: {e}")
        return False

def fix_enhanced_chat_interface():
    """Fix session handling in enhanced_chat_interface.py"""
    file_path = "enhanced_chat_interface.py"
    logger.info(f"Fixing session handling in {file_path}")
    print(f"Analyzing {file_path}...")
    
    # Create backup
    if not create_backup(file_path):
        return False
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        # Check if enhanced_chat_interface function needs to be updated
        enhanced_function = re.search(r"def enhanced_chat_interface\(\):(.*?)if __name__ ==", content, re.DOTALL)
        if enhanced_function:
            enhanced_code = enhanced_function.group(1)
            
            # Check if session state initialization is missing or incomplete
            if "enhanced_mode" in enhanced_code:
                # Check if additional session state initialization is needed
                if "if \"chat_history\" not in st.session_state:" not in enhanced_code:
                    logger.info("Adding session state initialization to enhanced_chat_interface")
                    
                    # Add initialization code
                    new_enhanced_code = enhanced_code.replace(
                        'if "enhanced_mode" not in st.session_state:',
                        '# Initialize essential session state variables\n'
                        '    if "chat_history" not in st.session_state:\n'
                        '        st.session_state.chat_history = []\n'
                        '    \n'
                        '    # Ensure compatibility with other interfaces\n'
                        '    if "selected_model" in st.session_state and "current_model" not in st.session_state:\n'
                        '        st.session_state.current_model = st.session_state.selected_model\n'
                        '    elif "current_model" in st.session_state and "selected_model" not in st.session_state:\n'
                        '        st.session_state.selected_model = st.session_state.current_model\n'
                        '    \n'
                        '    if "enhanced_mode" not in st.session_state:'
                    )
                    
                    # Update content
                    content = content.replace(enhanced_code, new_enhanced_code)
                    logger.info("Session state initialization added to enhanced_chat_interface")
                    print("Added session state initialization to enhanced_chat_interface")
        
        # Write updated content
        with open(file_path, "w") as f:
            f.write(content)
        
        logger.info(f"Successfully updated {file_path}")
        print(f"Successfully updated {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        print(f"Error fixing {file_path}: {e}")
        return False

def create_session_utils():
    """Create a utility module for session handling"""
    file_path = "session_utils.py"
    logger.info(f"Creating {file_path}")
    print(f"Creating {file_path}...")
    
    try:
        # Check if file already exists
        if os.path.exists(file_path):
            logger.info(f"{file_path} already exists, creating backup")
            if not create_backup(file_path):
                return False
        
        # Create session utils module
        content = """#!/usr/bin/env python3
\"\"\"
Session utilities for Ollama-Workbench

This module provides utilities for managing session state across different
chat interface implementations, ensuring consistent behavior.
\"\"\"

import streamlit as st
import os
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
SESSIONS_FOLDER = "sessions"
SETTINGS_FILE = "chat-settings.json"

# Ensure sessions folder exists
if not os.path.exists(SESSIONS_FOLDER):
    os.makedirs(SESSIONS_FOLDER)

def initialize_session_state():
    \"\"\"
    Initialize session state variables consistently across all interfaces.
    
    This function ensures that all necessary session state variables are
    initialized with appropriate default values, and that variables used
    by different interfaces are properly synchronized.
    \"\"\"
    logger.info("CHECKPOINT: Initializing session state")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        logger.info("Initialized chat_history")
    
    # Initialize model selection (ensuring compatibility between interfaces)
    try:
        from ollama_utils import get_available_models
        available_models = get_available_models()
        default_model = available_models[0] if available_models else "llama2"
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        default_model = "llama2"
    
    # Synchronize model selection variables
    if "selected_model" in st.session_state and "current_model" not in st.session_state:
        # If selected_model exists, use it for current_model
        st.session_state.current_model = st.session_state.selected_model
        logger.info(f"Synchronized current_model from selected_model: {st.session_state.selected_model}")
    elif "current_model" in st.session_state and "selected_model" not in st.session_state:
        # If current_model exists, use it for selected_model
        st.session_state.selected_model = st.session_state.current_model
        logger.info(f"Synchronized selected_model from current_model: {st.session_state.current_model}")
    elif "selected_model" not in st.session_state and "current_model" not in st.session_state:
        # If neither exists, initialize both
        st.session_state.selected_model = default_model
        st.session_state.current_model = default_model
        logger.info(f"Initialized both model variables to: {default_model}")
    
    # Initialize other common settings
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 4000
    
    if "presence_penalty" not in st.session_state:
        st.session_state.presence_penalty = 0.0
    
    if "frequency_penalty" not in st.session_state:
        st.session_state.frequency_penalty = 0.0
    
    # Initialize agent settings
    if "agent_type" not in st.session_state:
        st.session_state.agent_type = "None"
    
    if "metacognitive_type" not in st.session_state:
        st.session_state.metacognitive_type = "None"
    
    if "voice_type" not in st.session_state:
        st.session_state.voice_type = "None"
    
    if "selected_corpus" not in st.session_state:
        st.session_state.selected_corpus = "None"
    
    logger.info("CHECKPOINT: Session state initialization complete")

def save_chat_session():
    \"\"\"
    Save the current chat session to a file.
    
    This function saves the chat history and model selection to a file
    in the sessions folder, with a timestamp in the filename.
    \"\"\"
    logger.info("CHECKPOINT: Saving chat session")
    
    # Ensure session state is initialized
    if "chat_history" not in st.session_state:
        logger.warning("Cannot save session: chat_history not in session state")
        return
    
    # Ensure model variables are synchronized
    if "selected_model" in st.session_state and "current_model" in st.session_state:
        if st.session_state.selected_model != st.session_state.current_model:
            logger.info(f"Synchronizing model variables: {st.session_state.selected_model} -> {st.session_state.current_model}")
            st.session_state.current_model = st.session_state.selected_model
    
    # Create session data
    session_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chat_history": st.session_state.chat_history,
        "model": st.session_state.selected_model if "selected_model" in st.session_state else st.session_state.current_model
    }
    
    # Create filename with timestamp
    filename = f"chat_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(SESSIONS_FOLDER, filename)
    
    try:
        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)
        logger.info(f"Chat session saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving chat session: {e}")
        return None

def load_chat_session(file_path):
    \"\"\"
    Load a chat session from a file.
    
    This function loads the chat history and model selection from a file
    and updates the session state accordingly.
    
    Args:
        file_path: Path to the session file to load
    
    Returns:
        bool: True if session was loaded successfully, False otherwise
    \"\"\"
    logger.info(f"CHECKPOINT: Loading chat session from {file_path}")
    
    try:
        with open(file_path, "r") as f:
            session_data = json.load(f)
        
        # Update session state
        st.session_state.chat_history = session_data["chat_history"]
        
        # Update model selection (ensuring compatibility between interfaces)
        if "model" in session_data:
            st.session_state.selected_model = session_data["model"]
            st.session_state.current_model = session_data["model"]
        
        logger.info(f"Chat session loaded from {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading chat session: {e}")
        return False

def load_settings():
    \"\"\"
    Load settings from the settings file.
    
    This function loads settings from the settings file and updates
    the session state accordingly.
    \"\"\"
    logger.info("CHECKPOINT: Loading settings")
    
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
            
            # Update session state with settings
            for key, value in settings.items():
                st.session_state[key] = value
            
            # Ensure model variables are synchronized
            if "selected_model" in st.session_state:
                st.session_state.current_model = st.session_state.selected_model
            
            logger.info(f"Settings loaded from {SETTINGS_FILE}")
            return True
        else:
            logger.warning(f"Settings file {SETTINGS_FILE} not found")
            return False
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        return False

def save_settings():
    \"\"\"
    Save settings to the settings file.
    
    This function saves the current settings from the session state
    to the settings file.
    \"\"\"
    logger.info("CHECKPOINT: Saving settings")
    
    # Collect settings from session state
    settings = {}
    
    # Model selection
    if "selected_model" in st.session_state:
        settings["selected_model"] = st.session_state.selected_model
    elif "current_model" in st.session_state:
        settings["selected_model"] = st.session_state.current_model
    
    # Agent settings
    if "agent_type" in st.session_state:
        settings["agent_type"] = st.session_state.agent_type
    
    if "metacognitive_type" in st.session_state:
        settings["metacognitive_type"] = st.session_state.metacognitive_type
    
    if "voice_type" in st.session_state:
        settings["voice_type"] = st.session_state.voice_type
    
    # Generation settings
    if "temperature" in st.session_state:
        settings["temperature"] = st.session_state.temperature
    
    if "max_tokens" in st.session_state:
        settings["max_tokens"] = st.session_state.max_tokens
    
    if "presence_penalty" in st.session_state:
        settings["presence_penalty"] = st.session_state.presence_penalty
    
    if "frequency_penalty" in st.session_state:
        settings["frequency_penalty"] = st.session_state.frequency_penalty
    
    # Corpus settings
    if "selected_corpus" in st.session_state:
        settings["selected_corpus"] = st.session_state.selected_corpus
    
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
        logger.info(f"Settings saved to {SETTINGS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return False
"""
        
        # Write the file
        with open(file_path, "w") as f:
            f.write(content)
        
        logger.info(f"Successfully created {file_path}")
        print(f"Successfully created {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating {file_path}: {e}")
        print(f"Error creating {file_path}: {e}")
        return False

def create_test_runner():
    """Create a script to run the chat interface tests"""
    file_path = "run_chat_tests.sh"
    logger.info(f"Creating {file_path}")
    print(f"Creating {file_path}...")
    
    try:
        # Create test runner script
        content = """#!/bin/bash

# Run chat interface tests for Ollama-Workbench
echo "Running chat interface tests for Ollama-Workbench"
echo "================================================"

# Check if xmlrunner is installed
if ! python -c "import xmlrunner" &> /dev/null; then
    echo "Installing xmlrunner..."
    pip install unittest-xml-reporting
fi

# Run the tests
echo "Running tests..."
python tests/run_chat_tests.py

# Check the result
if [ $? -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed. See logs for details."
    exit 1
fi
"""
        
        # Write the file
        with open(file_path, "w") as f:
            f.write(content)
        
        # Make the file executable
        os.chmod(file_path, 0o755)
        
        logger.info(f"Successfully created {file_path}")
        print(f"Successfully created {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating {file_path}: {e}")
        print(f"Error creating {file_path}: {e}")
        return False

def main():
    """Main function"""
    print("Ollama-Workbench Chat Session Fix")
    print("================================")
    print("This script will fix session handling issues in the chat interfaces.")
    print("Backups will be created before making any changes.")
    print()
    
    logger.info("=" * 80)
    logger.info("Starting chat session fix")
    logger.info("=" * 80)
    
    # Fix chat interfaces
    success = True
    success = fix_modern_chat_interface() and success
    success = fix_chat_interface() and success
    success = fix_simple_modern_interface() and success
    success = fix_enhanced_chat_interface() and success
    
    # Create session utils module
    success = create_session_utils() and success
    
    # Create test runner
    success = create_test_runner() and success
    
    if success:
        print()
        print("All fixes applied successfully!")
        print("To run the tests, execute: ./run_chat_tests.sh")
        logger.info("All fixes applied successfully")
    else:
        print()
        print("Some fixes could not be applied. See log for details.")
        logger.error("Some fixes could not be applied")
    
    logger.info("=" * 80)
    logger.info("Finished chat session fix")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
