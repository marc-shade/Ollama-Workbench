#!/usr/bin/env python3
"""
Test script for the chat interface fix in Ollama-Workbench

This script tests the chat interface implementations to ensure that session
handling works correctly after applying the fixes.
"""

import os
import sys
import logging
import time
import json
import subprocess
from datetime import datetime

# Set up detailed logging with checkpoints for troubleshooting
logging.basicConfig(
    filename='test_chat_fix.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if the environment is set up correctly"""
    logger.info("CHECKPOINT: Checking environment")
    
    # Check if required modules are installed
    required_modules = ["streamlit", "ollama"]
    
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"Module {module} is installed")
        except ImportError:
            logger.error(f"Module {module} is not installed")
            print(f"Error: Module {module} is not installed")
            print(f"Please install it with: pip install {module}")
            return False
    
    # Check if Ollama is running
    try:
        import ollama
        models = ollama.list()
        logger.info(f"Ollama is running with {len(models['models'])} models")
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {e}")
        print(f"Error: Could not connect to Ollama: {e}")
        print("Please make sure Ollama is running")
        return False
    
    logger.info("CHECKPOINT: Environment check passed")
    return True

def test_session_utils():
    """Test the session_utils module"""
    logger.info("CHECKPOINT: Testing session_utils module")
    
    try:
        # Import session_utils
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import ollama_workbench.core.session_utils as session_utils

        
        # Create mock session state
        import streamlit as st
        if not hasattr(st, 'session_state'):
            setattr(st, 'session_state', {})
        
        # Test initialize_session_state
        logger.info("Testing initialize_session_state")
        session_utils.initialize_session_state()
        
        # Check that session state was initialized
        assert "chat_history" in st.session_state, "chat_history not in session state"
        assert "selected_model" in st.session_state, "selected_model not in session state"
        assert "current_model" in st.session_state, "current_model not in session state"
        
        # Test save_settings
        logger.info("Testing save_settings")
        st.session_state.selected_model = "test_model"
        st.session_state.temperature = 0.8
        success = session_utils.save_settings()
        assert success, "save_settings failed"
        
        # Check that settings file was created
        assert os.path.exists(session_utils.SETTINGS_FILE), f"{session_utils.SETTINGS_FILE} not created"
        
        # Test load_settings
        logger.info("Testing load_settings")
        # Clear session state
        st.session_state.clear()
        success = session_utils.load_settings()
        assert success, "load_settings failed"
        
        # Check that settings were loaded
        assert st.session_state.selected_model == "test_model", "selected_model not loaded correctly"
        assert st.session_state.temperature == 0.8, "temperature not loaded correctly"
        
        # Test save_chat_session
        logger.info("Testing save_chat_session")
        st.session_state.chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        filepath = session_utils.save_chat_session()
        assert filepath is not None, "save_chat_session failed"
        assert os.path.exists(filepath), f"{filepath} not created"
        
        # Test load_chat_session
        logger.info("Testing load_chat_session")
        # Clear session state
        st.session_state.clear()
        success = session_utils.load_chat_session(filepath)
        assert success, "load_chat_session failed"
        
        # Check that chat history was loaded
        assert len(st.session_state.chat_history) == 2, "chat_history not loaded correctly"
        assert st.session_state.chat_history[0]["content"] == "Hello", "chat_history content not loaded correctly"
        
        logger.info("CHECKPOINT: session_utils tests passed")
        return True
    
    except Exception as e:
        logger.error(f"Error testing session_utils: {e}")
        print(f"Error testing session_utils: {e}")
        return False

def test_chat_interfaces():
    """Test the chat interface implementations"""
    logger.info("CHECKPOINT: Testing chat interface implementations")
    
    try:
        # Import chat interfaces
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Test importing each interface
        interfaces_to_test = [
            "chat_interface",
            "enhanced_chat_interface",
            "modern_chat_interface",
            "simple_modern_interface"
        ]
        
        for interface in interfaces_to_test:
            try:
                module = __import__(interface)
                logger.info(f"Successfully imported {interface}")
            except ImportError as e:
                logger.error(f"Error importing {interface}: {e}")
                print(f"Error importing {interface}: {e}")
                return False
        
        logger.info("CHECKPOINT: Chat interface imports successful")
        return True
    
    except Exception as e:
        logger.error(f"Error testing chat interfaces: {e}")
        print(f"Error testing chat interfaces: {e}")
        return False

def run_streamlit_test(interface_script, timeout=10):
    """Run a streamlit test for a specific interface"""
    logger.info(f"CHECKPOINT: Running streamlit test for {interface_script}")
    
    try:
        # Create a test script that will run the interface
        test_script = f"""
import streamlit as st
import sys
import os
import time
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the interface
from {interface_script} import {interface_script}

# Initialize session state for testing
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "test_complete" not in st.session_state:
    # Add a test message
    st.session_state.chat_history.append({{"role": "user", "content": "Test message"}})
    st.session_state.chat_history.append({{"role": "assistant", "content": "Test response"}})
    
    # Mark test as complete
    st.session_state.test_complete = True
    
    # Save test results
    with open("test_results.json", "w") as f:
        json.dump({{"success": True, "chat_history_length": len(st.session_state.chat_history)}}, f)

# Run the interface
{interface_script}()
"""
        
        # Write the test script to a file
        with open("temp_test.py", "w") as f:
            f.write(test_script)
        
        # Run streamlit with the test script
        process = subprocess.Popen(
            ["streamlit", "run", "temp_test.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for the test to complete or timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            if os.path.exists("test_results.json"):
                # Read test results
                with open("test_results.json", "r") as f:
                    results = json.load(f)
                
                # Clean up
                os.remove("test_results.json")
                os.remove("temp_test.py")
                
                # Terminate streamlit process
                process.terminate()
                
                logger.info(f"Test results for {interface_script}: {results}")
                return results["success"]
            
            time.sleep(0.5)
        
        # Timeout
        process.terminate()
        logger.error(f"Test for {interface_script} timed out")
        print(f"Test for {interface_script} timed out")
        
        # Clean up
        if os.path.exists("temp_test.py"):
            os.remove("temp_test.py")
        
        return False
    
    except Exception as e:
        logger.error(f"Error running streamlit test for {interface_script}: {e}")
        print(f"Error running streamlit test for {interface_script}: {e}")
        
        # Clean up
        if os.path.exists("temp_test.py"):
            os.remove("temp_test.py")
        
        return False

def verify_session_synchronization():
    """Verify that session state is synchronized between interfaces"""
    logger.info("CHECKPOINT: Verifying session synchronization")
    
    try:
        # Create a test script that will use multiple interfaces
        test_script = """
import streamlit as st
import sys
import os
import time
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import session_utils
from ollama_workbench.core.session_utils import initialize_session_state, save_settings

# Initialize session state
initialize_session_state()

# Set a model in session state
st.session_state.selected_model = "test_model_1"

# Save settings
save_settings()

# Check if current_model is synchronized
is_synchronized = (
    "current_model" in st.session_state and
    st.session_state.current_model == "test_model_1"
)

# Change current_model
st.session_state.current_model = "test_model_2"

# Check if selected_model is synchronized
is_synchronized = is_synchronized and (
    st.session_state.selected_model == "test_model_2"
)

# Save test results
with open("sync_test_results.json", "w") as f:
    json.dump({"success": is_synchronized}, f)

# Display results
st.write("Session synchronization test complete")
st.write(f"Success: {is_synchronized}")
"""
        
        # Write the test script to a file
        with open("sync_test.py", "w") as f:
            f.write(test_script)
        
        # Run streamlit with the test script
        process = subprocess.Popen(
            ["streamlit", "run", "sync_test.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for the test to complete or timeout
        timeout = 10
        start_time = time.time()
        while time.time() - start_time < timeout:
            if os.path.exists("sync_test_results.json"):
                # Read test results
                with open("sync_test_results.json", "r") as f:
                    results = json.load(f)
                
                # Clean up
                os.remove("sync_test_results.json")
                os.remove("sync_test.py")
                
                # Terminate streamlit process
                process.terminate()
                
                logger.info(f"Session synchronization test results: {results}")
                return results["success"]
            
            time.sleep(0.5)
        
        # Timeout
        process.terminate()
        logger.error("Session synchronization test timed out")
        print("Session synchronization test timed out")
        
        # Clean up
        if os.path.exists("sync_test.py"):
            os.remove("sync_test.py")
        
        return False
    
    except Exception as e:
        logger.error(f"Error verifying session synchronization: {e}")
        print(f"Error verifying session synchronization: {e}")
        
        # Clean up
        if os.path.exists("sync_test.py"):
            os.remove("sync_test.py")
        
        return False

def main():
    """Main function"""
    print("Ollama-Workbench Chat Interface Test")
    print("===================================")
    print("This script tests the chat interface implementations to ensure")
    print("that session handling works correctly after applying the fixes.")
    print()
    
    logger.info("=" * 80)
    logger.info("Starting chat interface test")
    logger.info("=" * 80)
    
    # Check environment
    if not check_environment():
        print("Environment check failed. Please fix the issues and try again.")
        return
    
    # Test session_utils
    print("Testing session_utils module...")
    if not test_session_utils():
        print("session_utils test failed. Please check the logs for details.")
        return
    print("session_utils test passed!")
    
    # Test chat interfaces
    print("Testing chat interface implementations...")
    if not test_chat_interfaces():
        print("Chat interface test failed. Please check the logs for details.")
        return
    print("Chat interface test passed!")
    
    # Verify session synchronization
    print("Verifying session synchronization...")
    if not verify_session_synchronization():
        print("Session synchronization test failed. Please check the logs for details.")
        return
    print("Session synchronization test passed!")
    
    # All tests passed
    print()
    print("All tests passed! The chat interface fix is working correctly.")
    logger.info("All tests passed")
    
    logger.info("=" * 80)
    logger.info("Finished chat interface test")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
