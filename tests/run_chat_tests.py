#!/usr/bin/env python3
"""
Test runner for Ollama-Workbench chat interface tests

This script runs all the chat interface tests and provides detailed logging
to help identify any issues in the chat interface implementations.
"""

import os
import sys
import unittest
import logging
import time
from datetime import datetime

# Set up logging with timestamps for detailed analysis
logging.basicConfig(
    filename='chat_test_run.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_tests():
    """Run all chat interface tests"""
    logger.info("=" * 80)
    logger.info(f"Starting chat interface tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Create test directory if it doesn't exist
    os.makedirs("test_results", exist_ok=True)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    logger.info("CHECKPOINT: Loading test modules")
    
    try:
        from test_chat_interfaces import TestChatInterfaces, TestChatInterfaceIntegration
        from test_session_handling import TestSessionHandling
        from test_chat_components import TestChatComponents, TestMessageRendering
        
        # Add test cases to suite
        test_suite.addTest(unittest.makeSuite(TestChatInterfaces))
        test_suite.addTest(unittest.makeSuite(TestChatInterfaceIntegration))
        test_suite.addTest(unittest.makeSuite(TestSessionHandling))
        test_suite.addTest(unittest.makeSuite(TestChatComponents))
        test_suite.addTest(unittest.makeSuite(TestMessageRendering))
        
        logger.info("CHECKPOINT: Test modules loaded successfully")
    except ImportError as e:
        logger.error(f"Failed to import test modules: {e}")
        return False
    
    # Run tests
    logger.info("CHECKPOINT: Starting test execution")
    
    # Create test runner with XML output for detailed results
    import xmlrunner
    runner = xmlrunner.XMLTestRunner(output='test_results', verbosity=2)
    
    try:
        # Run tests and get result
        result = runner.run(test_suite)
        
        # Log results
        logger.info(f"Tests run: {result.testsRun}")
        logger.info(f"Errors: {len(result.errors)}")
        logger.info(f"Failures: {len(result.failures)}")
        logger.info(f"Skipped: {len(result.skipped)}")
        
        # Log errors and failures
        if result.errors:
            logger.error("Errors:")
            for test, error in result.errors:
                logger.error(f"{test}: {error}")
        
        if result.failures:
            logger.error("Failures:")
            for test, failure in result.failures:
                logger.error(f"{test}: {failure}")
        
        # Return True if all tests passed
        return len(result.errors) == 0 and len(result.failures) == 0
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False
    finally:
        logger.info("=" * 80)
        logger.info(f"Finished chat interface tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

def check_environment():
    """Check that the environment is set up correctly for testing"""
    logger.info("CHECKPOINT: Checking environment")
    
    # Check that required modules are installed
    required_modules = [
        "unittest", "xmlrunner", "streamlit", "numpy", "tiktoken"
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Missing required modules: {', '.join(missing_modules)}")
        logger.info("Installing missing modules...")
        
        # Install missing modules
        import subprocess
        for module in missing_modules:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", module])
                logger.info(f"Installed {module}")
            except subprocess.CalledProcessError:
                logger.error(f"Failed to install {module}")
                return False
    
    logger.info("CHECKPOINT: Environment check passed")
    return True

def fix_session_handling_issues():
    """
    Analyze and fix common session handling issues in the chat interfaces
    
    This function checks for common issues in the chat interfaces and applies fixes
    to improve session handling.
    """
    logger.info("CHECKPOINT: Analyzing chat interfaces for session handling issues")
    
    issues_found = False
    fixes_applied = False
    
    # Check for session state synchronization issues in main.py
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py"), "r") as f:
            main_content = f.read()
        
        # Check if main.py synchronizes selected_model and current_model
        if "selected_model" in main_content and "current_model" in main_content:
            sync_code = """
    # Make sure selected_model is initialized (for compatibility with modern_chat_interface)
    if "selected_model" in st.session_state and st.session_state.selected_model:
        # If selected_model exists, make sure current_model is synchronized
        if "current_model" not in st.session_state:
            st.session_state.current_model = st.session_state.selected_model
    elif "current_model" in st.session_state and st.session_state.current_model:
        # If current_model exists but selected_model doesn't, synchronize in the other direction
        st.session_state.selected_model = st.session_state.current_model
"""
            if sync_code.strip() not in main_content:
                logger.warning("Session state synchronization code missing in main.py")
                issues_found = True
            else:
                logger.info("Session state synchronization code found in main.py")
        else:
            logger.warning("Could not find session state variables in main.py")
            issues_found = True
    except Exception as e:
        logger.error(f"Error checking main.py: {e}")
        issues_found = True
    
    # Check for session state initialization in modern_chat_interface.py
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "modern_chat_interface.py"), "r") as f:
            modern_content = f.read()
        
        # Check if initialize_session_state is called
        if "initialize_session_state()" not in modern_content:
            logger.warning("initialize_session_state() not called in modern_chat_interface.py")
            issues_found = True
        else:
            logger.info("initialize_session_state() called in modern_chat_interface.py")
    except Exception as e:
        logger.error(f"Error checking modern_chat_interface.py: {e}")
        issues_found = True
    
    # Check for session state initialization in chat_interface.py
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "chat_interface.py"), "r") as f:
            chat_content = f.read()
        
        # Check if chat_history is initialized
        if "if \"chat_history\" not in st.session_state:" not in chat_content:
            logger.warning("chat_history initialization missing in chat_interface.py")
            issues_found = True
        else:
            logger.info("chat_history initialization found in chat_interface.py")
    except Exception as e:
        logger.error(f"Error checking chat_interface.py: {e}")
        issues_found = True
    
    if issues_found:
        logger.warning("Session handling issues found. See log for details.")
    else:
        logger.info("No session handling issues found.")
    
    return not issues_found

if __name__ == "__main__":
    print("Starting Ollama-Workbench chat interface tests...")
    print("Detailed logs will be written to chat_test_run.log")
    
    # Check environment
    if not check_environment():
        print("Environment check failed. See log for details.")
        sys.exit(1)
    
    # Analyze and fix session handling issues
    if not fix_session_handling_issues():
        print("Session handling issues found. See log for details.")
        print("You may need to fix these issues manually.")
    
    # Run tests
    success = run_tests()
    
    if success:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed. See log for details.")
        sys.exit(1)
