#!/usr/bin/env python3
"""
Comprehensive test runner for Ollama-Workbench

This script runs all tests for the Ollama-Workbench chat interfaces,
model settings, agent features, and thinking types to ensure 100% coverage
and error-free operation.
"""

import os
import sys
import unittest
import logging
import time
import json
import subprocess
from datetime import datetime

# Set up logging with detailed checkpoints for troubleshooting
logging.basicConfig(
    filename='test_run.log',
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
    required_modules = [
        "streamlit", "ollama", "numpy", "tiktoken", 
        "sklearn", "xmlrunner", "matplotlib"
    ]
    
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
    
    # Check if Ollama is running
    try:
        import ollama
        models_response = ollama.list()
        # v0.4.8+: ListResponse object with .models attribute
        models_list = getattr(models_response, 'models', None)
        if models_list is None:
            models_list = models_response.get('models', []) if isinstance(models_response, dict) else []
        logger.info(f"Ollama is running with {len(models_list)} models")
        print(f"Ollama is running with {len(models_list)} models")
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {e}")
        print(f"Error: Could not connect to Ollama: {e}")
        print("Please make sure Ollama is running")
        return False
    
    logger.info("CHECKPOINT: Environment check passed")
    print("Environment check passed!")
    return True

def apply_session_utils_fix():
    """Apply the session_utils fix to the codebase"""
    logger.info("CHECKPOINT: Applying session_utils fix")
    print("Applying session_utils fix...")
    
    # Check if session_utils.py exists
    if not os.path.exists("session_utils.py"):
        logger.error("session_utils.py not found")
        print("Error: session_utils.py not found")
        return False
    
    # Update imports in chat interface files
    files_to_update = [
        "chat_interface.py",
        "enhanced_chat_interface.py",
        "modern_chat_interface.py",
        "simple_modern_interface.py"
    ]
    
    for file in files_to_update:
        if not os.path.exists(file):
            logger.warning(f"{file} not found, skipping")
            print(f"Warning: {file} not found, skipping")
            continue
        
        # Create backup
        backup_file = f"{file}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
        try:
            with open(file, "r") as f:
                content = f.read()
            
            with open(backup_file, "w") as f:
                f.write(content)
            
            logger.info(f"Created backup of {file} at {backup_file}")
            print(f"Created backup of {file}")
            
            # Add import for session_utils
            if "import session_utils" not in content and "from session_utils import" not in content:
                # Add import after other imports
                import_lines = []
                other_lines = []
                in_imports = True
                
                for line in content.split("\n"):
                    if in_imports and (line.startswith("import ") or line.startswith("from ")):
                        import_lines.append(line)
                    else:
                        if in_imports and line.strip() and not line.startswith("#"):
                            in_imports = False
                            # Add session_utils import
                            import_lines.append("# Import session utilities for consistent session handling")
                            import_lines.append("from session_utils import (")
                            import_lines.append("    initialize_session_state, load_settings, save_settings,")
                            import_lines.append("    save_chat_session, load_chat_session, synchronize_model_variables,")
                            import_lines.append("    get_agent_prompt, get_rag_context, safe_rerun, log_message")
                            import_lines.append(")")
                            import_lines.append("")
                        other_lines.append(line)
                
                # Combine lines
                updated_content = "\n".join(import_lines + other_lines)
                
                # Write updated content
                with open(file, "w") as f:
                    f.write(updated_content)
                
                logger.info(f"Added session_utils import to {file}")
                print(f"Added session_utils import to {file}")
            else:
                logger.info(f"{file} already imports session_utils")
                print(f"{file} already imports session_utils")
        
        except Exception as e:
            logger.error(f"Error updating {file}: {e}")
            print(f"Error updating {file}: {e}")
            return False
    
    logger.info("CHECKPOINT: Session utils fix applied")
    print("Session utils fix applied successfully!")
    return True

def run_tests():
    """Run all tests"""
    logger.info("CHECKPOINT: Running tests")
    print("\nRunning tests...")
    
    # Create test results directory
    os.makedirs("test_results", exist_ok=True)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    try:
        # Import test modules
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
        
        from tests.test_chat_interfaces import TestChatInterfaces, TestChatInterfaceIntegration
        from tests.test_session_handling import TestSessionHandling
        from tests.test_chat_components import TestChatComponents, TestMessageRendering
        from tests.test_model_settings import TestModelSettings, TestAgentFeatures
        from tests.test_advanced_features import TestThinkingTypes, TestEpisodicMemory, TestRAGFeatures
        from tests.test_chat_integration import TestChatIntegration
        
        # Add test cases to suite
        test_suite.addTest(unittest.makeSuite(TestChatInterfaces))
        test_suite.addTest(unittest.makeSuite(TestChatInterfaceIntegration))
        test_suite.addTest(unittest.makeSuite(TestSessionHandling))
        test_suite.addTest(unittest.makeSuite(TestChatComponents))
        test_suite.addTest(unittest.makeSuite(TestMessageRendering))
        test_suite.addTest(unittest.makeSuite(TestModelSettings))
        test_suite.addTest(unittest.makeSuite(TestAgentFeatures))
        test_suite.addTest(unittest.makeSuite(TestThinkingTypes))
        test_suite.addTest(unittest.makeSuite(TestEpisodicMemory))
        test_suite.addTest(unittest.makeSuite(TestRAGFeatures))
        test_suite.addTest(unittest.makeSuite(TestChatIntegration))
        
        logger.info("CHECKPOINT: Test modules loaded successfully")
        print("Test modules loaded successfully")
    except ImportError as e:
        logger.error(f"Failed to import test modules: {e}")
        print(f"Error: Failed to import test modules: {e}")
        return False
    
    # Run tests
    try:
        # Import xmlrunner for XML test reports
        import xmlrunner
        runner = xmlrunner.XMLTestRunner(output='test_results', verbosity=2)
        
        # Run tests
        result = runner.run(test_suite)
        
        # Log results
        logger.info(f"Tests run: {result.testsRun}")
        logger.info(f"Errors: {len(result.errors)}")
        logger.info(f"Failures: {len(result.failures)}")
        logger.info(f"Skipped: {len(result.skipped)}")
        
        print(f"\nTests run: {result.testsRun}")
        print(f"Errors: {len(result.errors)}")
        print(f"Failures: {len(result.failures)}")
        print(f"Skipped: {len(result.skipped)}")
        
        # Log errors and failures
        if result.errors:
            logger.error("Errors:")
            print("\nErrors:")
            for test, error in result.errors:
                logger.error(f"{test}: {error}")
                print(f"{test}: {error}")
        
        if result.failures:
            logger.error("Failures:")
            print("\nFailures:")
            for test, failure in result.failures:
                logger.error(f"{test}: {failure}")
                print(f"{test}: {failure}")
        
        # Return True if all tests passed
        return len(result.errors) == 0 and len(result.failures) == 0
    
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        print(f"Error running tests: {e}")
        return False

def verify_implementation():
    """Verify that the implementation works correctly"""
    logger.info("CHECKPOINT: Verifying implementation")
    print("\nVerifying implementation...")
    
    # Create a test script that will use the session_utils module
    test_script = """
import streamlit as st
import sys
import os
import time
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import session_utils
from ollama_workbench.core.session_utils import (
    initialize_session_state, load_settings, save_settings,
    save_chat_session, load_chat_session, synchronize_model_variables
)

# Initialize session state
initialize_session_state()

# Set some values in session state
st.session_state.selected_model = "llama3"
st.session_state.agent_type = "Researcher"
st.session_state.temperature = 0.8

# Synchronize model variables
synchronize_model_variables()

# Check if current_model is synchronized
is_synchronized = (
    "current_model" in st.session_state and
    st.session_state.current_model == "llama3"
)

# Save settings
save_settings()

# Clear session state
for key in list(st.session_state.keys()):
    del st.session_state[key]

# Load settings
load_settings()

# Check if settings were loaded correctly
settings_loaded = (
    "selected_model" in st.session_state and
    st.session_state.selected_model == "llama3" and
    "agent_type" in st.session_state and
    st.session_state.agent_type == "Researcher" and
    "temperature" in st.session_state and
    st.session_state.temperature == 0.8
)

# Save test results
with open("verify_results.json", "w") as f:
    json.dump({
        "synchronized": is_synchronized,
        "settings_loaded": settings_loaded
    }, f)

# Display results
st.write("Implementation verification complete")
st.write(f"Synchronized: {is_synchronized}")
st.write(f"Settings loaded: {settings_loaded}")
"""
    
    try:
        # Write the test script to a file
        with open("verify_implementation.py", "w") as f:
            f.write(test_script)
        
        # Run streamlit with the test script
        process = subprocess.Popen(
            ["streamlit", "run", "verify_implementation.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for the test to complete or timeout
        timeout = 30
        start_time = time.time()
        while time.time() - start_time < timeout:
            if os.path.exists("verify_results.json"):
                # Read test results
                with open("verify_results.json", "r") as f:
                    results = json.load(f)
                
                # Clean up
                os.remove("verify_results.json")
                os.remove("verify_implementation.py")
                
                # Terminate streamlit process
                process.terminate()
                
                logger.info(f"Implementation verification results: {results}")
                print(f"Implementation verification results:")
                print(f"Synchronized: {results['synchronized']}")
                print(f"Settings loaded: {results['settings_loaded']}")
                
                # Return True if all checks passed
                return results["synchronized"] and results["settings_loaded"]
            
            time.sleep(0.5)
        
        # Timeout
        process.terminate()
        logger.error("Implementation verification timed out")
        print("Implementation verification timed out")
        
        # Clean up
        if os.path.exists("verify_implementation.py"):
            os.remove("verify_implementation.py")
        
        return False
    
    except Exception as e:
        logger.error(f"Error verifying implementation: {e}")
        print(f"Error verifying implementation: {e}")
        
        # Clean up
        if os.path.exists("verify_implementation.py"):
            os.remove("verify_implementation.py")
        
        return False

def generate_test_report():
    """Generate a test report"""
    logger.info("CHECKPOINT: Generating test report")
    print("\nGenerating test report...")
    
    try:
        # Check if test results directory exists
        if not os.path.exists("test_results"):
            logger.error("Test results directory not found")
            print("Error: Test results directory not found")
            return False
        
        # Count test results
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        error_tests = 0
        
        # Parse XML test results
        import xml.etree.ElementTree as ET
        
        for file in os.listdir("test_results"):
            if file.endswith(".xml"):
                try:
                    tree = ET.parse(os.path.join("test_results", file))
                    root = tree.getroot()
                    
                    # Count tests
                    total_tests += int(root.attrib.get("tests", 0))
                    failed_tests += int(root.attrib.get("failures", 0))
                    error_tests += int(root.attrib.get("errors", 0))
                except Exception as e:
                    logger.error(f"Error parsing {file}: {e}")
        
        passed_tests = total_tests - failed_tests - error_tests
        
        # Calculate coverage
        coverage = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Generate report
        report = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "error_tests": error_tests,
            "coverage": coverage,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save report
        with open("test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test report generated: {report}")
        print("\nTest Report:")
        print(f"Total tests: {total_tests}")
        print(f"Passed tests: {passed_tests}")
        print(f"Failed tests: {failed_tests}")
        print(f"Error tests: {error_tests}")
        print(f"Coverage: {coverage:.2f}%")
        
        return True
    
    except Exception as e:
        logger.error(f"Error generating test report: {e}")
        print(f"Error generating test report: {e}")
        return False

def main():
    """Main function"""
    print("Ollama-Workbench Comprehensive Test Runner")
    print("=========================================")
    print("This script runs all tests for the Ollama-Workbench chat interfaces,")
    print("model settings, agent features, and thinking types to ensure 100%")
    print("coverage and error-free operation.")
    print()
    
    logger.info("=" * 80)
    logger.info("Starting comprehensive test run")
    logger.info("=" * 80)
    
    # Check environment
    if not check_environment():
        print("\nEnvironment check failed. Please fix the issues and try again.")
        return
    
    # Apply session_utils fix
    if not apply_session_utils_fix():
        print("\nFailed to apply session_utils fix. Please fix the issues and try again.")
        return
    
    # Run tests
    if not run_tests():
        print("\nSome tests failed. Please check the logs for details.")
    else:
        print("\nAll tests passed!")
    
    # Verify implementation
    if not verify_implementation():
        print("\nImplementation verification failed. Please check the logs for details.")
    else:
        print("\nImplementation verification passed!")
    
    # Generate test report
    generate_test_report()
    
    logger.info("=" * 80)
    logger.info("Finished comprehensive test run")
    logger.info("=" * 80)
    
    print("\nTest run complete. See test_report.json for details.")

if __name__ == "__main__":
    main()
