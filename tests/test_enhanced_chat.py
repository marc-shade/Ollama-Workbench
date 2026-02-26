#!/usr/bin/env python
"""
Test script for enhanced_chat_interface.py

This script verifies that the enhanced chat interface can be properly imported
and that it has all the required components to run correctly.
"""

import sys
import logging
import importlib.util

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_enhanced_chat")

def test_import():
    """Test that enhanced_chat_interface can be imported."""
    try:
        import ollama_workbench.chat.enhanced_chat_interface as enhanced_chat_interface

        logger.info("Successfully imported enhanced_chat_interface")
        return True
    except Exception as e:
        logger.error(f"Failed to import enhanced_chat_interface: {str(e)}")
        return False

def test_dependencies():
    """Test that all dependencies for enhanced_chat_interface are available."""
    dependencies = [
        "streamlit", 
        "tiktoken", 
        "numpy", 
        "streamlit_extras",
        "gtts", 
        "groq"
    ]
    
    all_passed = True
    for dep in dependencies:
        try:
            spec = importlib.util.find_spec(dep)
            if spec is None:
                logger.error(f"Dependency {dep} not found")
                all_passed = False
            else:
                logger.info(f"Dependency {dep} is available")
        except ImportError as e:
            logger.error(f"Error checking for {dep}: {str(e)}")
            all_passed = False
    
    return all_passed

def test_original_functions():
    """Test that original chat_interface functions are accessible."""
    try:
        from ollama_workbench.chat.chat_interface import (
            ModelMemoryHandler, EpisodicMemory, extract_content_blocks, 
            calculate_modularity, instance_adaptive_cot
        )
        logger.info("Successfully imported required functions from chat_interface")
        return True
    except Exception as e:
        logger.error(f"Failed to import required functions from chat_interface: {str(e)}")
        return False

def run_all_tests():
    """Run all tests and return overall result."""
    import_result = test_import()
    dependencies_result = test_dependencies()
    functions_result = test_original_functions()
    
    overall_result = import_result and dependencies_result and functions_result
    
    if overall_result:
        logger.info("All tests passed! The enhanced chat interface is ready to use.")
    else:
        logger.error("Some tests failed. Please check the logs for details.")
    
    return overall_result

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)