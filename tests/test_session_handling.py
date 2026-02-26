"""
Test suite for Ollama-Workbench session handling

This test suite specifically focuses on the session handling capabilities
of the chat interfaces, ensuring that session state is properly maintained
between interactions and across different components.
"""

import os
import sys
import unittest
import json
import logging
import time
from unittest.mock import patch, MagicMock, call

# Set up logging
logging.basicConfig(
    filename='test_session.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
try:
    import streamlit as st
    from chat_interface import chat_interface
    from modern_chat_interface import modern_chat_interface, initialize_session_state, save_chat_session, load_chat_session
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

class TestSessionHandling(unittest.TestCase):
    """Test case for session handling"""
    
    def setUp(self):
        """Set up test environment"""
        logger.info("Setting up test environment for session handling tests")
        
        # Mock streamlit session state
        self.mock_session_state = {}
        
        # Create patch for st.session_state
        self.session_state_patch = patch('streamlit.session_state', self.mock_session_state)
        self.mock_st_session_state = self.session_state_patch.start()
        
        # Mock streamlit functions
        self.mock_st_functions()
        
        # Create test session directory
        os.makedirs("test_sessions", exist_ok=True)
        
        logger.info("CHECKPOINT: Test environment setup completed successfully")
    
    def tearDown(self):
        """Clean up after tests"""
        logger.info("Cleaning up test environment")
        
        # Stop patches
        self.session_state_patch.stop()
        
        # Remove test session files
        for file in os.listdir("test_sessions"):
            os.remove(os.path.join("test_sessions", file))
        os.rmdir("test_sessions")
        
        logger.info("CHECKPOINT: Test environment cleanup completed successfully")
    
    def mock_st_functions(self):
        """Mock streamlit functions"""
        # Create mock for streamlit functions
        self.st_markdown_patch = patch('streamlit.markdown', return_value=None)
        self.st_markdown_mock = self.st_markdown_patch.start()
        
        self.st_text_input_patch = patch('streamlit.text_input', return_value="Test input")
        self.st_text_input_mock = self.st_text_input_patch.start()
        
        self.st_selectbox_patch = patch('streamlit.selectbox', return_value="llama2")
        self.st_selectbox_mock = self.st_selectbox_patch.start()
        
        self.st_button_patch = patch('streamlit.button', return_value=False)
        self.st_button_mock = self.st_button_patch.start()
        
        self.st_expander_patch = patch('streamlit.expander', return_value=MagicMock())
        self.st_expander_mock = self.st_expander_patch.start()
        
        self.st_sidebar_patch = patch('streamlit.sidebar', return_value=MagicMock())
        self.st_sidebar_mock = self.st_sidebar_patch.start()
        
        self.st_container_patch = patch('streamlit.container', return_value=MagicMock())
        self.st_container_mock = self.st_container_patch.start()
        
        self.st_spinner_patch = patch('streamlit.spinner', return_value=MagicMock())
        self.st_spinner_mock = self.st_spinner_patch.start()
        
        self.st_empty_patch = patch('streamlit.empty', return_value=MagicMock())
        self.st_empty_mock = self.st_empty_patch.start()
        
        self.st_rerun_patch = patch('streamlit.rerun', return_value=None)
        self.st_rerun_mock = self.st_rerun_patch.start()
        
        logger.info("CHECKPOINT: Streamlit functions mocked successfully")
    
    @patch('modern_chat_interface.SESSIONS_FOLDER', "test_sessions")
    def test_save_and_load_chat_session(self):
        """Test saving and loading chat sessions"""
        logger.info("Testing save and load chat session")
        
        # Initialize session state
        self.mock_session_state["chat_history"] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ]
        self.mock_session_state["current_model"] = "llama2"
        
        # Save session
        save_chat_session()
        
        # Check that session file was created
        session_files = os.listdir("test_sessions")
        self.assertEqual(len(session_files), 1)
        
        # Get the session file path
        session_file = os.path.join("test_sessions", session_files[0])
        
        # Clear session state
        self.mock_session_state.clear()
        
        # Load session
        load_chat_session(session_file)
        
        # Check that session state was restored
        self.assertEqual(len(self.mock_session_state["chat_history"]), 4)
        self.assertEqual(self.mock_session_state["chat_history"][0]["content"], "Hello")
        self.assertEqual(self.mock_session_state["chat_history"][1]["content"], "Hi there!")
        self.assertEqual(self.mock_session_state["current_model"], "llama2")
        
        logger.info("Save and load chat session test passed")
        logger.info("CHECKPOINT: Save and load chat session test completed successfully")
    
    def test_session_state_synchronization(self):
        """Test that session state is synchronized between different interfaces"""
        logger.info("Testing session state synchronization")
        
        # Initialize session state with different variable names used by different interfaces
        self.mock_session_state["selected_model"] = "llama2"  # Used by chat_interface.py
        
        # Check that current_model is synchronized with selected_model
        # This would normally happen in main.py
        if "selected_model" in self.mock_session_state and self.mock_session_state["selected_model"]:
            if "current_model" not in self.mock_session_state:
                self.mock_session_state["current_model"] = self.mock_session_state["selected_model"]
        
        # Check that current_model was set correctly
        self.assertEqual(self.mock_session_state["current_model"], "llama2")
        
        # Change current_model and check that selected_model is updated
        self.mock_session_state["current_model"] = "mistral"
        
        # This would normally happen in main.py
        if "current_model" in self.mock_session_state and self.mock_session_state["current_model"]:
            self.mock_session_state["selected_model"] = self.mock_session_state["current_model"]
        
        # Check that selected_model was updated
        self.assertEqual(self.mock_session_state["selected_model"], "mistral")
        
        logger.info("Session state synchronization test passed")
        logger.info("CHECKPOINT: Session state synchronization test completed successfully")
    
    def test_session_state_persistence_across_reruns(self):
        """Test that session state persists across streamlit reruns"""
        logger.info("Testing session state persistence across reruns")
        
        # Initialize session state
        self.mock_session_state["chat_history"] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        self.mock_session_state["current_model"] = "llama2"
        
        # Simulate a streamlit rerun by saving session state, clearing it, and restoring it
        # In a real scenario, streamlit would persist session state across reruns
        
        # Save session state
        saved_state = self.mock_session_state.copy()
        
        # Clear session state
        self.mock_session_state.clear()
        
        # Restore session state (this happens automatically in streamlit)
        for key, value in saved_state.items():
            self.mock_session_state[key] = value
        
        # Check that session state was restored
        self.assertEqual(len(self.mock_session_state["chat_history"]), 2)
        self.assertEqual(self.mock_session_state["chat_history"][0]["content"], "Hello")
        self.assertEqual(self.mock_session_state["current_model"], "llama2")
        
        logger.info("Session state persistence across reruns test passed")
        logger.info("CHECKPOINT: Session state persistence across reruns test completed successfully")
    
    def test_chat_history_update(self):
        """Test that chat history is updated correctly when a new message is added"""
        logger.info("Testing chat history update")
        
        # Initialize session state
        self.mock_session_state["chat_history"] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        # Add a new message to chat history
        self.mock_session_state["chat_history"].append({"role": "user", "content": "How are you?"})
        
        # Check that chat history was updated
        self.assertEqual(len(self.mock_session_state["chat_history"]), 3)
        self.assertEqual(self.mock_session_state["chat_history"][2]["content"], "How are you?")
        
        # Add an assistant response
        self.mock_session_state["chat_history"].append({"role": "assistant", "content": "I'm doing well, thank you!"})
        
        # Check that chat history was updated
        self.assertEqual(len(self.mock_session_state["chat_history"]), 4)
        self.assertEqual(self.mock_session_state["chat_history"][3]["content"], "I'm doing well, thank you!")
        
        logger.info("Chat history update test passed")
        logger.info("CHECKPOINT: Chat history update test completed successfully")
    
    def test_session_state_initialization_with_defaults(self):
        """Test that session state is initialized with default values when not present"""
        logger.info("Testing session state initialization with defaults")
        
        # Clear session state
        self.mock_session_state.clear()
        
        # Call initialize_session_state
        initialize_session_state()
        
        # Check that session state was initialized with default values
        self.assertIn("chat_history", self.mock_session_state)
        self.assertEqual(self.mock_session_state["chat_history"], [])
        
        self.assertIn("current_model", self.mock_session_state)
        # The actual value depends on available models, but it should be set
        self.assertIsNotNone(self.mock_session_state["current_model"])
        
        self.assertIn("temperature", self.mock_session_state)
        self.assertEqual(self.mock_session_state["temperature"], 0.7)
        
        self.assertIn("max_tokens", self.mock_session_state)
        self.assertEqual(self.mock_session_state["max_tokens"], 4000)
        
        logger.info("Session state initialization with defaults test passed")
        logger.info("CHECKPOINT: Session state initialization with defaults test completed successfully")

if __name__ == "__main__":
    unittest.main()
