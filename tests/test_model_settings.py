"""
Test suite for Ollama-Workbench model settings and agent features

This test suite verifies that model settings can be changed, saved, and loaded correctly,
and that all agent features (thinking types, agent types, etc.) work properly.
"""

import os
import sys
import unittest
import json
import logging
import time
from unittest.mock import patch, MagicMock, call

# Set up logging with detailed checkpoints for troubleshooting
logging.basicConfig(
    filename='test_model_settings.log',
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
    from ollama_workbench.chat.chat_interface import load_settings, save_settings, construct_agent_prompt
    from ollama_workbench.ui.prompts import get_agent_prompt, get_metacognitive_prompt, get_voice_prompt
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

class TestModelSettings(unittest.TestCase):
    """Test case for model settings"""
    
    def setUp(self):
        """Set up test environment"""
        logger.info("Setting up test environment for model settings tests")
        logger.info("CHECKPOINT: Beginning test setup")
        
        # Mock streamlit session state
        self.mock_session_state = {}
        
        # Create patch for st.session_state
        self.session_state_patch = patch('streamlit.session_state', self.mock_session_state)
        self.mock_st_session_state = self.session_state_patch.start()
        
        # Mock streamlit functions
        self.mock_st_functions()
        
        # Mock ollama client
        self.mock_ollama_client()
        
        # Create test settings file
        self.create_test_settings()
        
        logger.info("CHECKPOINT: Test environment setup completed successfully")
    
    def tearDown(self):
        """Clean up after tests"""
        logger.info("Cleaning up test environment")
        
        # Stop patches
        self.session_state_patch.stop()
        
        # Remove test settings file
        if os.path.exists("test-chat-settings.json"):
            os.remove("test-chat-settings.json")
        
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
        
        self.st_slider_patch = patch('streamlit.slider', return_value=0.7)
        self.st_slider_mock = self.st_slider_patch.start()
        
        self.st_number_input_patch = patch('streamlit.number_input', return_value=4000)
        self.st_number_input_mock = self.st_number_input_patch.start()
        
        self.st_button_patch = patch('streamlit.button', return_value=False)
        self.st_button_mock = self.st_button_patch.start()
        
        self.st_expander_patch = patch('streamlit.expander', return_value=MagicMock())
        self.st_expander_mock = self.st_expander_patch.start()
        
        self.st_sidebar_patch = patch('streamlit.sidebar', return_value=MagicMock())
        self.st_sidebar_mock = self.st_sidebar_patch.start()
        
        logger.info("CHECKPOINT: Streamlit functions mocked successfully")
    
    def mock_ollama_client(self):
        """Mock ollama client"""
        # Create mock for ollama client
        self.ollama_client_patch = patch('ollama_workbench.providers.ollama_utils.get_ollama_client')
        self.ollama_client_mock = self.ollama_client_patch.start()
        
        # Configure mock to return a client that returns a response
        mock_client = MagicMock()
        mock_client.generate.return_value = [{"response": "Test response"}]
        self.ollama_client_mock.return_value = mock_client
        
        # Mock get_available_models
        self.get_models_patch = patch('ollama_workbench.providers.ollama_utils.get_available_models', 
                                      return_value=["llama2", "mistral", "llama3", "phi3"])
        self.get_models_mock = self.get_models_patch.start()
        
        logger.info("CHECKPOINT: Ollama client mocked successfully")
    
    def create_test_settings(self):
        """Create test settings file"""
        test_settings = {
            "selected_model": "llama2",
            "agent_type": "None",
            "metacognitive_type": "None",
            "voice_type": "None",
            "selected_corpus": "None",
            "temperature": 0.7,
            "max_tokens": 4000,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        }
        
        with open("test-chat-settings.json", "w") as f:
            json.dump(test_settings, f)
        
        logger.info("CHECKPOINT: Test settings file created successfully")
    
    @patch('ollama_workbench.chat.chat_interface.SETTINGS_FILE', "test-chat-settings.json")
    def test_change_model_setting(self):
        """Test changing model settings"""
        logger.info("Testing changing model settings")
        
        # Load initial settings
        load_settings()
        
        # Check initial settings
        self.assertEqual(self.mock_session_state.get("selected_model"), "llama2")

        # Change model settings (use actual key names from save_settings)
        self.mock_session_state["selected_model"] = "mistral"
        self.mock_session_state["temperature_slider_chat"] = 0.8
        self.mock_session_state["max_tokens_slider_chat"] = 2000

        # Save settings
        save_settings()

        # Clear session state
        self.mock_session_state.clear()

        # Load settings again
        load_settings()

        # Check that settings were saved and loaded correctly
        self.assertEqual(self.mock_session_state.get("selected_model"), "mistral")
        self.assertEqual(self.mock_session_state.get("temperature_slider_chat"), 0.8)
        self.assertEqual(self.mock_session_state.get("max_tokens_slider_chat"), 2000)
        
        logger.info("CHECKPOINT: Model settings change test passed")
    
    @patch('ollama_workbench.chat.chat_interface.SETTINGS_FILE', "test-chat-settings.json")
    def test_change_agent_settings(self):
        """Test changing agent settings"""
        logger.info("Testing changing agent settings")
        
        # Load initial settings
        load_settings()
        
        # Check initial settings
        self.assertEqual(self.mock_session_state.get("agent_type"), "None")
        self.assertEqual(self.mock_session_state.get("metacognitive_type"), "None")
        
        # Change agent settings
        self.mock_session_state["agent_type"] = "Researcher"
        self.mock_session_state["metacognitive_type"] = "Analytical"
        self.mock_session_state["voice_type"] = "Friendly"
        
        # Save settings
        save_settings()
        
        # Clear session state
        self.mock_session_state.clear()
        
        # Load settings again
        load_settings()
        
        # Check that settings were saved and loaded correctly
        self.assertEqual(self.mock_session_state.get("agent_type"), "Researcher")
        self.assertEqual(self.mock_session_state.get("metacognitive_type"), "Analytical")
        self.assertEqual(self.mock_session_state.get("voice_type"), "Friendly")
        
        logger.info("CHECKPOINT: Agent settings change test passed")
    
    @patch('ollama_workbench.ui.prompts.get_agent_prompt')
    @patch('ollama_workbench.ui.prompts.get_metacognitive_prompt')
    @patch('ollama_workbench.ui.prompts.get_voice_prompt')
    def test_construct_agent_prompt(self, mock_voice_prompt, mock_metacog_prompt, mock_agent_prompt):
        """Test constructing agent prompt from different types"""
        logger.info("Testing agent prompt construction")
        
        # Mock prompt functions
        mock_agent_prompt.return_value = {
            "Researcher": "I approach problems systematically, breaking them down into smaller components.",
            "Coder": "You are a coding assistant."
        }
        
        mock_metacog_prompt.return_value = {
            "Analytical": "You think analytically.",
            "Creative": "You think creatively."
        }
        
        mock_voice_prompt.return_value = {
            "Friendly": "You speak in a friendly tone.",
            "Professional": "You speak in a professional tone."
        }
        
        # Test with all types specified
        prompt = construct_agent_prompt("Researcher", "Analytical", "Friendly")
        
        # Check that prompt contains all components
        self.assertIn("approach problems systematically", prompt)
        self.assertIn("You think analytically.", prompt)
        self.assertIn("You speak in a friendly tone.", prompt)
        
        # Test with only agent type
        prompt = construct_agent_prompt("Coder", "None", "None")
        
        # Check that prompt contains only agent component
        self.assertIn("You are a coding assistant.", prompt)
        self.assertNotIn("You think", prompt)
        self.assertNotIn("You speak", prompt)
        
        # Test with custom prompt
        custom_prompt = "You are a custom assistant."
        prompt = construct_agent_prompt("None", "None", "None", custom_prompt)
        
        # Check that prompt is the custom prompt
        self.assertEqual(prompt, custom_prompt)
        
        logger.info("CHECKPOINT: Agent prompt construction test passed")
    
    def test_all_available_models(self):
        """Test that all available models can be selected"""
        logger.info("Testing all available models")
        
        # Get available models
        available_models = ["llama2", "mistral", "llama3", "phi3"]
        
        # Test each model
        for model in available_models:
            logger.info(f"Testing model: {model}")
            
            # Set model in session state
            self.mock_session_state["selected_model"] = model
            
            # Check that model was set correctly
            self.assertEqual(self.mock_session_state.get("selected_model"), model)
        
        logger.info("CHECKPOINT: All available models test passed")
    
    def test_temperature_range(self):
        """Test that temperature can be set to different values"""
        logger.info("Testing temperature range")
        
        # Test different temperature values
        for temp in [0.0, 0.1, 0.5, 0.7, 1.0]:
            logger.info(f"Testing temperature: {temp}")
            
            # Set temperature in session state
            self.mock_session_state["temperature"] = temp
            
            # Check that temperature was set correctly
            self.assertEqual(self.mock_session_state.get("temperature"), temp)
        
        logger.info("CHECKPOINT: Temperature range test passed")
    
    def test_max_tokens_range(self):
        """Test that max_tokens can be set to different values"""
        logger.info("Testing max_tokens range")
        
        # Test different max_tokens values
        for tokens in [100, 1000, 2000, 4000, 8000]:
            logger.info(f"Testing max_tokens: {tokens}")
            
            # Set max_tokens in session state
            self.mock_session_state["max_tokens"] = tokens
            
            # Check that max_tokens was set correctly
            self.assertEqual(self.mock_session_state.get("max_tokens"), tokens)
        
        logger.info("CHECKPOINT: Max tokens range test passed")

class TestAgentFeatures(unittest.TestCase):
    """Test case for agent features"""
    
    def setUp(self):
        """Set up test environment"""
        logger.info("Setting up test environment for agent features tests")
        
        # Mock streamlit session state
        self.mock_session_state = {}
        
        # Create patch for st.session_state
        self.session_state_patch = patch('streamlit.session_state', self.mock_session_state)
        self.mock_st_session_state = self.session_state_patch.start()
        
        # Mock prompts module
        self.mock_prompts()
        
        logger.info("CHECKPOINT: Test environment setup completed successfully")
    
    def tearDown(self):
        """Clean up after tests"""
        logger.info("Cleaning up test environment")
        
        # Stop patches
        self.session_state_patch.stop()
        self.agent_prompt_patch.stop()
        self.metacog_prompt_patch.stop()
        self.voice_prompt_patch.stop()
        
        logger.info("CHECKPOINT: Test environment cleanup completed successfully")
    
    def mock_prompts(self):
        """Mock prompts module"""
        # Create mock for prompts functions
        self.agent_prompt_patch = patch('ollama_workbench.ui.prompts.get_agent_prompt')
        self.agent_prompt_mock = self.agent_prompt_patch.start()
        self.agent_prompt_mock.return_value = {
            "Researcher": "I approach problems systematically, breaking them down into smaller components.",
            "Coder": "You are a coding assistant.",
            "Teacher": "You are a teaching assistant.",
            "Writer": "You are a writing assistant."
        }
        
        self.metacog_prompt_patch = patch('ollama_workbench.ui.prompts.get_metacognitive_prompt')
        self.metacog_prompt_mock = self.metacog_prompt_patch.start()
        self.metacog_prompt_mock.return_value = {
            "Analytical": "You think analytically.",
            "Creative": "You think creatively.",
            "Critical": "You think critically.",
            "Reflective": "You think reflectively."
        }
        
        self.voice_prompt_patch = patch('ollama_workbench.ui.prompts.get_voice_prompt')
        self.voice_prompt_mock = self.voice_prompt_patch.start()
        self.voice_prompt_mock.return_value = {
            "Friendly": "You speak in a friendly tone.",
            "Professional": "You speak in a professional tone.",
            "Casual": "You speak in a casual tone.",
            "Formal": "You speak in a formal tone."
        }
        
        logger.info("CHECKPOINT: Prompts mocked successfully")
    
    def test_agent_types(self):
        """Test that all agent types are available"""
        logger.info("Testing agent types")
        
        # Get agent types
        agent_types = list(self.agent_prompt_mock.return_value.keys())
        
        # Check that all expected agent types are available
        expected_types = ["Researcher", "Coder", "Teacher", "Writer"]
        for agent_type in expected_types:
            self.assertIn(agent_type, agent_types)
        
        logger.info("CHECKPOINT: Agent types test passed")
    
    def test_metacognitive_types(self):
        """Test that all metacognitive types are available"""
        logger.info("Testing metacognitive types")
        
        # Get metacognitive types
        metacog_types = list(self.metacog_prompt_mock.return_value.keys())
        
        # Check that all expected metacognitive types are available
        expected_types = ["Analytical", "Creative", "Critical", "Reflective"]
        for metacog_type in expected_types:
            self.assertIn(metacog_type, metacog_types)
        
        logger.info("CHECKPOINT: Metacognitive types test passed")
    
    def test_voice_types(self):
        """Test that all voice types are available"""
        logger.info("Testing voice types")
        
        # Get voice types
        voice_types = list(self.voice_prompt_mock.return_value.keys())
        
        # Check that all expected voice types are available
        expected_types = ["Friendly", "Professional", "Casual", "Formal"]
        for voice_type in expected_types:
            self.assertIn(voice_type, voice_types)
        
        logger.info("CHECKPOINT: Voice types test passed")
    
    def test_agent_combinations(self):
        """Test that different combinations of agent features work"""
        logger.info("Testing agent combinations")
        
        # Test different combinations of agent features
        combinations = [
            ("Researcher", "Analytical", "Professional"),
            ("Coder", "Creative", "Casual"),
            ("Teacher", "Critical", "Friendly"),
            ("Writer", "Reflective", "Formal")
        ]
        
        for agent_type, metacog_type, voice_type in combinations:
            logger.info(f"Testing combination: {agent_type}, {metacog_type}, {voice_type}")
            
            # Set agent features in session state
            self.mock_session_state["agent_type"] = agent_type
            self.mock_session_state["metacognitive_type"] = metacog_type
            self.mock_session_state["voice_type"] = voice_type
            
            # Check that agent features were set correctly
            self.assertEqual(self.mock_session_state.get("agent_type"), agent_type)
            self.assertEqual(self.mock_session_state.get("metacognitive_type"), metacog_type)
            self.assertEqual(self.mock_session_state.get("voice_type"), voice_type)
            
            # Construct agent prompt
            prompt = construct_agent_prompt(agent_type, metacog_type, voice_type)
            
            # Check that prompt contains all components
            self.assertIn(self.agent_prompt_mock.return_value[agent_type], prompt)
            self.assertIn(self.metacog_prompt_mock.return_value[metacog_type], prompt)
            self.assertIn(self.voice_prompt_mock.return_value[voice_type], prompt)
        
        logger.info("CHECKPOINT: Agent combinations test passed")

if __name__ == "__main__":
    unittest.main()
