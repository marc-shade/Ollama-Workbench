from unittest.mock import MagicMock
import unittest
from unittest.mock import patch
import streamlit as st

from multimodal_chat import multimodal_chat_interface  # Absolute import
import ollama_utils

class TestMultimodalChatInterface(unittest.TestCase):

    @patch('ollama_utils.get_available_models')
    def test_interface_exists(self, mock_get_available_models):
        # Mock get_available_models to return an empty list
        mock_get_available_models.return_value = []

        # This test will simply call the function to check if it runs without errors
        try:
            multimodal_chat_interface()
            self.assertTrue(True)  # If it runs without errors, the test passes
        except Exception as e:
            self.fail(f"Exception raised: {e}")

    @patch('ollama_utils.call_ollama_endpoint')
    @patch('streamlit.text_area')
    @patch('streamlit.chat_message')
    def test_text_input_ollama(self, mock_chat_message, mock_text_area, mock_generate_ollama_response):
        # Simulate user input
        mock_text_area.return_value = "Hello, world!"

        # Mock the Ollama response
        mock_generate_ollama_response.return_value = "Mocked Ollama response"

        with patch('streamlit.session_state', new_callable=MagicMock) as mock_session_state:
            # Initialize session state
            mock_session_state.multimodal_chat_history = []
            mock_session_state.multimodal_chat_history = []
            mock_session_state.multimodal_selected_model = 'ollama'
            mock_session_state.ollama_model = 'test-model'
            mock_session_state.get = MagicMock(side_effect=lambda key, default: 'Ollama' if key == 'multimodal_provider' else default)

            # Run the interface function
            print(f"Provider: {mock_session_state.get('multimodal_provider', 'Ollama')}")
            multimodal_chat_interface()
    
            # Assertions
            # Check if the user message was added to session state
            self.assertEqual(len(mock_session_state.multimodal_chat_history), 2)
            self.assertEqual(mock_session_state.multimodal_chat_history[0]['role'], 'user')
            self.assertEqual(mock_session_state.multimodal_chat_history[0]['content'], 'Hello, world!')

            self.assertEqual(mock_session_state.multimodal_chat_history[1]['role'], 'assistant')
            self.assertEqual(mock_session_state.multimodal_chat_history[1]['content'], 'Mocked Ollama response')

            # Check if generate_ollama_response was called with correct arguments
            mock_generate_ollama_response.assert_called_once_with(
                'test-model',
                [{'role': 'user', 'content': 'Hello, world!'}],
                None, # No image
                None  # No audio
            )

            # Check if chat_message was called for both user and assistant messages
            self.assertEqual(mock_chat_message.call_count, 2)

        # Check if generate_ollama_response was called with correct arguments
        mock_generate_ollama_response.assert_called_once_with(
            'test-model',
            [{'role': 'user', 'content': 'Hello, world!'}],
            None, # No image
            None  # No audio
        )

        # Check if chat_message was called for both user and assistant messages
        mock_chat_message.call_count == 2

if __name__ == '__main__':
    unittest.main()