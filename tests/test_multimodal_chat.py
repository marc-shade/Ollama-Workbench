from unittest.mock import MagicMock
import unittest
from unittest.mock import patch
import streamlit as st

from ollama_workbench.chat.multimodal_chat import multimodal_chat_interface  # Absolute import
import ollama_workbench.providers.ollama_utils as ollama_utils


class TestMultimodalChatInterface(unittest.TestCase):

    @patch('ollama_workbench.chat.multimodal_chat.get_available_models')
    def test_interface_exists(self, mock_get_available_models):
        # Mock get_available_models to return an empty list
        mock_get_available_models.return_value = []

        # This test will simply call the function to check if it runs without errors
        try:
            multimodal_chat_interface()
            self.assertTrue(True)  # If it runs without errors, the test passes
        except Exception as e:
            self.fail(f"Exception raised: {e}")

    @patch('ollama_workbench.chat.multimodal_chat.get_available_models')
    @patch('ollama_workbench.chat.multimodal_chat.get_ollama_client')
    @patch('streamlit.rerun')
    @patch('streamlit.button')
    @patch('streamlit.text_area')
    @patch('streamlit.chat_message')
    def test_text_input_ollama(self, mock_chat_message, mock_text_area, mock_button,
                               mock_rerun, mock_get_client, mock_get_models):
        # Simulate user input
        mock_text_area.return_value = "Hello, world!"

        # Hermetic model list (no live Ollama server)
        mock_get_models.return_value = ["test-model"]

        # Two st.button calls in the interface: "Clear Chat" (sidebar), then "Send"
        mock_button.side_effect = [False, True]

        # Mock the Ollama client; the interface sends text-only single messages
        # through client.chat and reads response["message"]["content"]
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "Mocked Ollama response"}}
        mock_get_client.return_value = mock_client

        with patch('streamlit.session_state', new_callable=MagicMock) as mock_session_state:
            # Initialize session state
            mock_session_state.multimodal_chat_history = []
            mock_session_state.multimodal_selected_model = 'test-model'
            mock_session_state.get = MagicMock(side_effect=lambda key, default: 'Ollama' if key == 'multimodal_provider' else default)

            # Run the interface function
            multimodal_chat_interface()

            # Assertions
            # Check if the user message was added to session state
            self.assertEqual(len(mock_session_state.multimodal_chat_history), 2)
            self.assertEqual(mock_session_state.multimodal_chat_history[0]['role'], 'user')
            self.assertEqual(mock_session_state.multimodal_chat_history[0]['content'], 'Hello, world!')

            self.assertEqual(mock_session_state.multimodal_chat_history[1]['role'], 'assistant')
            self.assertEqual(mock_session_state.multimodal_chat_history[1]['content'], 'Mocked Ollama response')

            # Check the Ollama backend was called once with the user's message
            # for the selected model (current contract: client.chat with a
            # messages list and generation options)
            mock_client.chat.assert_called_once_with(
                model='test-model',
                messages=[{'role': 'user', 'content': 'Hello, world!'}],
                options={
                    'temperature': 0.7,   # slider default (bare-mode widgets return their declared value)
                    'num_predict': 4000   # max-tokens slider default
                }
            )

            # Check if chat_message was called for both user and assistant messages
            self.assertEqual(mock_chat_message.call_count, 2)

            # The interface triggers a rerun after sending
            mock_rerun.assert_called_once()

    @patch('ollama_workbench.chat.multimodal_chat.subprocess.run')
    @patch('ollama_workbench.chat.multimodal_chat.get_available_models')
    @patch('ollama_workbench.chat.multimodal_chat.get_ollama_client')
    @patch('streamlit.rerun')
    @patch('streamlit.button')
    @patch('streamlit.file_uploader')
    @patch('streamlit.image')
    @patch('streamlit.text_area')
    @patch('streamlit.chat_message')
    def test_image_cli_fallback_uses_valid_ollama_args(self, mock_chat_message, mock_text_area,
                                                       mock_image, mock_file_uploader, mock_button,
                                                       mock_rerun, mock_get_client, mock_get_models,
                                                       mock_subprocess_run):
        """Regression test: the last-resort CLI image fallback must not pass
        a nonexistent --image flag to `ollama run` (images go in the prompt)."""
        mock_text_area.return_value = "What is this?"
        mock_get_models.return_value = ["llava"]
        mock_button.side_effect = [False, True]  # Clear Chat, Send

        # Fake uploaded image file
        fake_file = MagicMock()
        fake_file.read.return_value = b"fake-image-bytes"
        fake_file.type = "image/jpeg"
        mock_file_uploader.return_value = fake_file

        # Client whose chat API rejects multimodal input -> triggers fallbacks
        mock_client = MagicMock()
        mock_client.chat.side_effect = AttributeError("chat API not supported")
        mock_get_client.return_value = mock_client

        # CLI fallback succeeds
        cli_result = MagicMock()
        cli_result.stdout = "CLI response"
        mock_subprocess_run.return_value = cli_result

        with patch('streamlit.session_state', new_callable=MagicMock) as mock_session_state, \
             patch('ollama_workbench.providers.ollama_utils.call_ollama_endpoint',
                   side_effect=Exception("endpoint unavailable")):
            mock_session_state.multimodal_chat_history = []
            mock_session_state.get = MagicMock(side_effect=lambda key, default: 'Ollama' if key == 'multimodal_provider' else default)

            multimodal_chat_interface()

            # The CLI was invoked without the invalid --image flag, with the
            # image path embedded in the single prompt argument
            mock_subprocess_run.assert_called_once()
            cmd = mock_subprocess_run.call_args[0][0]
            self.assertNotIn("--image", cmd)
            self.assertEqual(cmd[:3], ["ollama", "run", "llava"])
            self.assertEqual(len(cmd), 4)
            self.assertTrue(cmd[3].startswith("What is this?"))

            # The CLI output made it into the chat history as the assistant reply
            self.assertEqual(mock_session_state.multimodal_chat_history[-1]['role'], 'assistant')
            self.assertEqual(mock_session_state.multimodal_chat_history[-1]['content'], 'CLI response')


if __name__ == '__main__':
    unittest.main()
