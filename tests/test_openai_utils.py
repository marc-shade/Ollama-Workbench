"""
Test suite for openai_utils.py - OpenAI provider integration
"""

import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Import the module to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ollama_workbench.providers.openai_utils import (
    OPENAI_MODELS, load_api_keys, save_api_keys, set_openai_api_key,
    call_openai_api, call_openai_embeddings, display_openai_settings
)


class TestAPIKeyManagement:
    """Test API key loading, saving, and setting functions"""
    
    def test_load_api_keys_exists(self, tmp_path):
        """Test loading API keys when file exists"""
        # Create a test file
        test_keys = {"openai_api_key": "sk-test-123", "other_key": "value"}
        api_file = tmp_path / "api_keys.json"
        with open(api_file, "w") as f:
            json.dump(test_keys, f)

        # Mock the file path and clear cache
        with patch('ollama_workbench.providers.ollama_utils.API_KEYS_FILE', str(api_file)), \
             patch('ollama_workbench.providers.ollama_utils._api_keys_cache', None), \
             patch('ollama_workbench.providers.ollama_utils._api_keys_cache_time', 0):
            loaded_keys = load_api_keys()
            assert loaded_keys == test_keys
            assert loaded_keys["openai_api_key"] == "sk-test-123"

    def test_load_api_keys_not_exists(self):
        """Test loading API keys when file doesn't exist"""
        with patch('ollama_workbench.providers.ollama_utils.API_KEYS_FILE', '/nonexistent/api_keys.json'), \
             patch('ollama_workbench.providers.ollama_utils._api_keys_cache', None), \
             patch('ollama_workbench.providers.ollama_utils._api_keys_cache_time', 0):
            loaded_keys = load_api_keys()
            assert loaded_keys == {}

    def test_save_api_keys(self, tmp_path):
        """Test saving API keys to file"""
        test_keys = {"openai_api_key": "sk-test-456", "another_key": "another_value"}
        api_file = tmp_path / "api_keys.json"

        with patch('ollama_workbench.providers.ollama_utils.API_KEYS_FILE', str(api_file)):
            save_api_keys(test_keys)

            # Verify file was saved correctly
            with open(api_file, "r") as f:
                saved_keys = json.load(f)
            assert saved_keys == test_keys
            assert saved_keys["openai_api_key"] == "sk-test-456"

    @patch('streamlit.success')
    def test_set_openai_api_key(self, mock_success, tmp_path):
        """Test setting OpenAI API key"""
        api_file = tmp_path / "api_keys.json"
        initial_keys = {"other_key": "value"}

        with open(api_file, "w") as f:
            json.dump(initial_keys, f)

        with patch('ollama_workbench.providers.ollama_utils.API_KEYS_FILE', str(api_file)), \
             patch('ollama_workbench.providers.ollama_utils._api_keys_cache', None), \
             patch('ollama_workbench.providers.ollama_utils._api_keys_cache_time', 0):
            set_openai_api_key("sk-new-key-789")

            # Verify key was saved
            with open(api_file, "r") as f:
                saved_keys = json.load(f)
            assert saved_keys["openai_api_key"] == "sk-new-key-789"
            assert saved_keys["other_key"] == "value"  # Other keys preserved

            # Verify success message
            mock_success.assert_called_once_with("OpenAI API key has been set.")


class TestOpenAIAPIFunctions:
    """Test OpenAI API calling functions"""
    
    @patch('ollama_workbench.providers.openai_utils.OpenAI')
    def test_call_openai_api_success(self, mock_openai_class):
        """Test successful OpenAI API call"""
        # Mock the client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock the response structure
        mock_message = Mock()
        mock_message.content = "Hello, I'm GPT-4!"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test the function
        messages = [{"role": "user", "content": "Hello"}]
        result = call_openai_api(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=100,
            openai_api_key="sk-test-key"
        )
        
        assert result == "Hello, I'm GPT-4!"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=100,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stream=False
        )
    
    @patch('ollama_workbench.providers.openai_utils.OpenAI')
    def test_call_openai_api_stream(self, mock_openai_class):
        """Test OpenAI API call with streaming"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock streaming response
        mock_stream = Mock()
        mock_client.chat.completions.create.return_value = mock_stream
        
        result = call_openai_api(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            stream=True,
            openai_api_key="sk-test-key"
        )
        
        assert result == mock_stream  # Should return the stream object
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.7,
            max_tokens=1000,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stream=True
        )
    
    @patch('streamlit.error')
    @patch('ollama_workbench.providers.openai_utils.OpenAI')
    def test_call_openai_api_error(self, mock_openai_class, mock_error):
        """Test OpenAI API call error handling"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock an exception
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        result = call_openai_api(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            openai_api_key="sk-test-key"
        )
        
        assert result is None
        mock_error.assert_called_once_with("Error calling OpenAI API: API Error")
    
    @patch('ollama_workbench.providers.openai_utils.OpenAI')
    def test_call_openai_embeddings_success(self, mock_openai_class):
        """Test successful OpenAI embeddings call"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock the response
        mock_embedding_data = Mock()
        mock_embedding_data.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_response = Mock()
        mock_response.data = [mock_embedding_data]
        
        mock_client.embeddings.create.return_value = mock_response
        
        # Need to mock load_api_keys for this function
        with patch('ollama_workbench.providers.openai_utils.load_api_keys', return_value={'openai_api_key': 'sk-test-key'}):
            result = call_openai_embeddings("text-embedding-ada-002", "Test text")
        
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input="Test text"
        )
    
    @patch('streamlit.error')
    @patch('ollama_workbench.providers.openai_utils.OpenAI')
    def test_call_openai_embeddings_error(self, mock_openai_class, mock_error):
        """Test OpenAI embeddings error handling"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock an exception
        mock_client.embeddings.create.side_effect = Exception("Embeddings Error")
        
        with patch('ollama_workbench.providers.openai_utils.load_api_keys', return_value={'openai_api_key': 'sk-test-key'}):
            result = call_openai_embeddings("text-embedding-ada-002", "Test text")
        
        assert result is None
        # Check that error was called with a message containing the error
        mock_error.assert_called_once()
        error_message = mock_error.call_args[0][0]
        assert "Error calling OpenAI API:" in error_message


class TestUIFunctions:
    """Test UI-related functions"""
    
    @patch('streamlit.sidebar.button')
    @patch('streamlit.sidebar.text_input')
    @patch('streamlit.sidebar.subheader')
    @patch('ollama_workbench.providers.openai_utils.set_openai_api_key')
    @patch('ollama_workbench.providers.openai_utils.load_api_keys')
    def test_display_openai_settings_save_key(
        self, mock_load_keys, mock_set_key, mock_subheader, 
        mock_text_input, mock_button
    ):
        """Test displaying OpenAI settings and saving key"""
        # Setup mocks
        mock_load_keys.return_value = {"openai_api_key": "existing-key"}
        mock_text_input.return_value = "new-api-key"
        mock_button.return_value = True  # Simulate button click
        
        # Call the function
        display_openai_settings()
        
        # Verify calls
        mock_subheader.assert_called_once_with("OpenAI API Key")
        mock_text_input.assert_called_once_with(
            "Enter your OpenAI API key:",
            value="existing-key",
            type="password"
        )
        mock_button.assert_called_once_with("Save OpenAI API Key")
        mock_set_key.assert_called_once_with("new-api-key")
    
    @patch('streamlit.sidebar.button')
    @patch('streamlit.sidebar.text_input')
    @patch('streamlit.sidebar.subheader')
    @patch('ollama_workbench.providers.openai_utils.load_api_keys')
    def test_display_openai_settings_no_save(
        self, mock_load_keys, mock_subheader, mock_text_input, mock_button
    ):
        """Test displaying OpenAI settings without saving"""
        # Setup mocks
        mock_load_keys.return_value = {}
        mock_text_input.return_value = "some-key"
        mock_button.return_value = False  # No button click
        
        # Call the function
        display_openai_settings()
        
        # Verify calls
        mock_text_input.assert_called_once_with(
            "Enter your OpenAI API key:",
            value="",  # No existing key
            type="password"
        )
        # set_openai_api_key should not be called
        with patch('ollama_workbench.providers.openai_utils.set_openai_api_key') as mock_set_key:
            mock_set_key.assert_not_called()


class TestConstants:
    """Test module constants"""
    
    def test_openai_models_list(self):
        """Test that OPENAI_MODELS is properly defined"""
        assert isinstance(OPENAI_MODELS, list)
        assert len(OPENAI_MODELS) > 0
        
        # Check some expected models are present
        assert "gpt-4" in OPENAI_MODELS
        assert "gpt-3.5-turbo" in OPENAI_MODELS
        assert "gpt-4o" in OPENAI_MODELS
        
        # All items should be strings
        assert all(isinstance(model, str) for model in OPENAI_MODELS)


class TestIntegration:
    """Integration tests for complete workflows"""
    
    @patch('ollama_workbench.providers.openai_utils.OpenAI')
    def test_complete_api_flow(self, mock_openai_class, tmp_path):
        """Test complete flow: set key, save, load, and use API"""
        api_file = tmp_path / "api_keys.json"

        with patch('ollama_workbench.providers.ollama_utils.API_KEYS_FILE', str(api_file)), \
             patch('ollama_workbench.providers.ollama_utils._api_keys_cache', None), \
             patch('ollama_workbench.providers.ollama_utils._api_keys_cache_time', 0):
            # Set API key
            with patch('streamlit.success'):
                set_openai_api_key("sk-integration-test")

            # Clear cache after save so load picks up new data
            import ollama_workbench.providers.ollama_utils as _ou
            _ou._api_keys_cache = None
            _ou._api_keys_cache_time = 0

            # Verify key was saved
            keys = load_api_keys()
            assert keys["openai_api_key"] == "sk-integration-test"
            
            # Use the key in API call
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            # Mock response
            mock_message = Mock()
            mock_message.content = "Integration test response"
            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            
            result = call_openai_api(
                model="gpt-4",
                messages=[{"role": "user", "content": "Test"}],
                openai_api_key=keys["openai_api_key"]
            )
            
            assert result == "Integration test response"
    
    def test_all_models_valid(self):
        """Test that all models in OPENAI_MODELS are valid model names"""
        # Basic validation - all should follow OpenAI naming conventions
        for model in OPENAI_MODELS:
            assert isinstance(model, str)
            assert len(model) > 0
            # OpenAI models contain "gpt", "text", "embedding", or reasoning prefixes like "o3-", "o4-"
            assert any(substr in model.lower() for substr in ["gpt", "text", "embedding", "o3-", "o4-", "o1-"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])