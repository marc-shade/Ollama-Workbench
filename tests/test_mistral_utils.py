"""
Test suite for mistral_utils.py - Mistral provider integration
"""

import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import tempfile
import asyncio

# Import the module to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ollama_workbench.providers.mistral_utils import (
    MISTRAL_MODELS, load_api_keys, save_api_keys, get_mistral_client,
    call_mistral_api, call_mistral_api_async, call_mistral_embeddings,
    display_mistral_settings
)


class TestConstants:
    """Test module constants"""
    
    def test_mistral_models_list(self):
        """Test that MISTRAL_MODELS is properly defined"""
        assert isinstance(MISTRAL_MODELS, list)
        assert len(MISTRAL_MODELS) > 0
        
        # Check some expected models are present
        expected_models = ["mistral-large-latest", "mistral-tiny", "mistral-embed"]
        for model in expected_models:
            assert model in MISTRAL_MODELS
        
        # All items should be strings
        assert all(isinstance(model, str) for model in MISTRAL_MODELS)


class TestAPIKeyManagement:
    """Test API key loading and saving functions"""
    
    def test_load_api_keys_exists(self, tmp_path):
        """Test loading API keys when file exists"""
        test_keys = {"mistral_api_key": "msk_test123", "other_key": "value"}
        api_file = tmp_path / "api_keys.json"
        with open(api_file, "w") as f:
            json.dump(test_keys, f)
        
        with patch('mistral_utils.API_KEYS_FILE', str(api_file)):
            loaded_keys = load_api_keys()
            assert loaded_keys == test_keys
            assert loaded_keys["mistral_api_key"] == "msk_test123"
    
    def test_load_api_keys_not_exists(self):
        """Test loading API keys when file doesn't exist"""
        with patch('os.path.exists', return_value=False):
            loaded_keys = load_api_keys()
            assert loaded_keys == {}
    
    def test_save_api_keys(self, tmp_path):
        """Test saving API keys to file"""
        test_keys = {"mistral_api_key": "msk_test456", "another_key": "another_value"}
        api_file = tmp_path / "api_keys.json"
        
        with patch('mistral_utils.API_KEYS_FILE', str(api_file)):
            save_api_keys(test_keys)
            
            # Verify file was saved correctly
            with open(api_file, "r") as f:
                saved_keys = json.load(f)
            assert saved_keys == test_keys
            assert saved_keys["mistral_api_key"] == "msk_test456"


class TestMistralClient:
    """Test Mistral client initialization"""
    
    @patch('mistral_utils.Mistral')
    def test_get_mistral_client_with_key(self, mock_mistral_class):
        """Test getting Mistral client with valid API key"""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        client = get_mistral_client("msk_valid_key")
        
        assert client == mock_client
        mock_mistral_class.assert_called_once_with(api_key="msk_valid_key")
    
    @patch('mistral_utils.load_api_keys')
    @patch('mistral_utils.Mistral')
    def test_get_mistral_client_from_saved_key(self, mock_mistral_class, mock_load_keys):
        """Test getting Mistral client from saved API key"""
        mock_load_keys.return_value = {"mistral_api_key": "msk_saved_key"}
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        client = get_mistral_client()  # No API key provided
        
        assert client == mock_client
        mock_mistral_class.assert_called_once_with(api_key="msk_saved_key")
    
    def test_get_mistral_client_no_key(self):
        """Test getting Mistral client with no API key"""
        with patch('mistral_utils.load_api_keys', return_value={}):
            client = get_mistral_client()
            assert client is None
    
    @patch('streamlit.warning')
    @patch('mistral_utils.Mistral')
    def test_get_mistral_client_error(self, mock_mistral_class, mock_warning):
        """Test Mistral client error handling"""
        mock_mistral_class.side_effect = Exception("Invalid API key")
        
        client = get_mistral_client("msk_invalid_key")
        
        assert client is None
        mock_warning.assert_called_once()
        warning_message = mock_warning.call_args[0][0]
        assert "Error initializing Mistral client:" in warning_message


class TestMistralAPI:
    """Test Mistral API calling functions"""
    
    @patch('mistral_utils.get_mistral_client')
    def test_call_mistral_api_no_client(self, mock_get_client):
        """Test API call with no client"""
        mock_get_client.return_value = None
        
        result = call_mistral_api("mistral-large-latest", prompt="Test")
        assert result is None
    
    @patch('mistral_utils.get_mistral_client')
    def test_call_mistral_api_with_prompt(self, mock_get_client):
        """Test successful Mistral API call with prompt"""
        # Mock client and response
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = "Hello from Mistral!"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        mock_client.chat.complete.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        # Test the function
        result = call_mistral_api(
            model="mistral-large-latest",
            prompt="Hello",
            temperature=0.7,
            max_tokens=100
        )
        
        assert result == "Hello from Mistral!"
        mock_client.chat.complete.assert_called_once()
        
        # Check the call arguments
        call_args = mock_client.chat.complete.call_args[1]
        assert call_args["model"] == "mistral-large-latest"
        assert call_args["messages"] == [{"role": "user", "content": "Hello"}]
        assert call_args["temperature"] == 0.7
        assert call_args["max_tokens"] == 100
    
    @patch('mistral_utils.get_mistral_client')
    def test_call_mistral_api_with_messages(self, mock_get_client):
        """Test Mistral API call with messages"""
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = "Response to conversation"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        mock_client.chat.complete.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
        
        result = call_mistral_api(
            model="mistral-tiny",
            messages=messages
        )
        
        assert result == "Response to conversation"
        call_args = mock_client.chat.complete.call_args[1]
        assert call_args["messages"] == messages
    
    @patch('mistral_utils.get_mistral_client')
    def test_call_mistral_api_stream(self, mock_get_client):
        """Test Mistral API call with streaming"""
        mock_client = Mock()
        mock_stream = Mock()
        mock_client.chat.stream.return_value = mock_stream
        mock_get_client.return_value = mock_client
        
        result = call_mistral_api(
            model="mistral-tiny",
            prompt="Test",
            stream=True
        )
        
        assert result == mock_stream
        mock_client.chat.stream.assert_called_once()
    
    @patch('mistral_utils.get_mistral_client')
    def test_call_mistral_api_json_response(self, mock_get_client):
        """Test Mistral API call with JSON response format"""
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = '{"result": "json response"}'
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        mock_client.chat.complete.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        result = call_mistral_api(
            model="mistral-large-latest",
            prompt="Return JSON",
            json_response=True
        )
        
        assert result == '{"result": "json response"}'
        call_args = mock_client.chat.complete.call_args[1]
        assert call_args["response_format"] == {"type": "json_object"}
    
    @patch('streamlit.error')
    @patch('mistral_utils.get_mistral_client')
    def test_call_mistral_api_error(self, mock_get_client, mock_error):
        """Test Mistral API call error handling"""
        mock_client = Mock()
        mock_client.chat.complete.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client
        
        result = call_mistral_api("mistral-tiny", prompt="Test")
        
        assert result is None
        mock_error.assert_called_once()
        error_message = mock_error.call_args[0][0]
        assert "Error calling Mistral API:" in error_message
    
    def test_call_mistral_api_no_prompt_or_messages(self):
        """Test API call without prompt or messages"""
        mock_client = Mock()
        
        with patch('mistral_utils.get_mistral_client', return_value=mock_client):
            with patch('streamlit.error') as mock_error:
                result = call_mistral_api("mistral-tiny")
                
                assert result is None
                mock_error.assert_called_once()
                error_message = mock_error.call_args[0][0]
                assert "Either prompt or messages must be provided" in error_message


class TestMistralAPIAsync:
    """Test async Mistral API functions"""
    
    @pytest.mark.asyncio
    @patch('mistral_utils.get_mistral_client')
    async def test_call_mistral_api_async_success(self, mock_get_client):
        """Test successful async API call"""
        mock_client = Mock()
        mock_async_response = AsyncMock()
        mock_client.chat.stream_async = mock_async_response
        mock_get_client.return_value = mock_client
        
        result = await call_mistral_api_async(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.8,
            max_tokens=200
        )
        
        assert result == mock_async_response.return_value
        mock_async_response.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('mistral_utils.get_mistral_client')
    async def test_call_mistral_api_async_no_client(self, mock_get_client):
        """Test async API call with no client"""
        mock_get_client.return_value = None
        
        result = await call_mistral_api_async(
            model="mistral-tiny",
            messages=[{"role": "user", "content": "Test"}]
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    @patch('streamlit.error')
    @patch('mistral_utils.get_mistral_client')
    async def test_call_mistral_api_async_error(self, mock_get_client, mock_error):
        """Test async API error handling"""
        mock_client = Mock()
        mock_client.chat.stream_async = AsyncMock(side_effect=Exception("Async error"))
        mock_get_client.return_value = mock_client
        
        result = await call_mistral_api_async(
            model="mistral-tiny",
            messages=[{"role": "user", "content": "Test"}]
        )
        
        assert result is None
        mock_error.assert_called_once()
        error_message = mock_error.call_args[0][0]
        assert "Error calling Mistral API async:" in error_message


class TestEmbeddings:
    """Test embedding functions"""
    
    @patch('mistral_utils.get_mistral_client')
    def test_call_mistral_embeddings_single_text(self, mock_get_client):
        """Test embeddings with single text"""
        mock_client = Mock()
        mock_embedding_data = Mock()
        mock_embedding_data.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_response = Mock()
        mock_response.data = [mock_embedding_data]
        
        mock_client.embeddings.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        result = call_mistral_embeddings("Test text", model="mistral-embed")
        
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_client.embeddings.create.assert_called_once_with(
            model="mistral-embed",
            inputs=["Test text"]
        )
    
    @patch('mistral_utils.get_mistral_client')
    def test_call_mistral_embeddings_multiple_texts(self, mock_get_client):
        """Test embeddings with multiple texts"""
        mock_client = Mock()
        mock_embeddings = []
        for i, embedding in enumerate([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]):
            mock_data = Mock()
            mock_data.embedding = embedding
            mock_embeddings.append(mock_data)
        
        mock_response = Mock()
        mock_response.data = mock_embeddings
        
        mock_client.embeddings.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        texts = ["Text 1", "Text 2", "Text 3"]
        result = call_mistral_embeddings(texts)
        
        assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_client.embeddings.create.assert_called_once_with(
            model="mistral-embed",
            inputs=texts
        )
    
    @patch('mistral_utils.get_mistral_client')
    def test_call_mistral_embeddings_no_client(self, mock_get_client):
        """Test embeddings with no client"""
        mock_get_client.return_value = None
        
        result = call_mistral_embeddings("Test")
        assert result is None
    
    @patch('streamlit.error')
    @patch('mistral_utils.get_mistral_client')
    def test_call_mistral_embeddings_error(self, mock_get_client, mock_error):
        """Test embeddings error handling"""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("Embeddings error")
        mock_get_client.return_value = mock_client
        
        result = call_mistral_embeddings("Test")
        
        assert result is None
        mock_error.assert_called_once()
        error_message = mock_error.call_args[0][0]
        assert "Error calling Mistral Embeddings API:" in error_message


class TestUIFunctions:
    """Test UI-related functions"""
    
    @patch('streamlit.success')
    @patch('mistral_utils.save_api_keys')
    @patch('streamlit.sidebar.button')
    @patch('streamlit.sidebar.text_input')
    @patch('streamlit.sidebar.subheader')
    @patch('mistral_utils.load_api_keys')
    def test_display_mistral_settings_save_key(
        self, mock_load_keys, mock_subheader, mock_text_input,
        mock_button, mock_save_keys, mock_success
    ):
        """Test displaying Mistral settings and saving key"""
        # Setup mocks
        mock_load_keys.return_value = {"mistral_api_key": "existing-key"}
        mock_text_input.return_value = "new-mistral-key"
        mock_button.return_value = True  # Simulate button click
        
        # Call the function
        display_mistral_settings()
        
        # Verify calls
        mock_subheader.assert_called_once_with("Mistral API Key")
        mock_text_input.assert_called_once_with(
            "Enter your Mistral API key:",
            value="existing-key",
            type="password"
        )
        mock_button.assert_called_once_with("Save Mistral API Key")
        
        # Verify API key was saved
        expected_keys = {"mistral_api_key": "new-mistral-key"}
        mock_save_keys.assert_called_once_with(expected_keys)
        mock_success.assert_called_once_with("Mistral API key saved!")
    
    @patch('mistral_utils.save_api_keys')
    @patch('streamlit.sidebar.button')
    @patch('streamlit.sidebar.text_input')
    @patch('streamlit.sidebar.subheader')
    @patch('mistral_utils.load_api_keys')
    def test_display_mistral_settings_no_save(
        self, mock_load_keys, mock_subheader, mock_text_input,
        mock_button, mock_save_keys
    ):
        """Test displaying Mistral settings without saving"""
        # Setup mocks
        mock_load_keys.return_value = {}
        mock_text_input.return_value = "some-key"
        mock_button.return_value = False  # No button click
        
        # Call the function
        display_mistral_settings()
        
        # Verify text input was called with empty value
        mock_text_input.assert_called_once_with(
            "Enter your Mistral API key:",
            value="",  # No existing key
            type="password"
        )
        
        # save_api_keys should not be called
        mock_save_keys.assert_not_called()


class TestIntegration:
    """Integration tests for complete workflows"""
    
    @patch('mistral_utils.Mistral')
    def test_complete_api_flow(self, mock_mistral_class, tmp_path):
        """Test complete flow: save key, load, create client, and use API"""
        api_file = tmp_path / "api_keys.json"
        
        with patch('mistral_utils.API_KEYS_FILE', str(api_file)):
            # Save API key
            save_api_keys({"mistral_api_key": "msk_integration_test"})
            
            # Load and verify
            keys = load_api_keys()
            assert keys["mistral_api_key"] == "msk_integration_test"
            
            # Create client
            mock_client = Mock()
            mock_mistral_class.return_value = mock_client
            client = get_mistral_client(keys["mistral_api_key"])
            assert client == mock_client
            
            # Use API
            mock_message = Mock()
            mock_message.content = "Integration test response"
            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_client.chat.complete.return_value = mock_response
            
            result = call_mistral_api(
                model="mistral-large-latest",
                prompt="Test",
                mistral_api_key=keys["mistral_api_key"]
            )
            
            assert result == "Integration test response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])