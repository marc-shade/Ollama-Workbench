"""
Test suite for groq_utils.py - Groq provider integration
"""

import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock
import tempfile
import numpy as np

# Import the module to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ollama_workbench.providers.groq_utils import (
    GROQ_MODELS, load_embedding_model, load_api_keys, save_api_keys,
    get_groq_client, call_groq_api, get_local_embeddings, display_groq_settings
)


class TestConstants:
    """Test module constants"""
    
    def test_groq_models_list(self):
        """Test that GROQ_MODELS is properly defined"""
        assert isinstance(GROQ_MODELS, list)
        assert len(GROQ_MODELS) > 0
        
        # Check some expected models are present
        expected_models = ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
        for model in expected_models:
            assert model in GROQ_MODELS
        
        # All items should be strings
        assert all(isinstance(model, str) for model in GROQ_MODELS)


class TestAPIKeyManagement:
    """Test API key loading and saving functions"""
    
    def test_load_api_keys_exists(self, tmp_path):
        """Test loading API keys when file exists"""
        test_keys = {"groq_api_key": "gsk_test123", "other_key": "value"}
        api_file = tmp_path / "api_keys.json"
        with open(api_file, "w") as f:
            json.dump(test_keys, f)
        
        with patch('groq_utils.API_KEYS_FILE', str(api_file)):
            loaded_keys = load_api_keys()
            assert loaded_keys == test_keys
            assert loaded_keys["groq_api_key"] == "gsk_test123"
    
    def test_load_api_keys_not_exists(self):
        """Test loading API keys when file doesn't exist"""
        with patch('os.path.exists', return_value=False):
            loaded_keys = load_api_keys()
            assert loaded_keys == {}
    
    def test_save_api_keys(self, tmp_path):
        """Test saving API keys to file"""
        test_keys = {"groq_api_key": "gsk_test456", "another_key": "another_value"}
        api_file = tmp_path / "api_keys.json"
        
        with patch('groq_utils.API_KEYS_FILE', str(api_file)):
            save_api_keys(test_keys)
            
            # Verify file was saved correctly
            with open(api_file, "r") as f:
                saved_keys = json.load(f)
            assert saved_keys == test_keys
            assert saved_keys["groq_api_key"] == "gsk_test456"


class TestGroqClient:
    """Test Groq client initialization"""
    
    @patch('groq_utils.Groq')
    def test_get_groq_client_with_key(self, mock_groq_class):
        """Test getting Groq client with valid API key"""
        mock_client = Mock()
        mock_groq_class.return_value = mock_client
        
        client = get_groq_client("gsk_valid_key")
        
        assert client == mock_client
        mock_groq_class.assert_called_once_with(api_key="gsk_valid_key")
    
    def test_get_groq_client_no_key(self):
        """Test getting Groq client with no API key"""
        client = get_groq_client("")
        assert client is None
        
        client = get_groq_client(None)
        assert client is None
    
    @patch('streamlit.warning')
    @patch('groq_utils.Groq')
    def test_get_groq_client_error(self, mock_groq_class, mock_warning):
        """Test Groq client error handling"""
        mock_groq_class.side_effect = Exception("Invalid API key")
        
        client = get_groq_client("gsk_invalid_key")
        
        assert client is None
        mock_warning.assert_called_once_with(
            "Groq API key not configured. Some features will be limited to local models."
        )


class TestGroqAPI:
    """Test Groq API calling functions"""
    
    def test_call_groq_api_no_client(self):
        """Test API call with no client"""
        result = call_groq_api(None, "llama3-70b-8192", [{"role": "user", "content": "Test"}])
        assert result is None
    
    def test_call_groq_api_success(self):
        """Test successful Groq API call"""
        # Mock client and response
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = "Hello from Groq!"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_completion = Mock()
        mock_completion.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Test the function
        messages = [{"role": "user", "content": "Hello"}]
        result = call_groq_api(
            mock_client,
            "llama3-70b-8192",
            messages,
            temperature=0.7,
            max_tokens=100
        )
        
        assert result == "Hello from Groq!"
        mock_client.chat.completions.create.assert_called_once_with(
            model="llama3-70b-8192",
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
    
    @patch('streamlit.warning')
    def test_call_groq_api_error(self, mock_warning):
        """Test Groq API call error handling"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        result = call_groq_api(
            mock_client,
            "llama3-70b-8192",
            [{"role": "user", "content": "Test"}]
        )
        
        assert result is None
        mock_warning.assert_called_once()
        warning_message = mock_warning.call_args[0][0]
        assert "Error calling Groq API:" in warning_message
        assert "API Error" in warning_message


class TestEmbeddings:
    """Test embedding generation functions"""
    
    @patch('groq_utils.SentenceTransformer')
    def test_load_embedding_model_cached(self, mock_transformer_class):
        """Test that embedding model loading is cached"""
        # Mock the SentenceTransformer
        mock_model = Mock()
        mock_transformer_class.return_value = mock_model
        
        # Clear cache decorator for testing
        load_embedding_model.clear()
        
        # First call
        model1 = load_embedding_model()
        # Second call (should use cache)
        model2 = load_embedding_model()
        
        assert model1 == mock_model
        assert model2 == mock_model
        # Should only be called once due to caching
        mock_transformer_class.assert_called_once_with('all-MiniLM-L6-v2')
    
    @patch('groq_utils.SentenceTransformer')
    def test_load_embedding_model_loads_correct_model(self, mock_transformer_class):
        """Test that the correct embedding model is loaded"""
        mock_model = Mock()
        mock_transformer_class.return_value = mock_model
        
        # Clear cache and reload module to apply mock
        load_embedding_model.clear()
        
        with patch('groq_utils.st.cache_resource', lambda f: f):
            model = load_embedding_model()
        
        assert model == mock_model
        mock_transformer_class.assert_called_once_with('all-MiniLM-L6-v2')
    
    @patch('groq_utils.load_embedding_model')
    def test_get_local_embeddings(self, mock_load_model):
        """Test local embedding generation"""
        # Mock the model and its encode method
        mock_model = Mock()
        mock_embeddings = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_model.encode.return_value = mock_embeddings
        mock_load_model.return_value = mock_model
        
        # Test embedding generation
        embeddings = get_local_embeddings("Test text")
        
        assert embeddings == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_model.encode.assert_called_once_with("Test text")
    
    @patch('groq_utils.load_embedding_model')
    def test_get_local_embeddings_different_texts(self, mock_load_model):
        """Test embedding generation with different texts"""
        mock_model = Mock()
        
        # Mock different embeddings for different texts
        def encode_side_effect(text):
            if text == "Hello":
                return np.array([0.1, 0.2, 0.3])
            elif text == "World":
                return np.array([0.4, 0.5, 0.6])
            else:
                return np.array([0.0, 0.0, 0.0])
        
        mock_model.encode.side_effect = encode_side_effect
        mock_load_model.return_value = mock_model
        
        # Test different texts
        embeddings1 = get_local_embeddings("Hello")
        embeddings2 = get_local_embeddings("World")
        embeddings3 = get_local_embeddings("Unknown")
        
        assert embeddings1 == [0.1, 0.2, 0.3]
        assert embeddings2 == [0.4, 0.5, 0.6]
        assert embeddings3 == [0.0, 0.0, 0.0]


class TestUIFunctions:
    """Test UI-related functions"""
    
    @patch('streamlit.success')
    @patch('groq_utils.save_api_keys')
    @patch('streamlit.sidebar.button')
    @patch('streamlit.sidebar.text_input')
    @patch('streamlit.sidebar.subheader')
    @patch('groq_utils.load_api_keys')
    def test_display_groq_settings_save_key(
        self, mock_load_keys, mock_subheader, mock_text_input,
        mock_button, mock_save_keys, mock_success
    ):
        """Test displaying Groq settings and saving key"""
        # Setup mocks
        mock_load_keys.return_value = {"groq_api_key": "existing-key"}
        mock_text_input.return_value = "new-groq-key"
        mock_button.return_value = True  # Simulate button click
        
        # Call the function
        display_groq_settings()
        
        # Verify calls
        mock_subheader.assert_called_once_with("Groq API Key")
        mock_text_input.assert_called_once_with(
            "Enter your Groq API key:",
            value="existing-key",
            type="password"
        )
        mock_button.assert_called_once_with("Save Groq API Key")
        
        # Verify API key was saved
        expected_keys = {"groq_api_key": "new-groq-key"}
        mock_save_keys.assert_called_once_with(expected_keys)
        mock_success.assert_called_once_with("Groq API key saved!")
    
    @patch('groq_utils.save_api_keys')
    @patch('streamlit.sidebar.button')
    @patch('streamlit.sidebar.text_input')
    @patch('streamlit.sidebar.subheader')
    @patch('groq_utils.load_api_keys')
    def test_display_groq_settings_no_save(
        self, mock_load_keys, mock_subheader, mock_text_input,
        mock_button, mock_save_keys
    ):
        """Test displaying Groq settings without saving"""
        # Setup mocks
        mock_load_keys.return_value = {}
        mock_text_input.return_value = "some-key"
        mock_button.return_value = False  # No button click
        
        # Call the function
        display_groq_settings()
        
        # Verify text input was called with empty value
        mock_text_input.assert_called_once_with(
            "Enter your Groq API key:",
            value="",  # No existing key
            type="password"
        )
        
        # save_api_keys should not be called
        mock_save_keys.assert_not_called()


class TestIntegration:
    """Integration tests for complete workflows"""
    
    @patch('groq_utils.Groq')
    def test_complete_api_flow(self, mock_groq_class, tmp_path):
        """Test complete flow: save key, load, create client, and use API"""
        api_file = tmp_path / "api_keys.json"
        
        with patch('groq_utils.API_KEYS_FILE', str(api_file)):
            # Save API key
            save_api_keys({"groq_api_key": "gsk_integration_test"})
            
            # Load and verify
            keys = load_api_keys()
            assert keys["groq_api_key"] == "gsk_integration_test"
            
            # Create client
            mock_client = Mock()
            mock_groq_class.return_value = mock_client
            client = get_groq_client(keys["groq_api_key"])
            assert client == mock_client
            
            # Use API
            mock_message = Mock()
            mock_message.content = "Integration test response"
            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_completion = Mock()
            mock_completion.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_completion
            
            result = call_groq_api(
                client,
                "llama3-70b-8192",
                [{"role": "user", "content": "Test"}]
            )
            
            assert result == "Integration test response"
    
    @patch('groq_utils.SentenceTransformer')
    def test_embedding_flow(self, mock_transformer_class):
        """Test embedding generation flow"""
        # Mock the transformer
        mock_model = Mock()
        mock_embeddings = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_model.encode.return_value = mock_embeddings
        mock_transformer_class.return_value = mock_model
        
        # Clear cache and use no-op decorator
        load_embedding_model.clear()
        with patch('groq_utils.st.cache_resource', lambda f: f):
            # Generate embeddings
            embeddings = get_local_embeddings("Test embedding generation")
        
        assert embeddings == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_model.encode.assert_called_with("Test embedding generation")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])