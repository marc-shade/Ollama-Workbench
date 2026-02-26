"""
Comprehensive tests for external_providers.py module.

Tests all external provider functionality including:
- API key management UI components
- Provider configuration handling
- Model availability based on user settings
- Integration with ollama_utils functions
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock, call
from unittest import TestCase
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ollama_workbench.providers.external_providers import (
    ADVANCED_GROQ_MODELS,
    external_providers_ui,
    get_available_groq_models
)


def create_column_mocks():
    """Helper function to create proper mock columns that support context manager protocol"""
    col1_mock = Mock()
    col1_mock.__enter__ = Mock(return_value=col1_mock)
    col1_mock.__exit__ = Mock(return_value=None)
    
    col2_mock = Mock()
    col2_mock.__enter__ = Mock(return_value=col2_mock)
    col2_mock.__exit__ = Mock(return_value=None)
    
    return [col1_mock, col2_mock]


class TestConstants(TestCase):
    """Test module constants"""
    
    def test_advanced_groq_models_list(self):
        """Test that ADVANCED_GROQ_MODELS contains expected models"""
        expected_models = [
            "llama-3.1-405b-reasoning",
            "llama-3.1-70b-versatile", 
            "llama-3.1-8b-instant",
        ]
        
        self.assertEqual(ADVANCED_GROQ_MODELS, expected_models)
        self.assertEqual(len(ADVANCED_GROQ_MODELS), 3)
    
    def test_advanced_groq_models_types(self):
        """Test that all advanced models are strings"""
        for model in ADVANCED_GROQ_MODELS:
            self.assertIsInstance(model, str)
            self.assertGreater(len(model), 0)


class TestGetAvailableGroqModels(TestCase):
    """Test get_available_groq_models function"""
    
    def test_base_models_only(self):
        """Test getting base models when advanced models not enabled"""
        api_keys = {"use_advanced_groq_models": False}
        
        result = get_available_groq_models(api_keys)
        
        expected_base_models = [
            "llama3-8b-8192",
            "llama3-70b-8192",
            "llama3-groq-8b-8192-tool-use-preview",
            "llama3-groq-70b-8192-tool-use-preview",
            "mixtral-8x7b-32768",
            "gemma-7b-it",
            "gemma2-9b-it",
        ]
        
        self.assertEqual(result, expected_base_models)
        
        # Ensure no advanced models are included
        for advanced_model in ADVANCED_GROQ_MODELS:
            self.assertNotIn(advanced_model, result)
    
    def test_base_models_when_key_missing(self):
        """Test getting base models when advanced key is missing from api_keys"""
        api_keys = {}  # Missing use_advanced_groq_models key
        
        result = get_available_groq_models(api_keys)
        
        expected_base_models = [
            "llama3-8b-8192",
            "llama3-70b-8192",
            "llama3-groq-8b-8192-tool-use-preview",
            "llama3-groq-70b-8192-tool-use-preview",
            "mixtral-8x7b-32768",
            "gemma-7b-it",
            "gemma2-9b-it",
        ]
        
        self.assertEqual(result, expected_base_models)
    
    def test_all_models_when_advanced_enabled(self):
        """Test getting all models when advanced models are enabled"""
        api_keys = {"use_advanced_groq_models": True}
        
        result = get_available_groq_models(api_keys)
        
        expected_base_models = [
            "llama3-8b-8192",
            "llama3-70b-8192",
            "llama3-groq-8b-8192-tool-use-preview",
            "llama3-groq-70b-8192-tool-use-preview",
            "mixtral-8x7b-32768",
            "gemma-7b-it",
            "gemma2-9b-it",
        ]
        
        # Should contain all base models
        for base_model in expected_base_models:
            self.assertIn(base_model, result)
        
        # Should contain all advanced models
        for advanced_model in ADVANCED_GROQ_MODELS:
            self.assertIn(advanced_model, result)
        
        # Should have the correct total count
        expected_total = len(expected_base_models) + len(ADVANCED_GROQ_MODELS)
        self.assertEqual(len(result), expected_total)
    
    def test_model_order_preserved(self):
        """Test that model order is preserved (base models first, then advanced)"""
        api_keys = {"use_advanced_groq_models": True}
        
        result = get_available_groq_models(api_keys)
        
        expected_base_models = [
            "llama3-8b-8192",
            "llama3-70b-8192",
            "llama3-groq-8b-8192-tool-use-preview",
            "llama3-groq-70b-8192-tool-use-preview",
            "mixtral-8x7b-32768",
            "gemma-7b-it",
            "gemma2-9b-it",
        ]
        
        # Check that base models come first in the correct order
        for i, expected_model in enumerate(expected_base_models):
            self.assertEqual(result[i], expected_model)
        
        # Check that advanced models come after base models
        base_count = len(expected_base_models)
        for i, advanced_model in enumerate(ADVANCED_GROQ_MODELS):
            self.assertEqual(result[base_count + i], advanced_model)
    
    def test_empty_api_keys(self):
        """Test function with empty api_keys dictionary"""
        api_keys = {}
        
        result = get_available_groq_models(api_keys)
        
        # Should return only base models
        self.assertEqual(len(result), 7)
        self.assertNotIn("llama-3.1-405b-reasoning", result)
    
    def test_none_api_keys(self):
        """Test function with None as api_keys"""
        # This should handle the case gracefully
        try:
            result = get_available_groq_models(None)
            # If it doesn't crash, check that it returns base models
            self.assertIsInstance(result, list)
        except AttributeError:
            # This is also acceptable behavior
            pass


class TestExternalProvidersUI(TestCase):
    """Test external_providers_ui function"""
    
    def setUp(self):
        """Set up common test data"""
        self.sample_api_keys = {
            "serpapi_api_key": "test_serpapi_key",
            "serper_api_key": "test_serper_key",
            "google_api_key": "test_google_key",
            "google_cse_id": "test_google_cse_id",
            "bing_api_key": "test_bing_key",
            "openai_api_key": "test_openai_key",
            "groq_api_key": "test_groq_key",
            "mistral_api_key": "test_mistral_key",
            "use_advanced_groq_models": True
        }
    
    def create_column_mocks(self):
        """Create proper mock columns that support context manager protocol"""
        col1_mock = Mock()
        col1_mock.__enter__ = Mock(return_value=col1_mock)
        col1_mock.__exit__ = Mock(return_value=None)
        
        col2_mock = Mock()
        col2_mock.__enter__ = Mock(return_value=col2_mock)
        col2_mock.__exit__ = Mock(return_value=None)
        
        return [col1_mock, col2_mock]
    
    @patch('streamlit.success')
    @patch('streamlit.button')
    @patch('streamlit.checkbox')
    @patch('streamlit.text_input')
    @patch('streamlit.header')
    @patch('streamlit.columns')
    @patch('streamlit.title')
    @patch('external_providers.save_api_keys')
    @patch('external_providers.load_api_keys')
    def test_external_providers_ui_basic_layout(self, mock_load_keys, mock_save_keys,
                                               mock_title, mock_columns, mock_header,
                                               mock_text_input, mock_checkbox, mock_button,
                                               mock_success):
        """Test that the UI creates the basic layout components"""
        mock_load_keys.return_value = self.sample_api_keys
        mock_columns.return_value = self.create_column_mocks()
        mock_button.return_value = False  # Button not pressed
        mock_text_input.return_value = ""
        mock_checkbox.return_value = False
        
        external_providers_ui()
        
        # Verify basic UI structure
        mock_title.assert_called_once_with("☁️ External Providers")
        mock_columns.assert_called_once_with(2)
        
        # Verify headers are created
        header_calls = mock_header.call_args_list
        self.assertEqual(len(header_calls), 2)
        self.assertIn(call("Search Providers"), header_calls)
        self.assertIn(call("AI Model Providers"), header_calls)
    
    @patch('streamlit.success')
    @patch('streamlit.button')
    @patch('streamlit.checkbox')
    @patch('streamlit.text_input')
    @patch('streamlit.header')
    @patch('streamlit.columns')
    @patch('streamlit.title')
    @patch('external_providers.save_api_keys')
    @patch('external_providers.load_api_keys')
    def test_external_providers_ui_search_provider_inputs(self, mock_load_keys, mock_save_keys,
                                                         mock_title, mock_columns, mock_header,
                                                         mock_text_input, mock_checkbox, mock_button,
                                                         mock_success):
        """Test that search provider inputs are created correctly"""
        mock_load_keys.return_value = self.sample_api_keys
        mock_columns.return_value = self.create_column_mocks()
        mock_button.return_value = False
        mock_text_input.return_value = ""
        mock_checkbox.return_value = False
        
        external_providers_ui()
        
        # Check that all search provider text inputs are created
        text_input_calls = mock_text_input.call_args_list
        
        # Extract just the labels from the calls
        input_labels = [call[0][0] for call in text_input_calls]
        
        expected_search_inputs = [
            "SerpApi API Key",
            "Serper API Key", 
            "Google Custom Search API Key",
            "Google Custom Search Engine ID",
            "Bing Search API Key"
        ]
        
        for expected_input in expected_search_inputs:
            self.assertIn(expected_input, input_labels)
    
    @patch('streamlit.success')
    @patch('streamlit.button')
    @patch('streamlit.checkbox')
    @patch('streamlit.text_input')
    @patch('streamlit.header')
    @patch('streamlit.columns')
    @patch('streamlit.title')
    @patch('external_providers.save_api_keys')
    @patch('external_providers.load_api_keys')
    def test_external_providers_ui_ai_provider_inputs(self, mock_load_keys, mock_save_keys,
                                                     mock_title, mock_columns, mock_header,
                                                     mock_text_input, mock_checkbox, mock_button,
                                                     mock_success):
        """Test that AI provider inputs are created correctly"""
        mock_load_keys.return_value = self.sample_api_keys
        mock_columns.return_value = self.create_column_mocks()
        mock_button.return_value = False
        mock_text_input.return_value = ""
        mock_checkbox.return_value = False
        
        external_providers_ui()
        
        # Check that all AI provider text inputs are created
        text_input_calls = mock_text_input.call_args_list
        input_labels = [call[0][0] for call in text_input_calls]
        
        expected_ai_inputs = [
            "OpenAI API Key",
            "Groq API Key",
            "Mistral API Key"
        ]
        
        for expected_input in expected_ai_inputs:
            self.assertIn(expected_input, input_labels)
        
        # Check that checkbox for advanced Groq models is created
        mock_checkbox.assert_called_once_with(
            "Enable Advanced Groq Models",
            value=True,  # From sample_api_keys
            help="Check this box if your Groq account is approved to use advanced models."
        )
    
    @patch('streamlit.success')
    @patch('streamlit.button')
    @patch('streamlit.checkbox')
    @patch('streamlit.text_input')
    @patch('streamlit.header')
    @patch('streamlit.columns')
    @patch('streamlit.title')
    @patch('external_providers.save_api_keys')
    @patch('external_providers.load_api_keys')
    def test_external_providers_ui_password_input_type(self, mock_load_keys, mock_save_keys,
                                                      mock_title, mock_columns, mock_header,
                                                      mock_text_input, mock_checkbox, mock_button,
                                                      mock_success):
        """Test that all text inputs are password type for security"""
        mock_load_keys.return_value = self.sample_api_keys
        mock_columns.return_value = self.create_column_mocks()
        mock_button.return_value = False
        mock_text_input.return_value = ""
        mock_checkbox.return_value = False
        
        external_providers_ui()
        
        # Check that all text_input calls use type="password"
        text_input_calls = mock_text_input.call_args_list
        
        for call in text_input_calls:
            kwargs = call[1]  # Get keyword arguments
            self.assertEqual(kwargs.get("type"), "password")
    
    @patch('streamlit.success')
    @patch('streamlit.button')
    @patch('streamlit.checkbox')
    @patch('streamlit.text_input')
    @patch('streamlit.header')
    @patch('streamlit.columns')
    @patch('streamlit.title')
    @patch('external_providers.save_api_keys')
    @patch('external_providers.load_api_keys')
    def test_external_providers_ui_default_values(self, mock_load_keys, mock_save_keys,
                                                 mock_title, mock_columns, mock_header,
                                                 mock_text_input, mock_checkbox, mock_button,
                                                 mock_success):
        """Test that inputs use values from loaded API keys"""
        mock_load_keys.return_value = self.sample_api_keys
        mock_columns.return_value = self.create_column_mocks()
        mock_button.return_value = False
        mock_text_input.return_value = ""
        mock_checkbox.return_value = False
        
        external_providers_ui()
        
        # Check that text inputs get default values from api_keys
        text_input_calls = mock_text_input.call_args_list
        
        # Find the SerpApi call
        serpapi_call = next(call for call in text_input_calls if call[0][0] == "SerpApi API Key")
        self.assertEqual(serpapi_call[1]["value"], "test_serpapi_key")
        
        # Find the OpenAI call
        openai_call = next(call for call in text_input_calls if call[0][0] == "OpenAI API Key")
        self.assertEqual(openai_call[1]["value"], "test_openai_key")
        
        # Check checkbox default value
        mock_checkbox.assert_called_once_with(
            "Enable Advanced Groq Models",
            value=True,  # From sample_api_keys
            help="Check this box if your Groq account is approved to use advanced models."
        )
    
    @patch('streamlit.success')
    @patch('streamlit.button')
    @patch('streamlit.checkbox')
    @patch('streamlit.text_input')
    @patch('streamlit.header')
    @patch('streamlit.columns')
    @patch('streamlit.title')
    @patch('external_providers.save_api_keys')
    @patch('external_providers.load_api_keys')
    def test_external_providers_ui_save_button_not_pressed(self, mock_load_keys, mock_save_keys,
                                                          mock_title, mock_columns, mock_header,
                                                          mock_text_input, mock_checkbox, mock_button,
                                                          mock_success):
        """Test behavior when save button is not pressed"""
        mock_load_keys.return_value = self.sample_api_keys
        mock_columns.return_value = self.create_column_mocks()
        mock_button.return_value = False  # Button not pressed
        mock_text_input.return_value = ""
        mock_checkbox.return_value = False
        
        external_providers_ui()
        
        # Button should be created
        mock_button.assert_called_once_with("💾 Save API Keys")
        
        # But save_api_keys should not be called
        mock_save_keys.assert_not_called()
        
        # And success message should not be shown
        mock_success.assert_not_called()
    
    @patch('streamlit.success')
    @patch('streamlit.button')
    @patch('streamlit.checkbox')
    @patch('streamlit.text_input')
    @patch('streamlit.header')
    @patch('streamlit.columns')
    @patch('streamlit.title')
    @patch('external_providers.save_api_keys')
    @patch('external_providers.load_api_keys')
    def test_external_providers_ui_save_button_pressed(self, mock_load_keys, mock_save_keys,
                                                      mock_title, mock_columns, mock_header,
                                                      mock_text_input, mock_checkbox, mock_button,
                                                      mock_success):
        """Test behavior when save button is pressed"""
        mock_load_keys.return_value = {}  # Start with empty keys
        mock_columns.return_value = self.create_column_mocks()
        mock_button.return_value = True  # Button pressed
        
        # Mock the input values
        def text_input_side_effect(label, value="", type="text"):
            input_values = {
                "SerpApi API Key": "new_serpapi_key",
                "Serper API Key": "new_serper_key",
                "Google Custom Search API Key": "new_google_key",
                "Google Custom Search Engine ID": "new_google_cse_id",
                "Bing Search API Key": "new_bing_key",
                "OpenAI API Key": "new_openai_key",
                "Groq API Key": "new_groq_key",
                "Mistral API Key": "new_mistral_key"
            }
            return input_values.get(label, "")
        
        mock_text_input.side_effect = text_input_side_effect
        mock_checkbox.return_value = True  # Advanced models enabled
        
        external_providers_ui()
        
        # save_api_keys should be called
        mock_save_keys.assert_called_once()
        
        # Check the saved data structure
        saved_data = mock_save_keys.call_args[0][0]
        expected_keys = [
            "serpapi_api_key", "serper_api_key", "google_api_key", 
            "google_cse_id", "bing_api_key", "openai_api_key",
            "groq_api_key", "mistral_api_key", "use_advanced_groq_models"
        ]
        
        for key in expected_keys:
            self.assertIn(key, saved_data)
        
        # Check specific values
        self.assertEqual(saved_data["serpapi_api_key"], "new_serpapi_key")
        self.assertEqual(saved_data["openai_api_key"], "new_openai_key")
        self.assertTrue(saved_data["use_advanced_groq_models"])
        
        # Success message should be shown
        mock_success.assert_called_once_with("🟢 API keys saved!")
    
    @patch('streamlit.success')
    @patch('streamlit.button')
    @patch('streamlit.checkbox')
    @patch('streamlit.text_input')
    @patch('streamlit.header')
    @patch('streamlit.columns')
    @patch('streamlit.title')
    @patch('external_providers.save_api_keys')
    @patch('external_providers.load_api_keys')
    def test_external_providers_ui_empty_api_keys(self, mock_load_keys, mock_save_keys,
                                                 mock_title, mock_columns, mock_header,
                                                 mock_text_input, mock_checkbox, mock_button,
                                                 mock_success):
        """Test handling when load_api_keys returns empty dictionary"""
        mock_load_keys.return_value = {}  # Empty API keys
        mock_columns.return_value = self.create_column_mocks()
        mock_button.return_value = False
        mock_text_input.return_value = ""
        mock_checkbox.return_value = False
        
        external_providers_ui()
        
        # Should handle empty keys gracefully
        text_input_calls = mock_text_input.call_args_list
        
        # All text inputs should have empty default values
        for call in text_input_calls:
            kwargs = call[1]
            self.assertEqual(kwargs.get("value"), "")
        
        # Checkbox should default to False
        mock_checkbox.assert_called_once_with(
            "Enable Advanced Groq Models",
            value=False,  # Default when key is missing
            help="Check this box if your Groq account is approved to use advanced models."
        )


class TestUIIntegration(TestCase):
    """Test integration scenarios and edge cases"""
    
    @patch('external_providers.load_api_keys')
    def test_load_api_keys_integration(self, mock_load_keys):
        """Test that load_api_keys is called during UI initialization"""
        mock_load_keys.return_value = {}
        
        with patch('streamlit.title'), \
             patch('streamlit.columns', return_value=create_column_mocks()), \
             patch('streamlit.header'), \
             patch('streamlit.text_input', return_value=""), \
             patch('streamlit.checkbox', return_value=False), \
             patch('streamlit.button', return_value=False):
            
            external_providers_ui()
        
        mock_load_keys.assert_called_once()
    
    @patch('external_providers.save_api_keys')
    @patch('external_providers.load_api_keys')
    def test_save_api_keys_integration(self, mock_load_keys, mock_save_keys):
        """Test that save_api_keys is called with correct data when button pressed"""
        mock_load_keys.return_value = {"existing_key": "existing_value"}
        
        with patch('streamlit.title'), \
             patch('streamlit.columns', return_value=create_column_mocks()), \
             patch('streamlit.header'), \
             patch('streamlit.text_input', return_value="test_value"), \
             patch('streamlit.checkbox', return_value=True), \
             patch('streamlit.button', return_value=True), \
             patch('streamlit.success'):
            
            external_providers_ui()
        
        mock_save_keys.assert_called_once()
        
        # Verify the saved data contains expected keys
        saved_data = mock_save_keys.call_args[0][0]
        required_keys = [
            "serpapi_api_key", "serper_api_key", "google_api_key",
            "google_cse_id", "bing_api_key", "openai_api_key", 
            "groq_api_key", "mistral_api_key", "use_advanced_groq_models"
        ]
        
        for key in required_keys:
            self.assertIn(key, saved_data)
    
    def test_model_availability_consistency(self):
        """Test that model availability is consistent across different settings"""
        # Test with advanced models disabled
        base_only = get_available_groq_models({"use_advanced_groq_models": False})
        
        # Test with advanced models enabled  
        all_models = get_available_groq_models({"use_advanced_groq_models": True})
        
        # All base models should be in both lists
        for model in base_only:
            self.assertIn(model, all_models)
        
        # Advanced models should only be in the full list
        for advanced_model in ADVANCED_GROQ_MODELS:
            self.assertNotIn(advanced_model, base_only)
            self.assertIn(advanced_model, all_models)
        
        # The full list should be larger
        self.assertGreater(len(all_models), len(base_only))
        self.assertEqual(len(all_models), len(base_only) + len(ADVANCED_GROQ_MODELS))


class TestErrorHandling(TestCase):
    """Test error handling and edge cases"""
    
    @patch('external_providers.load_api_keys')
    def test_load_api_keys_exception_handling(self, mock_load_keys):
        """Test handling when load_api_keys raises an exception"""
        mock_load_keys.side_effect = Exception("API key loading failed")
        
        # The UI should handle exceptions gracefully
        with patch('streamlit.title'), \
             patch('streamlit.columns', return_value=create_column_mocks()), \
             patch('streamlit.header'), \
             patch('streamlit.text_input', return_value=""), \
             patch('streamlit.checkbox', return_value=False), \
             patch('streamlit.button', return_value=False):
            
            # Should raise an exception since the function doesn't handle it
            with self.assertRaises(Exception) as context:
                external_providers_ui()
            
            self.assertIn("API key loading failed", str(context.exception))
    
    @patch('external_providers.save_api_keys')
    @patch('external_providers.load_api_keys')
    def test_save_api_keys_exception_handling(self, mock_load_keys, mock_save_keys):
        """Test handling when save_api_keys raises an exception"""
        mock_load_keys.return_value = {}
        mock_save_keys.side_effect = Exception("API key saving failed")
        
        with patch('streamlit.title'), \
             patch('streamlit.columns', return_value=create_column_mocks()), \
             patch('streamlit.header'), \
             patch('streamlit.text_input', return_value="test"), \
             patch('streamlit.checkbox', return_value=False), \
             patch('streamlit.button', return_value=True), \
             patch('streamlit.success') as mock_success:
            
            # Should raise an exception since the function doesn't handle it
            with self.assertRaises(Exception) as context:
                external_providers_ui()
            
            self.assertIn("API key saving failed", str(context.exception))
            # Success should not be called since exception occurred
            mock_success.assert_not_called()
    
    def test_get_available_groq_models_edge_cases(self):
        """Test edge cases for get_available_groq_models"""
        # Test with various edge case inputs
        edge_cases = [
            ({"use_advanced_groq_models": None}, False),
            ({"use_advanced_groq_models": "true"}, True),   # String "true" is truthy
            ({"use_advanced_groq_models": 1}, True),        # Integer 1 is truthy
            ({"use_advanced_groq_models": []}, False),      # Empty list is falsy
            ({"other_key": True}, False),                  # Wrong key name
            ({"use_advanced_groq_models": 0}, False),      # Zero is falsy
            ({"use_advanced_groq_models": ""}, False),     # Empty string is falsy
            ({"use_advanced_groq_models": False}, False),  # Explicit False
        ]
        
        for api_keys, should_include_advanced in edge_cases:
            result = get_available_groq_models(api_keys)
            
            # Should return base models for any case
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)
            
            # Check if advanced models are included based on truthiness
            if should_include_advanced:
                for advanced_model in ADVANCED_GROQ_MODELS:
                    self.assertIn(advanced_model, result, 
                                f"Advanced model {advanced_model} should be included for api_keys: {api_keys}")
            else:
                for advanced_model in ADVANCED_GROQ_MODELS:
                    self.assertNotIn(advanced_model, result,
                                   f"Advanced model {advanced_model} should NOT be included for api_keys: {api_keys}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
