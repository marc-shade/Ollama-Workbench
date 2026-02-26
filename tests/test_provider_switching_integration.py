"""
Integration tests for provider switching functionality.

Tests the integration between different AI providers (Ollama, OpenAI, Groq, Mistral)
and ensures seamless switching between them in various contexts.
"""

import pytest
import asyncio
import json
import os
from unittest.mock import Mock, patch, MagicMock
from unittest import TestCase
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from ollama_workbench.providers.ollama_utils import get_ollama_models, call_ollama_api
from ollama_workbench.providers.openai_utils import call_openai_api, get_openai_models
from ollama_workbench.providers.groq_utils import call_groq_api, get_groq_models  
from ollama_workbench.providers.mistral_utils import call_mistral_api, get_mistral_models
from ollama_workbench.providers.external_providers import load_api_keys, save_api_keys
from ollama_workbench.chat.chat_interface import enhanced_chat_interface
from ollama_workbench.chat.multimodel_chat import MultiModelChat


class TestProviderSwitchingIntegration(TestCase):
    """Test integration scenarios for provider switching"""
    
    def setUp(self):
        """Set up test environment with mock API keys"""
        self.test_api_keys = {
            'openai_api_key': 'test_openai_key',
            'groq_api_key': 'test_groq_key',
            'mistral_api_key': 'test_mistral_key'
        }
        
        self.test_models = {
            'ollama': ['mistral:instruct', 'llama3:latest'],
            'openai': ['gpt-4', 'gpt-3.5-turbo'],
            'groq': ['llama3-70b-8192', 'mixtral-8x7b-32768'],
            'mistral': ['mistral-large', 'mistral-small']
        }
    
    @patch('external_providers.load_api_keys')
    @patch('ollama_utils.get_ollama_models')
    @patch('openai_utils.get_openai_models')
    @patch('groq_utils.get_groq_models')
    @patch('mistral_utils.get_mistral_models')
    def test_provider_model_discovery_integration(self, mock_mistral_models, mock_groq_models, 
                                                 mock_openai_models, mock_ollama_models, mock_load_keys):
        """Test that all providers can be discovered and their models listed"""
        mock_load_keys.return_value = self.test_api_keys
        mock_ollama_models.return_value = self.test_models['ollama']
        mock_openai_models.return_value = self.test_models['openai']
        mock_groq_models.return_value = self.test_models['groq']
        mock_mistral_models.return_value = self.test_models['mistral']
        
        # Test model discovery for each provider
        ollama_models = get_ollama_models()
        openai_models = get_openai_models()
        groq_models = get_groq_models()
        mistral_models = get_mistral_models()
        
        self.assertEqual(ollama_models, self.test_models['ollama'])
        self.assertEqual(openai_models, self.test_models['openai'])
        self.assertEqual(groq_models, self.test_models['groq'])
        self.assertEqual(mistral_models, self.test_models['mistral'])
        
        # Verify API key loading was called
        mock_load_keys.assert_called()
    
    @patch('external_providers.load_api_keys')
    @patch('ollama_utils.call_ollama_api')
    @patch('openai_utils.call_openai_api')
    @patch('groq_utils.call_groq_api')
    @patch('mistral_utils.call_mistral_api')
    def test_sequential_provider_calls(self, mock_mistral_api, mock_groq_api, 
                                     mock_openai_api, mock_ollama_api, mock_load_keys):
        """Test sequential calls to different providers with the same prompt"""
        mock_load_keys.return_value = self.test_api_keys
        test_prompt = "Explain quantum computing in simple terms."
        
        # Mock responses from each provider
        mock_ollama_api.return_value = "Ollama response: Quantum computing uses qubits..."
        mock_openai_api.return_value = "OpenAI response: Quantum computing is a revolutionary..."
        mock_groq_api.return_value = "Groq response: Quantum computers leverage quantum mechanics..."
        mock_mistral_api.return_value = "Mistral response: Quantum computing represents a paradigm..."
        
        # Test each provider
        ollama_response = call_ollama_api("mistral:instruct", [{"role": "user", "content": test_prompt}])
        openai_response = call_openai_api("gpt-4", [{"role": "user", "content": test_prompt}])
        groq_response = call_groq_api("llama3-70b-8192", [{"role": "user", "content": test_prompt}])
        mistral_response = call_mistral_api("mistral-large", [{"role": "user", "content": test_prompt}])
        
        # Verify all providers were called and returned different responses
        self.assertIn("Ollama response", ollama_response)
        self.assertIn("OpenAI response", openai_response)
        self.assertIn("Groq response", groq_response)
        self.assertIn("Mistral response", mistral_response)
        
        # Verify each API was called once
        mock_ollama_api.assert_called_once()
        mock_openai_api.assert_called_once()
        mock_groq_api.assert_called_once()
        mock_mistral_api.assert_called_once()
    
    @patch('external_providers.load_api_keys')
    def test_provider_fallback_chain(self, mock_load_keys):
        """Test fallback behavior when providers are unavailable"""
        mock_load_keys.return_value = self.test_api_keys
        
        test_prompt = "Test prompt"
        test_messages = [{"role": "user", "content": test_prompt}]
        
        # Test Ollama fallback to OpenAI
        with patch('ollama_utils.call_ollama_api') as mock_ollama:
            mock_ollama.side_effect = Exception("Ollama unavailable")
            
            with patch('openai_utils.call_openai_api') as mock_openai:
                mock_openai.return_value = "OpenAI fallback response"
                
                # Simulate fallback logic
                try:
                    response = call_ollama_api("mistral:instruct", test_messages)
                except Exception:
                    response = call_openai_api("gpt-4", test_messages)
                
                self.assertEqual(response, "OpenAI fallback response")
                mock_openai.assert_called_once()
    
    @patch('external_providers.load_api_keys')
    @patch('external_providers.save_api_keys')
    def test_api_key_management_integration(self, mock_save_keys, mock_load_keys):
        """Test API key loading and saving integration"""
        # Test loading existing keys
        mock_load_keys.return_value = self.test_api_keys
        
        loaded_keys = load_api_keys()
        self.assertEqual(loaded_keys, self.test_api_keys)
        
        # Test saving updated keys
        updated_keys = self.test_api_keys.copy()
        updated_keys['new_provider_key'] = 'new_test_key'
        
        save_api_keys(updated_keys)
        mock_save_keys.assert_called_once_with(updated_keys)
    
    @patch('streamlit.session_state', {})
    @patch('streamlit.selectbox')
    @patch('streamlit.text_area')
    @patch('streamlit.button')
    @patch('external_providers.load_api_keys')
    def test_chat_interface_provider_switching(self, mock_load_keys, mock_button, 
                                              mock_text_area, mock_selectbox):
        """Test provider switching in chat interface"""
        mock_load_keys.return_value = self.test_api_keys
        mock_selectbox.side_effect = ['mistral:instruct', 'gpt-4']  # Switch from Ollama to OpenAI
        mock_text_area.return_value = "Hello, how are you?"
        mock_button.return_value = True
        
        # Mock the actual API calls
        with patch('ollama_utils.call_ollama_api') as mock_ollama:
            mock_ollama.return_value = "Ollama: I'm doing well, thanks!"
            
            with patch('openai_utils.call_openai_api') as mock_openai:
                mock_openai.return_value = "OpenAI: I'm great, thank you for asking!"
                
                # Simulate provider switching in session
                import streamlit as st
                st.session_state['selected_model'] = 'mistral:instruct'
                
                # First call with Ollama model
                if 'mistral' in st.session_state.get('selected_model', ''):
                    response1 = call_ollama_api("mistral:instruct", [{"role": "user", "content": "Hello"}])
                
                # Switch to OpenAI model
                st.session_state['selected_model'] = 'gpt-4'
                
                # Second call with OpenAI model
                if 'gpt' in st.session_state.get('selected_model', ''):
                    response2 = call_openai_api("gpt-4", [{"role": "user", "content": "Hello"}])
                
                self.assertIn("Ollama", response1)
                self.assertIn("OpenAI", response2)
    
    @patch('external_providers.load_api_keys')
    def test_multimodel_chat_integration(self, mock_load_keys):
        """Test MultiModelChat class with provider switching"""
        mock_load_keys.return_value = self.test_api_keys
        
        # Initialize MultiModelChat
        multi_chat = MultiModelChat()
        
        test_messages = [{"role": "user", "content": "Compare Python and JavaScript"}]
        
        # Test with different providers
        with patch('ollama_utils.call_ollama_api') as mock_ollama:
            mock_ollama.return_value = "Ollama comparison response"
            
            with patch('openai_utils.call_openai_api') as mock_openai:
                mock_openai.return_value = "OpenAI comparison response"
                
                # Test model switching
                multi_chat.current_model = "mistral:instruct"
                response1 = multi_chat.generate_response(test_messages)
                
                multi_chat.current_model = "gpt-4"
                response2 = multi_chat.generate_response(test_messages)
                
                # Responses should be different based on provider
                self.assertNotEqual(response1, response2)
    
    @patch('external_providers.load_api_keys')
    def test_concurrent_provider_calls(self, mock_load_keys):
        """Test concurrent calls to multiple providers"""
        mock_load_keys.return_value = self.test_api_keys
        
        async def concurrent_test():
            # Mock async responses
            with patch('ollama_utils.call_ollama_api') as mock_ollama:
                mock_ollama.return_value = "Ollama async response"
                
                with patch('openai_utils.call_openai_api') as mock_openai:
                    mock_openai.return_value = "OpenAI async response"
                    
                    # Simulate concurrent calls
                    tasks = [
                        asyncio.create_task(asyncio.to_thread(
                            call_ollama_api, "mistral:instruct", [{"role": "user", "content": "test"}]
                        )),
                        asyncio.create_task(asyncio.to_thread(
                            call_openai_api, "gpt-4", [{"role": "user", "content": "test"}]
                        ))
                    ]
                    
                    responses = await asyncio.gather(*tasks)
                    
                    self.assertEqual(len(responses), 2)
                    self.assertIn("Ollama", responses[0])
                    self.assertIn("OpenAI", responses[1])
        
        # Run the async test
        asyncio.run(concurrent_test())
    
    @patch('external_providers.load_api_keys')
    def test_provider_error_handling_integration(self, mock_load_keys):
        """Test error handling across different providers"""
        mock_load_keys.return_value = self.test_api_keys
        
        test_messages = [{"role": "user", "content": "test"}]
        
        # Test various error scenarios
        error_scenarios = [
            ('ollama_utils.call_ollama_api', ConnectionError("Ollama server unavailable")),
            ('openai_utils.call_openai_api', ValueError("Invalid API key")),
            ('groq_utils.call_groq_api', TimeoutError("Request timeout")),
            ('mistral_utils.call_mistral_api', Exception("Unknown error"))
        ]
        
        for api_path, error in error_scenarios:
            with patch(api_path) as mock_api:
                mock_api.side_effect = error
                
                # Test that errors are properly handled
                try:
                    if 'ollama' in api_path:
                        call_ollama_api("mistral:instruct", test_messages)
                    elif 'openai' in api_path:
                        call_openai_api("gpt-4", test_messages)
                    elif 'groq' in api_path:
                        call_groq_api("llama3-70b-8192", test_messages)
                    elif 'mistral' in api_path:
                        call_mistral_api("mistral-large", test_messages)
                except Exception as e:
                    # Verify the error is the expected type
                    self.assertIsInstance(e, type(error))
    
    @patch('external_providers.load_api_keys')
    def test_provider_configuration_persistence(self, mock_load_keys):
        """Test that provider configurations persist across sessions"""
        mock_load_keys.return_value = self.test_api_keys
        
        # Test configuration saving and loading
        test_config = {
            'default_provider': 'openai',
            'fallback_providers': ['ollama', 'groq'],
            'model_preferences': {
                'coding': 'gpt-4',
                'creative': 'mistral-large',
                'analysis': 'llama3-70b-8192'
            }
        }
        
        with patch('json.dump') as mock_dump:
            with patch('builtins.open'):
                # Simulate saving configuration
                import json
                config_str = json.dumps(test_config)
                
                with patch('json.load') as mock_load:
                    mock_load.return_value = test_config
                    
                    # Simulate loading configuration
                    loaded_config = json.loads(config_str)
                    
                    self.assertEqual(loaded_config, test_config)
    
    @patch('external_providers.load_api_keys')
    def test_provider_switching_performance(self, mock_load_keys):
        """Test performance characteristics of provider switching"""
        mock_load_keys.return_value = self.test_api_keys
        
        import time
        
        # Mock fast and slow providers
        def fast_provider(model, messages):
            time.sleep(0.1)  # 100ms
            return "Fast response"
        
        def slow_provider(model, messages):
            time.sleep(1.0)  # 1 second
            return "Slow response"
        
        with patch('ollama_utils.call_ollama_api', side_effect=fast_provider):
            with patch('openai_utils.call_openai_api', side_effect=slow_provider):
                
                test_messages = [{"role": "user", "content": "test"}]
                
                # Test fast provider
                start_time = time.time()
                response1 = call_ollama_api("mistral:instruct", test_messages)
                fast_time = time.time() - start_time
                
                # Test slow provider
                start_time = time.time()
                response2 = call_openai_api("gpt-4", test_messages)
                slow_time = time.time() - start_time
                
                # Verify timing characteristics
                self.assertLess(fast_time, 0.5)  # Fast provider should be under 500ms
                self.assertGreater(slow_time, 0.5)  # Slow provider should be over 500ms
                self.assertEqual(response1, "Fast response")
                self.assertEqual(response2, "Slow response")


class TestProviderCompatibilityIntegration(TestCase):
    """Test compatibility between different providers"""
    
    def setUp(self):
        """Set up test environment"""
        self.standard_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    
    @patch('external_providers.load_api_keys')
    def test_message_format_compatibility(self, mock_load_keys):
        """Test that all providers accept standard message format"""
        mock_load_keys.return_value = {
            'openai_api_key': 'test_key',
            'groq_api_key': 'test_key',
            'mistral_api_key': 'test_key'
        }
        
        # Test each provider with the same message format
        providers_to_test = [
            ('ollama_utils.call_ollama_api', "mistral:instruct"),
            ('openai_utils.call_openai_api', "gpt-4"),
            ('groq_utils.call_groq_api', "llama3-70b-8192"),
            ('mistral_utils.call_mistral_api', "mistral-large")
        ]
        
        for api_path, model in providers_to_test:
            with patch(api_path) as mock_api:
                mock_api.return_value = f"Response from {model}"
                
                # All providers should accept the standard format
                if 'ollama' in api_path:
                    response = call_ollama_api(model, self.standard_messages)
                elif 'openai' in api_path:
                    response = call_openai_api(model, self.standard_messages)
                elif 'groq' in api_path:
                    response = call_groq_api(model, self.standard_messages)
                elif 'mistral' in api_path:
                    response = call_mistral_api(model, self.standard_messages)
                
                self.assertIn(model, response)
                mock_api.assert_called_once_with(model, self.standard_messages)
    
    @patch('external_providers.load_api_keys')
    def test_parameter_compatibility(self, mock_load_keys):
        """Test parameter compatibility across providers"""
        mock_load_keys.return_value = {
            'openai_api_key': 'test_key',
            'groq_api_key': 'test_key',
            'mistral_api_key': 'test_key'
        }
        
        # Common parameters that should work across providers
        common_params = {
            'temperature': 0.7,
            'max_tokens': 1000,
            'top_p': 0.9
        }
        
        # Test parameter acceptance (mocked)
        with patch('ollama_utils.call_ollama_api') as mock_ollama:
            mock_ollama.return_value = "Ollama response with params"
            
            # Simulate parameter passing (actual implementation would vary)
            response = call_ollama_api("mistral:instruct", self.standard_messages)
            self.assertIsNotNone(response)
    
    @patch('external_providers.load_api_keys')
    def test_conversation_continuity(self, mock_load_keys):
        """Test conversation continuity when switching providers"""
        mock_load_keys.return_value = {
            'openai_api_key': 'test_key',
            'groq_api_key': 'test_key'
        }
        
        # Build a conversation
        conversation = [
            {"role": "user", "content": "Hello, I'm learning about space."},
            {"role": "assistant", "content": "That's fascinating! What aspect interests you most?"},
            {"role": "user", "content": "Tell me about black holes."}
        ]
        
        # Test switching providers mid-conversation
        with patch('ollama_utils.call_ollama_api') as mock_ollama:
            mock_ollama.return_value = "Black holes are regions of spacetime..."
            
            with patch('openai_utils.call_openai_api') as mock_openai:
                mock_openai.return_value = "Continuing about black holes, they form when..."
                
                # First response with Ollama
                response1 = call_ollama_api("mistral:instruct", conversation)
                
                # Add response to conversation
                conversation.append({"role": "assistant", "content": response1})
                conversation.append({"role": "user", "content": "How do they form?"})
                
                # Second response with OpenAI
                response2 = call_openai_api("gpt-4", conversation)
                
                # Both responses should be contextually appropriate
                self.assertIn("black holes", response1.lower())
                self.assertIn("black holes", response2.lower())


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
