"""
Test script for model selection functionality

This script tests the model categorization and selection UI functionality
to ensure that Ollama models are properly detected and displayed.

CHECKPOINT: Test script initialization
"""

import sys
import logging
import unittest
from unittest.mock import patch, MagicMock

# Set up logging with detailed checkpoints
logging.basicConfig(
    filename='test_model_selection.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.info("CHECKPOINT: Starting model selection tests")

# Import the modules to test
try:
    import robust_ollama_utils as ollama_utils
    from model_categorization import categorize_models, get_model_description
    logger.info("CHECKPOINT: Successfully imported modules for testing")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing modules: {str(e)}")
    print(f"Error importing modules: {str(e)}")
    sys.exit(1)

class TestModelSelection(unittest.TestCase):
    """Test cases for model selection functionality"""
    
    def setUp(self):
        """Set up test environment"""
        logger.info("CHECKPOINT: Setting up test environment")
        
        # Sample model data for testing
        self.sample_models = [
            "llama3", "mistral", "gemma", "phi", "codellama",  # Ollama models
            "gpt-4", "gpt-3.5-turbo",  # OpenAI models
            "llama2-70b-4096", "mixtral-8x7b-32768",  # Groq models
            "mistral-small", "mistral-medium",  # Mistral models
            "claude-3-opus"  # Anthropic models
        ]
        
        # Create a mock response for Ollama API
        self.mock_ollama_response = {
            "models": [
                {"name": "llama3", "modified_at": "2023-01-01T00:00:00Z", "size": 1000000},
                {"name": "mistral", "modified_at": "2023-01-01T00:00:00Z", "size": 1000000},
                {"name": "gemma", "modified_at": "2023-01-01T00:00:00Z", "size": 1000000},
                {"name": "phi", "modified_at": "2023-01-01T00:00:00Z", "size": 1000000},
                {"name": "codellama", "modified_at": "2023-01-01T00:00:00Z", "size": 1000000}
            ]
        }
        
        logger.info("CHECKPOINT: Test environment setup complete")
    
    def test_model_categorization(self):
        """Test that models are properly categorized by provider"""
        logger.info("CHECKPOINT: Testing model categorization")
        
        categorized = categorize_models(self.sample_models)
        
        # Verify categorization
        self.assertIn("llama3", categorized["ollama"])
        self.assertIn("gpt-4", categorized["openai"])
        self.assertIn("llama2-70b-4096", categorized["groq"])
        self.assertIn("mistral-small", categorized["mistral"])
        self.assertIn("claude-3-opus", categorized["anthropic"])
        
        # Verify counts
        self.assertEqual(len(categorized["ollama"]), 5)
        self.assertEqual(len(categorized["openai"]), 2)
        self.assertEqual(len(categorized["groq"]), 2)
        # The mistral category includes both 'mistral' and 'mistral-small', 'mistral-medium'
        self.assertEqual(len(categorized["mistral"]), 3)
        self.assertEqual(len(categorized["anthropic"]), 1)
        
        logger.info("CHECKPOINT: Model categorization test passed")
    
    def test_model_descriptions(self):
        """Test that model descriptions are provided correctly"""
        logger.info("CHECKPOINT: Testing model descriptions")
        
        # Test known models
        self.assertIn("Meta's Llama 3", get_model_description("llama3"))
        self.assertIn("OpenAI's most powerful", get_model_description("gpt-4"))
        
        # Test unknown model
        self.assertIn("Model running on", get_model_description("unknown-model"))
        
        logger.info("CHECKPOINT: Model descriptions test passed")
    
    @patch('requests.get')
    def test_get_available_models_api(self, mock_get):
        """Test that available models are fetched correctly via API"""
        logger.info("CHECKPOINT: Testing get_available_models via API")
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_ollama_response
        mock_get.return_value = mock_response
        
        # Patch the ollama client to simulate it not being available
        with patch.object(ollama_utils, 'OLLAMA_AVAILABLE', False):
            models = ollama_utils.get_available_models()
            
            # Verify models
            self.assertEqual(len(models), 5)
            self.assertIn("llama3", models)
            self.assertIn("mistral", models)
            self.assertIn("gemma", models)
            
            # Verify API was called
            mock_get.assert_called()
        
        logger.info("CHECKPOINT: get_available_models API test passed")
    
    @patch('ollama.list')
    def test_get_available_models_client(self, mock_list):
        """Test that available models are fetched correctly via client"""
        logger.info("CHECKPOINT: Testing get_available_models via client")
        
        # Mock successful client response
        mock_list.return_value = self.mock_ollama_response
        
        # Patch the ollama client to simulate it being available
        with patch.object(ollama_utils, 'OLLAMA_AVAILABLE', True):
            with patch.object(ollama_utils, 'get_ollama_client', return_value=None):
                models = ollama_utils.get_available_models()
                
                # Verify models
                self.assertEqual(len(models), 5)
                self.assertIn("llama3", models)
                self.assertIn("mistral", models)
                self.assertIn("gemma", models)
                
                # Verify client was called
                mock_list.assert_called_once()
        
        logger.info("CHECKPOINT: get_available_models client test passed")
    
    def test_get_all_models(self):
        """Test that all models from different providers are combined correctly"""
        logger.info("CHECKPOINT: Testing get_all_models")
        
        # Mock the get_available_models function to return our sample Ollama models
        with patch.object(ollama_utils, 'get_available_models', return_value=self.sample_models[:5]):
            # Mock the external model lists
            with patch.object(ollama_utils, 'GROQ_MODELS', self.sample_models[7:9]):
                with patch.object(ollama_utils, 'GROQ_AVAILABLE', True):
                    with patch.object(ollama_utils, 'OPENAI_MODELS', self.sample_models[5:7]):
                        with patch.object(ollama_utils, 'OPENAI_AVAILABLE', True):
                            with patch.object(ollama_utils, 'MISTRAL_MODELS', self.sample_models[9:11]):
                                with patch.object(ollama_utils, 'MISTRAL_AVAILABLE', True):
                                    all_models = ollama_utils.get_all_models()
                                    
                                    # Verify all models are included
                                    self.assertEqual(len(all_models), 11)
                                    self.assertIn("llama3", all_models)
                                    self.assertIn("gpt-4", all_models)
                                    self.assertIn("llama2-70b-4096", all_models)
                                    self.assertIn("mistral-small", all_models)
        
        logger.info("CHECKPOINT: get_all_models test passed")
    
    def test_fallback_models(self):
        """Test that default models are provided when none are found"""
        logger.info("CHECKPOINT: Testing fallback models")
        
        # Mock empty responses from all sources
        with patch.object(ollama_utils, 'get_available_models', return_value=[]):
            with patch.object(ollama_utils, 'GROQ_MODELS', []):
                with patch.object(ollama_utils, 'GROQ_AVAILABLE', True):
                    with patch.object(ollama_utils, 'OPENAI_MODELS', []):
                        with patch.object(ollama_utils, 'OPENAI_AVAILABLE', True):
                            with patch.object(ollama_utils, 'MISTRAL_MODELS', []):
                                with patch.object(ollama_utils, 'MISTRAL_AVAILABLE', True):
                                    all_models = ollama_utils.get_all_models()
                                    
                                    # Verify default models are provided
                                    self.assertEqual(len(all_models), 4)
                                    self.assertIn("llama3", all_models)
                                    self.assertIn("mistral", all_models)
                                    self.assertIn("gemma", all_models)
                                    self.assertIn("llama2", all_models)
        
        logger.info("CHECKPOINT: Fallback models test passed")

def run_tests():
    """Run the test suite"""
    logger.info("CHECKPOINT: Running test suite")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    logger.info("CHECKPOINT: Test suite completed")

if __name__ == "__main__":
    print("Running model selection tests...")
    run_tests()
    print("Tests completed. Check test_model_selection.log for detailed results.")
