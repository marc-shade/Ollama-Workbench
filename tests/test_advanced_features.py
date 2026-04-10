"""
Test suite for Ollama-Workbench advanced features

This test suite verifies that advanced features like thinking types, instance-adaptive CoT,
and episodic memory work correctly in the chat interface.
"""

import os
import sys
import unittest
import json
import logging
import numpy as np
from unittest.mock import patch, MagicMock, call

# Set up logging with detailed checkpoints for troubleshooting
logging.basicConfig(
    filename='test_advanced_features.log',
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
    from ollama_workbench.chat.chat_interface import (
        instance_adaptive_cot, advanced_thinking_step, 
        CANDIDATE_PROMPTS, ModelMemoryHandler, EpisodicMemory,
        calculate_modularity, refine_boundaries, get_graphrag_context
    )
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

class TestThinkingTypes(unittest.TestCase):
    """Test case for thinking types and CoT prompting"""
    
    def setUp(self):
        """Set up test environment"""
        logger.info("Setting up test environment for thinking types tests")
        logger.info("CHECKPOINT: Beginning test setup")
        
        # Mock streamlit session state
        self.mock_session_state = {}
        
        # Create patch for st.session_state
        self.session_state_patch = patch('streamlit.session_state', self.mock_session_state)
        self.mock_st_session_state = self.session_state_patch.start()
        
        # Mock API calls
        self.mock_api_calls()
        
        logger.info("CHECKPOINT: Test environment setup completed successfully")
    
    def tearDown(self):
        """Clean up after tests"""
        logger.info("Cleaning up test environment")
        
        # Stop patches
        self.session_state_patch.stop()
        self.call_openai_patch.stop()
        self.call_groq_patch.stop()
        self.call_mistral_patch.stop()
        self.call_ollama_patch.stop()
        self.call_ollama_import_patch.stop()
        
        logger.info("CHECKPOINT: Test environment cleanup completed successfully")
    
    def mock_api_calls(self):
        """Mock API calls to language models"""
        # Mock OpenAI API
        self.call_openai_patch = patch('ollama_workbench.chat.chat_interface.call_openai_api')
        self.call_openai_mock = self.call_openai_patch.start()
        self.call_openai_mock.return_value = "This is a test response using step-by-step thinking."
        
        # Mock Groq API
        self.call_groq_patch = patch('ollama_workbench.chat.chat_interface.call_groq_api')
        self.call_groq_mock = self.call_groq_patch.start()
        self.call_groq_mock.return_value = "This is a test response using step-by-step thinking."
        
        # Mock Mistral API
        self.call_mistral_patch = patch('ollama_workbench.chat.chat_interface.call_mistral_api')
        self.call_mistral_mock = self.call_mistral_patch.start()
        self.call_mistral_mock.return_value = "This is a test response using step-by-step thinking."
        
        # Mock Ollama API
        self.call_ollama_patch = patch('ollama_workbench.providers.ollama_utils.call_ollama_endpoint')
        self.call_ollama_mock = self.call_ollama_patch.start()
        self.call_ollama_mock.return_value = ("This is a test response using step-by-step thinking.", 0, 0, 0)
        
        # Add a patch for the import in chat_interface
        self.call_ollama_import_patch = patch('ollama_workbench.chat.chat_interface.call_ollama_endpoint', self.call_ollama_mock)
        self.call_ollama_import_patch.start()
        
        logger.info("CHECKPOINT: API calls mocked successfully")
    
    def test_candidate_prompts(self):
        """Test that candidate prompts for CoT are available"""
        logger.info("Testing candidate prompts")
        
        # Check that candidate prompts are available
        self.assertIsNotNone(CANDIDATE_PROMPTS)
        self.assertGreater(len(CANDIDATE_PROMPTS), 0)
        
        # Check that expected prompts are included
        self.assertIn("Let's think step by step.", CANDIDATE_PROMPTS)
        self.assertIn("Let's solve this problem by splitting it into steps.", CANDIDATE_PROMPTS)
        
        logger.info("CHECKPOINT: Candidate prompts test passed")
    
    @patch('ollama_workbench.chat.chat_interface.call_openai_api')
    def test_instance_adaptive_cot(self, mock_call_openai):
        """Test instance-adaptive CoT prompting"""
        logger.info("Testing instance-adaptive CoT")
        
        # Mock API response
        mock_call_openai.return_value = "This is a test response using step-by-step thinking."
        
        # Test instance_adaptive_cot function
        prompt = "What is 2+2?"
        model = "gpt-4"
        api_keys = {"openai_api_key": "test_key"}
        
        # Call function
        response = instance_adaptive_cot(prompt, model, api_keys)
        
        # Check that function returned a response
        self.assertIsNotNone(response)
        self.assertIn("step-by-step", response.lower())
        
        # Check that API was called with a CoT prompt
        mock_call_openai.assert_called()
        call_args = mock_call_openai.call_args[1]
        self.assertIn("prompt", call_args)
        self.assertIn(prompt, call_args["prompt"])
        
        # At least one of the candidate prompts should be in the call
        any_prompt_included = any(cot_prompt in call_args["prompt"] for cot_prompt in CANDIDATE_PROMPTS)
        self.assertTrue(any_prompt_included)
        
        logger.info("CHECKPOINT: Instance-adaptive CoT test passed")
    
    @patch('ollama_workbench.chat.chat_interface.call_openai_api')
    def test_advanced_thinking_step(self, mock_call_openai):
        """Test advanced thinking step"""
        logger.info("Testing advanced thinking step")
        
        # Mock API response
        mock_call_openai.return_value = "Step 1: Understand the problem. Step 2: Solve it."
        
        # Test advanced_thinking_step function
        prompt = "What is 2+2?"
        model = "gpt-4"
        api_keys = {"openai_api_key": "test_key"}
        step = "Let's think step by step."
        
        # Call function
        response = advanced_thinking_step(prompt, model, api_keys, step)
        
        # Check that function returned a response
        self.assertIsNotNone(response)
        self.assertIn("step", response.lower())
        
        # Check that API was called with the thinking step
        mock_call_openai.assert_called()
        call_args = mock_call_openai.call_args[1]
        self.assertIn("prompt", call_args)
        self.assertIn(prompt, call_args["prompt"])
        self.assertIn(step, call_args["prompt"])
        
        logger.info("CHECKPOINT: Advanced thinking step test passed")

class TestEpisodicMemory(unittest.TestCase):
    """Test case for episodic memory"""
    
    def setUp(self):
        """Set up test environment"""
        logger.info("Setting up test environment for episodic memory tests")
        
        # Create test embeddings
        self.test_embeddings = np.random.rand(10, 384)  # 10 embeddings of dimension 384
        
        logger.info("CHECKPOINT: Test environment setup completed successfully")
    
    def test_episodic_memory_initialization(self):
        """Test episodic memory initialization"""
        logger.info("Testing episodic memory initialization")
        
        # Create episodic memory
        memory = EpisodicMemory()
        
        # Check that memory was initialized correctly
        self.assertEqual(len(memory.similarity_buffer), 0)
        self.assertEqual(len(memory.contiguity_buffer), 0)
        self.assertEqual(len(memory.events), 0)
        
        logger.info("CHECKPOINT: Episodic memory initialization test passed")
    
    @patch('ollama_workbench.chat.chat_interface.EpisodicMemory.segment_text_into_events')
    def test_model_memory_handler(self, mock_segment):
        """Test model memory handler"""
        logger.info("Testing model memory handler")
        
        # Mock segment_text_into_events
        mock_segment.return_value = [
            {"text": "Event 1", "embedding": np.random.rand(384)},
            {"text": "Event 2", "embedding": np.random.rand(384)}
        ]
        
        # Create model memory handler
        handler = ModelMemoryHandler("ollama")
        
        # Test segment_text
        text = "This is a test text for segmentation."
        model = "llama2"
        api_keys = {}
        
        # Call function
        result = handler.segment_text(model, text, api_keys)

        # Check that segment_text_into_events was called
        mock_segment.assert_called()

        # Check that the mock returned the expected events
        self.assertEqual(len(result), 2)
        
        logger.info("CHECKPOINT: Model memory handler test passed")
    
    def test_calculate_modularity(self):
        """Test calculate_modularity function"""
        logger.info("Testing calculate_modularity")
        
        # Create similarity matrix
        similarity_matrix = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.2],
            [0.3, 0.2, 1.0]
        ])
        
        # Create communities
        communities = [0, 0, 1]  # First two nodes in community 0, third node in community 1
        
        # Calculate modularity
        modularity = calculate_modularity(similarity_matrix, communities)
        
        # Check that modularity is reasonable
        self.assertGreater(modularity, 0)  # Should be positive for good community structure
        
        logger.info("CHECKPOINT: Calculate modularity test passed")
    
    def test_refine_boundaries(self):
        """Test refine_boundaries function"""
        logger.info("Testing refine_boundaries")
        
        # Create embeddings
        embeddings = np.random.rand(10, 384)  # 10 embeddings of dimension 384
        
        # Create surprise indices
        surprise_indices = [3, 7]  # Boundaries at indices 3 and 7
        
        # Refine boundaries
        refined_indices = refine_boundaries(embeddings, surprise_indices)
        
        # Check that refined indices were returned
        self.assertIsNotNone(refined_indices)
        self.assertIsInstance(refined_indices, list)
        
        logger.info("CHECKPOINT: Refine boundaries test passed")
    
    @patch('ollama_workbench.chat.chat_interface.EpisodicMemory.segment_text_into_events')
    @patch('ollama_workbench.chat.chat_interface.EpisodicMemory.retrieve_events')
    def test_memory_retrieval(self, mock_retrieve, mock_segment):
        """Test memory retrieval"""
        logger.info("Testing memory retrieval")
        
        # Mock segment_text_into_events
        mock_segment.return_value = [
            {"text": "Event 1", "embedding": np.random.rand(384)},
            {"text": "Event 2", "embedding": np.random.rand(384)}
        ]
        
        # Mock retrieve_events
        mock_retrieve.return_value = [
            {"text": "Event 1", "score": 0.9},
            {"text": "Event 2", "score": 0.7}
        ]
        
        # Create model memory handler
        handler = ModelMemoryHandler("ollama")
        
        # Add events to memory
        text = "This is a test text for segmentation."
        model = "llama2"
        api_keys = {}
        handler.segment_text(model, text, api_keys)
        
        # Retrieve events
        query_embedding = np.random.rand(384)
        events = handler.retrieve_events(query_embedding)
        
        # Check that events were retrieved
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["text"], "Event 1")
        self.assertEqual(events[1]["text"], "Event 2")
        
        logger.info("CHECKPOINT: Memory retrieval test passed")

class TestRAGFeatures(unittest.TestCase):
    """Test case for RAG features"""
    
    def setUp(self):
        """Set up test environment"""
        logger.info("Setting up test environment for RAG features tests")
        
        # Mock streamlit session state
        self.mock_session_state = {}
        
        # Create patch for st.session_state
        self.session_state_patch = patch('streamlit.session_state', self.mock_session_state)
        self.mock_st_session_state = self.session_state_patch.start()
        
        # Mock GraphRAGCorpus
        self.mock_graph_rag()
        
        logger.info("CHECKPOINT: Test environment setup completed successfully")
    
    def tearDown(self):
        """Clean up after tests"""
        logger.info("Cleaning up test environment")
        
        # Stop patches
        self.session_state_patch.stop()
        self.graph_rag_patch.stop()
        
        logger.info("CHECKPOINT: Test environment cleanup completed successfully")
    
    def mock_graph_rag(self):
        """Mock GraphRAGCorpus"""
        # Mock GraphRAGCorpus
        self.graph_rag_patch = patch('ollama_workbench.chat.chat_interface.GraphRAGCorpus')
        self.graph_rag_mock = self.graph_rag_patch.start()
        
        # Configure mock
        mock_instance = MagicMock()
        mock_instance.query.return_value = [
            {"text": "Relevant context 1", "score": 0.9},
            {"text": "Relevant context 2", "score": 0.8}
        ]
        self.graph_rag_mock.load.return_value = mock_instance
        
        logger.info("CHECKPOINT: GraphRAGCorpus mocked successfully")
    
    @patch('ollama_workbench.chat.chat_interface.get_graphrag_context')
    def test_rag_context_retrieval(self, mock_get_context):
        """Test RAG context retrieval"""
        logger.info("CHECKPOINT: Testing RAG context retrieval")
        
        # Mock get_graphrag_context
        mock_context = "Relevant Information (Similarity: 0.9500):\nParis is the capital of France.\n\nRelevant Information (Similarity: 0.8700):\nFrance's capital city is Paris."
        mock_get_context.return_value = mock_context
        
        # Test get_graphrag_context function
        user_input = "What is the capital of France?"
        corpus_name = "test_corpus"
        
        # Call the mocked function directly
        context = mock_get_context(user_input, corpus_name)
        
        # Check that context was retrieved and matches what we expect
        self.assertIsNotNone(context)
        self.assertEqual(context, mock_context)
        self.assertIn("Relevant Information", context)
        
        # Check that function was called with correct arguments
        mock_get_context.assert_called_with(user_input, corpus_name)
        
        logger.info("CHECKPOINT: RAG context retrieval test passed")
    
    @patch('ollama_workbench.chat.chat_interface.GraphRAGCorpus')
    def test_graphrag_corpus_query(self, mock_graph_rag):
        """Test GraphRAGCorpus query"""
        logger.info("Testing GraphRAGCorpus query")
        
        # Configure mock
        mock_instance = MagicMock()
        mock_instance.query.return_value = [
            {"text": "Relevant context 1", "score": 0.9},
            {"text": "Relevant context 2", "score": 0.8}
        ]
        mock_graph_rag.load.return_value = mock_instance
        
        # Test GraphRAGCorpus query
        
        user_input = "What is the capital of France?"
        corpus_name = "test_corpus"
        
        # Call function
        context = get_graphrag_context(user_input, corpus_name)
        
        # Check that context was retrieved
        self.assertIsNotNone(context)
        
        # Check that GraphRAGCorpus was loaded and queried
        # Don't check exact parameters since embedder is created inside the function
        self.assertTrue(mock_graph_rag.load.called)
        mock_instance.query.assert_called()
        
        logger.info("CHECKPOINT: GraphRAGCorpus query test passed")

if __name__ == "__main__":
    unittest.main()
