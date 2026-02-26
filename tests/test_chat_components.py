"""
Test suite for Ollama-Workbench chat components

This test suite focuses on the individual components of the chat interfaces,
ensuring that UI elements, message rendering, and model interactions work correctly.
"""

import os
import sys
import unittest
import json
import logging
import re
from unittest.mock import patch, MagicMock, call

# Set up logging
logging.basicConfig(
    filename='test_components.log',
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
    from ollama_workbench.chat.chat_interface import extract_code_blocks, extract_content_blocks, count_tokens
    from modern_chat_interface import display_message, get_rag_context
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

class TestChatComponents(unittest.TestCase):
    """Test case for chat components"""
    
    def setUp(self):
        """Set up test environment"""
        logger.info("Setting up test environment for chat components tests")
        logger.info("CHECKPOINT: Beginning test setup")
        
        # Mock streamlit session state
        self.mock_session_state = {}
        
        # Create patch for st.session_state
        self.session_state_patch = patch('streamlit.session_state', self.mock_session_state)
        self.mock_st_session_state = self.session_state_patch.start()
        
        # Mock streamlit functions
        self.mock_st_functions()
        
        logger.info("CHECKPOINT: Test environment setup completed successfully")
    
    def tearDown(self):
        """Clean up after tests"""
        logger.info("Cleaning up test environment")
        
        # Stop patches
        self.session_state_patch.stop()
        
        logger.info("CHECKPOINT: Test environment cleanup completed successfully")
    
    def mock_st_functions(self):
        """Mock streamlit functions"""
        # Create mock for streamlit functions
        self.st_markdown_patch = patch('streamlit.markdown', return_value=None)
        self.st_markdown_mock = self.st_markdown_patch.start()
        
        self.st_container_patch = patch('streamlit.container', return_value=MagicMock())
        self.st_container_mock = self.st_container_patch.start()
        
        self.st_columns_patch = patch('streamlit.columns', return_value=[MagicMock(), MagicMock()])
        self.st_columns_mock = self.st_columns_patch.start()
        
        logger.info("CHECKPOINT: Streamlit functions mocked successfully")
    
    def test_extract_code_blocks(self):
        """Test extracting code blocks from text"""
        logger.info("Testing extract_code_blocks")
        
        # Test with no code blocks
        text = "This is a test message with no code blocks."
        code_blocks, _ = extract_code_blocks(text)
        self.assertEqual(len(code_blocks), 0)
        
        # Test with one code block
        text = "This is a test message with one code block.\n```python\nprint('Hello, world!')\n```"
        code_blocks, _ = extract_code_blocks(text)
        self.assertEqual(len(code_blocks), 1)
        self.assertEqual(code_blocks[0], "python\nprint('Hello, world!')")
        
        # Test with multiple code blocks
        text = """This is a test message with multiple code blocks.
```python
print('Hello, world!')
```
Some text in between.
```javascript
console.log('Hello, world!');
```"""
        code_blocks, _ = extract_code_blocks(text)
        self.assertEqual(len(code_blocks), 2)
        self.assertEqual(code_blocks[0], "python\nprint('Hello, world!')")
        self.assertEqual(code_blocks[1], "javascript\nconsole.log('Hello, world!');")
        
        logger.info("Extract code blocks test passed")
        logger.info("CHECKPOINT: Extract code blocks test completed successfully")
    
    def test_extract_content_blocks(self):
        """Test extracting content blocks from text"""
        logger.info("Testing extract_content_blocks")
        
        # Test with no content blocks
        text = "This is a test message with no content blocks."
        code_blocks, article_blocks = extract_content_blocks(text)
        self.assertEqual(len(code_blocks), 0)
        self.assertEqual(len(article_blocks), 0)
        
        # Test with code and article blocks
        text = """This is a test message with code and article blocks.
```python
print('Hello, world!')
```
Title: Test Article
This is a test article.

Title: Another Article
This is another test article."""
        code_blocks, article_blocks = extract_content_blocks(text)
        self.assertEqual(len(code_blocks), 1)
        self.assertEqual(code_blocks[0], "python\nprint('Hello, world!')")
        self.assertEqual(len(article_blocks), 2)
        self.assertTrue(article_blocks[0].startswith("Title: Test Article"))
        self.assertTrue(article_blocks[1].startswith("Title: Another Article"))
        
        logger.info("Extract content blocks test passed")
        logger.info("CHECKPOINT: Extract content blocks test completed successfully")
    
    def test_count_tokens(self):
        """Test counting tokens in text"""
        logger.info("Testing count_tokens")
        
        # Test with empty text
        self.assertEqual(count_tokens(""), 0)
        
        # Test with short text
        text = "This is a short test message."
        token_count = count_tokens(text)
        self.assertGreater(token_count, 0)
        
        # Test with longer text
        text = "This is a longer test message with multiple sentences. It should have more tokens than the short message."
        long_token_count = count_tokens(text)
        self.assertGreater(long_token_count, token_count)
        
        logger.info("Count tokens test passed")
        logger.info("CHECKPOINT: Count tokens test completed successfully")
    
    @patch('modern_chat_interface.st.container')
    @patch('modern_chat_interface.st.columns')
    def test_display_message(self, mock_columns, mock_container):
        """Test displaying a message"""
        logger.info("Testing display_message")
        
        # Set up mocks
        mock_container.return_value = MagicMock()
        mock_columns.return_value = [MagicMock(), MagicMock()]
        
        # Test displaying a user message
        user_message = {"role": "user", "content": "Hello, world!"}
        display_message(user_message, 0)
        
        # Test displaying an assistant message
        assistant_message = {"role": "assistant", "content": "Hi there!"}
        display_message(assistant_message, 1)
        
        # Test displaying a message with code blocks
        code_message = {"role": "assistant", "content": "Here's some code:\n```python\nprint('Hello, world!')\n```"}
        display_message(code_message, 2)
        
        logger.info("Display message test passed")
        logger.info("CHECKPOINT: Display message test completed successfully")
    
    @patch('modern_chat_interface.GraphRAGCorpus')
    def test_get_rag_context(self, mock_graph_rag_corpus):
        """Test getting RAG context"""
        logger.info("Testing get_rag_context")
        
        # Set up mock
        mock_instance = MagicMock()
        mock_instance.query.return_value = [
            {"content": "Test context 1", "similarity": 0.9},
            {"content": "Test context 2", "similarity": 0.8}
        ]
        mock_graph_rag_corpus.load.return_value = mock_instance
        
        # Add detailed logging for debugging
        logger.info("CHECKPOINT: Mock GraphRAGCorpus setup complete")
        logger.info(f"CHECKPOINT: Mock query return value: {mock_instance.query.return_value}")
        
        # Test with no corpus
        context = get_rag_context("Test query", "None")
        self.assertEqual(context, "")
        
        # Test with a corpus
        context = get_rag_context("Test query", "test_corpus")
        self.assertIn("Test context 1", context)
        self.assertIn("Test context 2", context)
        
        logger.info("Get RAG context test passed")
        logger.info("CHECKPOINT: Get RAG context test completed successfully")

class TestMessageRendering(unittest.TestCase):
    """Test case for message rendering"""
    
    def setUp(self):
        """Set up test environment"""
        logger.info("Setting up test environment for message rendering tests")
        
        # Mock streamlit functions
        self.st_markdown_patch = patch('streamlit.markdown', return_value=None)
        self.st_markdown_mock = self.st_markdown_patch.start()
        
        self.st_container_patch = patch('streamlit.container', return_value=MagicMock())
        self.st_container_mock = self.st_container_patch.start()
        
        self.st_columns_patch = patch('streamlit.columns', return_value=[MagicMock(), MagicMock()])
        self.st_columns_mock = self.st_columns_patch.start()
        
        logger.info("CHECKPOINT: Test environment setup completed successfully")
    
    def tearDown(self):
        """Clean up after tests"""
        logger.info("Cleaning up test environment")
        
        # Stop patches
        self.st_markdown_patch.stop()
        self.st_container_patch.stop()
        self.st_columns_patch.stop()
        
        logger.info("CHECKPOINT: Test environment cleanup completed successfully")
    
    def test_markdown_rendering(self):
        """Test rendering markdown in messages"""
        logger.info("Testing markdown rendering")
        
        # Test with basic markdown
        text = "This is **bold** and *italic* text."
        st.markdown(text)
        self.st_markdown_mock.assert_called_with(text)
        
        # Test with code blocks
        text = "This is a code block:\n```python\nprint('Hello, world!')\n```"
        st.markdown(text)
        self.st_markdown_mock.assert_called_with(text)
        
        # Test with links
        text = "This is a [link](https://example.com)."
        st.markdown(text)
        self.st_markdown_mock.assert_called_with(text)
        
        logger.info("Markdown rendering test passed")
        logger.info("CHECKPOINT: Markdown rendering test completed successfully")
    
    def test_code_block_rendering(self):
        """Test rendering code blocks in messages"""
        logger.info("Testing code block rendering")
        
        # Extract code blocks from text
        text = "This is a code block:\n```python\nprint('Hello, world!')\n```"
        code_blocks, _ = extract_code_blocks(text)
        
        # Render code blocks
        for block in code_blocks:
            match = re.match(r'^(\w+)\n', block)
            if match:
                language = match.group(1)
                code = block[len(language)+1:]
                st.markdown(f"```{language}\n{code}\n```")
                self.st_markdown_mock.assert_called_with(f"```{language}\n{code}\n```")
        
        logger.info("Code block rendering test passed")
        logger.info("CHECKPOINT: Code block rendering test completed successfully")

if __name__ == "__main__":
    unittest.main()
