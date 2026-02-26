"""
Test suite for multimodel_chat.py - Multi-model chat feature
"""

import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock, call
import tempfile

# Import the module to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a mock for streamlit session_state that supports both dict and attribute access
class SessionStateMock(dict):
    """Mock for streamlit session_state that supports both dict and attribute access"""
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value

import streamlit as st
from ollama_workbench.chat.multimodel_chat import MultiModelChat


class TestMultiModelChatInitialization:
    """Test MultiModelChat initialization and setup"""
    
    @patch('multimodel_chat.st.session_state', SessionStateMock())
    @patch('multimodel_chat.os.path.exists', return_value=False)
    @patch('multimodel_chat.load_api_keys', return_value={})
    def test_init_no_settings_file(self, mock_load_keys, mock_exists):
        """Test initialization when no settings file exists"""
        chat = MultiModelChat()
        
        # Verify session state is initialized
        assert "models" in st.session_state
        assert "multimodel_chat_history" in st.session_state
        assert "selected_models" in st.session_state
        assert "comparison_mode" in st.session_state
        assert st.session_state.comparison_mode == "side-by-side"
        assert "model_settings" in st.session_state
        assert "shared_context" in st.session_state
        assert st.session_state.shared_context is True
        assert "total_tokens" in st.session_state
        assert isinstance(st.session_state.total_tokens, dict)
    
    @patch('multimodel_chat.st.session_state', SessionStateMock())
    @patch('multimodel_chat.load_api_keys', return_value={"test": "key"})
    @patch('multimodel_chat.os.path.exists', return_value=True)
    @patch('builtins.open', create=True)
    @patch('multimodel_chat.json.load')
    def test_init_with_settings_file(self, mock_json_load, mock_open, mock_exists, mock_load_keys):
        """Test initialization when settings file exists"""
        test_settings = {
            "selected_models": ["gpt-4", "llama2"],
            "comparison_mode": "tabbed",
            "shared_context": False
        }
        mock_json_load.return_value = test_settings
        
        chat = MultiModelChat()
        
        # Verify settings were loaded
        assert st.session_state.selected_models == ["gpt-4", "llama2"]
        assert st.session_state.comparison_mode == "tabbed"
        assert st.session_state.shared_context is False
        assert st.session_state.api_keys == {"test": "key"}
    
    @patch('multimodel_chat.st.session_state', SessionStateMock({"total_tokens": "invalid"}))
    @patch('multimodel_chat.load_api_keys', return_value={})
    def test_init_fixes_invalid_total_tokens(self, mock_load_keys):
        """Test that initialization fixes invalid total_tokens type"""
        with patch('multimodel_chat.os.path.exists', return_value=False):
            chat = MultiModelChat()
            
            # Verify total_tokens was converted to dict
            assert isinstance(st.session_state.total_tokens, dict)


class TestSettingsManagement:
    """Test settings save/load functionality"""
    
    @patch('multimodel_chat.st.session_state', SessionStateMock({
        "selected_models": ["model1", "model2"],
        "comparison_mode": "side-by-side",
        "model_settings": {"model1": {"temp": 0.5}},
        "shared_context": True
    }))
    @patch('builtins.open', create=True)
    @patch('multimodel_chat.json.dump')
    @patch('multimodel_chat.st.success')
    def test_save_settings_success(self, mock_success, mock_json_dump, mock_open):
        """Test successful settings save"""
        chat = MultiModelChat()
        chat.save_settings()
        
        # Verify correct data was saved
        expected_settings = {
            "selected_models": ["model1", "model2"],
            "comparison_mode": "side-by-side",
            "model_settings": {"model1": {"temp": 0.5}},
            "shared_context": True
        }
        mock_json_dump.assert_called_once()
        saved_data = mock_json_dump.call_args[0][0]
        assert saved_data == expected_settings
        
        mock_success.assert_called_once_with("Settings saved successfully!")
    
    @patch('multimodel_chat.st.session_state', SessionStateMock())
    @patch('builtins.open', side_effect=IOError("Write error"))
    @patch('multimodel_chat.st.error')
    def test_save_settings_error(self, mock_error, mock_open):
        """Test settings save error handling"""
        chat = MultiModelChat()
        chat.save_settings()
        
        mock_error.assert_called_once()
        error_message = mock_error.call_args[0][0]
        assert "Error saving settings:" in error_message


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_extract_content_blocks_with_code(self):
        """Test extracting code blocks from text"""
        chat = MultiModelChat()
        text = """Here's some code:
```python
def hello():
    print("Hello, World!")
```
And some more text"""
        
        code_blocks, article_blocks = chat.extract_content_blocks(text)
        
        assert len(code_blocks) == 1
        assert "def hello():" in code_blocks[0]
        assert "print(" in code_blocks[0]
    
    def test_extract_content_blocks_with_articles(self):
        """Test extracting article blocks from text"""
        chat = MultiModelChat()
        text = """Title: First Article
Content of first article

Title: Second Article
Content of second article"""
        
        code_blocks, article_blocks = chat.extract_content_blocks(text)
        
        assert len(code_blocks) == 0
        assert len(article_blocks) == 2
        assert "First Article" in article_blocks[0]
        assert "Second Article" in article_blocks[1]
    
    def test_extract_content_blocks_none(self):
        """Test extracting blocks from None"""
        chat = MultiModelChat()
        code_blocks, article_blocks = chat.extract_content_blocks(None)
        
        assert code_blocks == []
        assert article_blocks == []
    
    @patch('multimodel_chat.tiktoken.get_encoding')
    def test_count_tokens(self, mock_get_encoding):
        """Test token counting"""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
        mock_get_encoding.return_value = mock_encoding
        
        chat = MultiModelChat()
        count = chat.count_tokens("Test text")
        
        assert count == 5
        mock_encoding.encode.assert_called_once_with("Test text")
    
    def test_get_model_display_name(self):
        """Test model display name formatting"""
        chat = MultiModelChat()
        
        assert chat.get_model_display_name("openai/gpt-4") == "OpenAI - gpt-4"
        assert chat.get_model_display_name("groq/llama3-70b") == "Groq - llama3-70b"
        assert chat.get_model_display_name("mistral/mistral-large") == "Mistral - mistral-large"
        assert chat.get_model_display_name("llama2") == "Ollama - llama2"


class TestModelSelection:
    """Test model selection functionality"""
    
    @patch('multimodel_chat.st.sidebar.multiselect')
    @patch('multimodel_chat.st.sidebar.subheader')
    @patch('multimodel_chat.st.sidebar.radio')
    @patch('multimodel_chat.st.sidebar.checkbox')
    @patch('multimodel_chat.st.sidebar.button')
    @patch('multimodel_chat.get_all_models', return_value=["model1", "model2", "model3"])
    @patch('multimodel_chat.st.session_state', SessionStateMock({
        "selected_models": [],
        "total_tokens": {},
        "model_settings": {},
        "comparison_mode": "side-by-side",
        "shared_context": True
    }))
    def test_model_selector_basic(self, mock_get_models, mock_button, mock_checkbox, 
                                  mock_radio, mock_subheader, mock_multiselect):
        """Test basic model selection"""
        mock_multiselect.return_value = ["model1", "model2"]
        mock_radio.return_value = "side-by-side"
        mock_checkbox.return_value = True
        mock_button.return_value = False
        
        chat = MultiModelChat()
        selected = chat.model_selector()
        
        assert selected == ["model1", "model2"]
        assert st.session_state.selected_models == ["model1", "model2"]
        
        # Verify model settings were initialized
        assert "model1" in st.session_state.model_settings
        assert "model2" in st.session_state.model_settings
        assert st.session_state.model_settings["model1"]["temperature"] == 0.7
        assert st.session_state.model_settings["model1"]["max_tokens"] == 4000
    
    @patch('multimodel_chat.st.sidebar.multiselect')
    @patch('multimodel_chat.st.sidebar.warning')
    @patch('multimodel_chat.get_all_models', return_value=["m1", "m2", "m3", "m4", "m5"])
    @patch('multimodel_chat.st.session_state', SessionStateMock({
        "selected_models": [],
        "total_tokens": {},
        "model_settings": {}
    }))
    def test_model_selector_limit_exceeded(self, mock_get_models, mock_warning, mock_multiselect):
        """Test model selection with more than 4 models"""
        # Return 5 models selected
        mock_multiselect.return_value = ["m1", "m2", "m3", "m4", "m5"]
        
        with patch.object(MultiModelChat, 'save_settings'):
            chat = MultiModelChat()
            selected = chat.model_selector()
        
        # Should limit to 4 models
        assert len(selected) == 4
        assert selected == ["m1", "m2", "m3", "m4"]
        mock_warning.assert_called_once()


class TestResponseGeneration:
    """Test response generation from models"""
    
    @patch('multimodel_chat.st.session_state', SessionStateMock({
        "api_keys": {"openai_api_key": "test-key"},
        "model_settings": {"gpt-4": {"temperature": 0.7, "max_tokens": 1000}},
        "agent_type": "None",
        "metacognitive_type": "None", 
        "voice_type": "None",
        "multimodel_chat_history": [],
        "shared_context": True,
        "selected_corpus": "None"
    }))
    @patch('multimodel_chat.call_openai_api')
    def test_generate_response_openai(self, mock_openai):
        """Test response generation from OpenAI model"""
        mock_openai.return_value = "OpenAI response"
        
        chat = MultiModelChat()
        with patch('multimodel_chat.OPENAI_MODELS', ["gpt-4"]):
            response = chat.generate_model_response("gpt-4", "Hello")
        
        assert response == "OpenAI response"
        mock_openai.assert_called_once()
    
    @patch('multimodel_chat.st.session_state', SessionStateMock({
        "api_keys": {"groq_api_key": "test-key"},
        "model_settings": {"llama3-70b": {"temperature": 0.8, "max_tokens": 2000}}
    }))
    @patch('multimodel_chat.call_groq_api')
    def test_generate_response_groq(self, mock_groq):
        """Test response generation from Groq model"""
        mock_groq.return_value = "Groq response"
        
        chat = MultiModelChat()
        with patch('multimodel_chat.GROQ_MODELS', ["llama3-70b"]):
            response = chat.generate_model_response("llama3-70b", "Hello")
        
        assert response == "Groq response"
        mock_groq.assert_called_once()
    
    @patch('multimodel_chat.st.session_state', SessionStateMock({
        "api_keys": {"mistral_api_key": "test-key"},
        "model_settings": {"mistral-large": {"temperature": 0.9, "max_tokens": 3000}}
    }))
    @patch('multimodel_chat.call_mistral_api')
    def test_generate_response_mistral(self, mock_mistral):
        """Test response generation from Mistral model"""
        mock_mistral.return_value = "Mistral response"
        
        chat = MultiModelChat()
        with patch('multimodel_chat.MISTRAL_MODELS', ["mistral-large"]):
            response = chat.generate_model_response("mistral-large", "Hello")
        
        assert response == "Mistral response"
        mock_mistral.assert_called_once()
    
    @patch('multimodel_chat.st.session_state', SessionStateMock({
        "api_keys": {},
        "model_settings": {"llama2": {"temperature": 0.5, "max_tokens": 4000}}
    }))
    @patch('multimodel_chat.get_ollama_client')
    def test_generate_response_ollama_with_client(self, mock_get_client):
        """Test response generation from Ollama model with client"""
        mock_client = Mock()
        mock_client.generate.return_value = {"response": "Ollama response"}
        mock_get_client.return_value = mock_client
        
        chat = MultiModelChat()
        with patch('multimodel_chat.OPENAI_MODELS', []), \
             patch('multimodel_chat.GROQ_MODELS', []), \
             patch('multimodel_chat.MISTRAL_MODELS', []):
            response = chat.generate_model_response("llama2", "Hello")
        
        assert response == "Ollama response"
        mock_client.generate.assert_called_once()
    
    @patch('multimodel_chat.st.session_state', SessionStateMock({
        "api_keys": {},
        "model_settings": {"llama2": {"temperature": 0.5, "max_tokens": 4000}}
    }))
    @patch('multimodel_chat.get_ollama_client', return_value=None)
    @patch('multimodel_chat.call_ollama_endpoint')
    def test_generate_response_ollama_fallback(self, mock_endpoint, mock_get_client):
        """Test response generation from Ollama model with fallback"""
        mock_endpoint.return_value = ("Ollama fallback response", None, None, None)
        
        chat = MultiModelChat()
        with patch('multimodel_chat.OPENAI_MODELS', []), \
             patch('multimodel_chat.GROQ_MODELS', []), \
             patch('multimodel_chat.MISTRAL_MODELS', []):
            response = chat.generate_model_response("llama2", "Hello")
        
        assert response == "Ollama fallback response"
        mock_endpoint.assert_called_once()
    
    @patch('multimodel_chat.st.session_state', SessionStateMock({"api_keys": {}, "model_settings": {}}))
    @patch('multimodel_chat.call_openai_api', side_effect=Exception("API Error"))
    def test_generate_response_error(self, mock_openai):
        """Test error handling in response generation"""
        chat = MultiModelChat()
        with patch('multimodel_chat.OPENAI_MODELS', ["gpt-4"]):
            response = chat.generate_model_response("gpt-4", "Hello")
        
        assert "Error:" in response
        assert "API Error" in response


class TestPromptCreation:
    """Test prompt creation functionality"""
    
    @patch('multimodel_chat.st.session_state', SessionStateMock({
        "api_keys": {},
        "agent_type": "coding_assistant",
        "metacognitive_type": "reflection",
        "voice_type": "professional",
        "multimodel_chat_history": [
            {"role": "user", "content": "Previous message"},
            {"role": "assistant", "content": "Previous response", "model": "test-model"}
        ],
        "shared_context": True,
        "selected_corpus": "None"
    }))
    @patch('multimodel_chat.get_agent_prompt')
    @patch('multimodel_chat.get_metacognitive_prompt')
    @patch('multimodel_chat.get_voice_prompt')
    def test_create_model_prompt_with_agents(self, mock_voice, mock_meta, mock_agent):
        """Test prompt creation with agents enabled"""
        mock_agent.return_value = {"coding_assistant": {"prompt": "You are a coding assistant"}}
        mock_meta.return_value = {"reflection": "Think step by step"}
        mock_voice.return_value = {"professional": "Maintain professional tone"}
        
        chat = MultiModelChat()
        prompt = chat.create_model_prompt("test-model", "New question")
        
        assert "You are a coding assistant" in prompt
        assert "Think step by step" in prompt
        assert "Maintain professional tone" in prompt
        assert "New question" in prompt
        assert "Previous message" in prompt  # Chat history
    
    @patch('multimodel_chat.st.session_state', SessionStateMock({
        "api_keys": {},
        "agent_type": "None",
        "metacognitive_type": "None",
        "voice_type": "None",
        "multimodel_chat_history": [],
        "shared_context": False,
        "selected_corpus": "None"
    }))
    def test_create_model_prompt_minimal(self):
        """Test minimal prompt creation"""
        chat = MultiModelChat()
        prompt = chat.create_model_prompt("test-model", "Simple question")
        
        # The prompt includes a default AI assistant prompt when all agents are "None"
        assert "Simple question" in prompt
        assert "You are an AI assistant" in prompt


class TestDisplayFunctions:
    """Test display and UI functions"""
    
    @patch('multimodel_chat.st.chat_message')
    @patch('multimodel_chat.st.markdown')
    def test_display_chat_message_user(self, mock_markdown, mock_chat_message):
        """Test displaying user message"""
        chat = MultiModelChat()
        message = {"role": "user", "content": "User message"}
        
        chat.display_chat_message(message)
        
        mock_chat_message.assert_called_once_with("user")
        mock_markdown.assert_called_once_with("User message")
    
    @patch('multimodel_chat.st.chat_message')
    @patch('multimodel_chat.st.markdown')
    @patch('multimodel_chat.st.code')
    def test_display_chat_message_assistant_with_code(self, mock_code, mock_markdown, mock_chat_message):
        """Test displaying assistant message with code blocks"""
        chat = MultiModelChat()
        message = {
            "role": "assistant",
            "content": "Here's code:\n```python\nprint('Hello')\n```\nEnd of message"
        }
        
        chat.display_chat_message(message, model="gpt-4")
        
        mock_chat_message.assert_called_once_with("assistant")
        mock_code.assert_called_once_with("python\nprint('Hello')")
        # Should have model label and non-code content
        assert mock_markdown.call_count >= 2
    
    @patch('multimodel_chat.st.chat_message')
    @patch('multimodel_chat.st.warning')
    def test_display_chat_message_no_content(self, mock_warning, mock_chat_message):
        """Test displaying message with no content"""
        chat = MultiModelChat()
        message = {"role": "assistant", "content": None}
        
        chat.display_chat_message(message)
        
        mock_warning.assert_called_once_with("This message has no content.")


class TestModelSettings:
    """Test model-specific settings UI"""
    
    @patch('multimodel_chat.st.sidebar.tabs')
    @patch('multimodel_chat.st.sidebar.subheader')
    @patch('multimodel_chat.st.session_state', SessionStateMock({
        "selected_models": ["model1", "model2"],
        "model_settings": {
            "model1": {"temperature": 0.7, "max_tokens": 2000, "presence_penalty": 0.0, "frequency_penalty": 0.0},
            "model2": {"temperature": 0.5, "max_tokens": 4000, "presence_penalty": 0.0, "frequency_penalty": 0.0}
        },
        "total_tokens": {"model1": 100, "model2": 200}
    }))
    def test_model_settings_ui(self, mock_subheader, mock_tabs):
        """Test model settings UI creation"""
        # Mock tab context managers
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_tabs.return_value = [mock_tab1, mock_tab2]
        
        # Mock sliders and info
        with patch('multimodel_chat.st.slider') as mock_slider, \
             patch('multimodel_chat.st.info') as mock_info:
            
            mock_slider.side_effect = [0.7, 2000, 0.0, 0.0, 0.5, 4000, 0.0, 0.0]
            
            chat = MultiModelChat()
            chat.model_settings_ui()
            
            # Verify tabs were created
            mock_tabs.assert_called_once()
            
            # Verify sliders were created for each model
            assert mock_slider.call_count == 8  # 4 sliders per model, 2 models
            
            # Verify info displays - should be called twice for 2 models
            assert mock_info.call_count == 2
            # Check that both info calls were made with token counts
            info_calls = [call[0][0] for call in mock_info.call_args_list]
            assert any("Total tokens used:" in call for call in info_calls)


class TestIntegration:
    """Integration tests"""
    
    @patch('multimodel_chat.st.session_state', SessionStateMock())
    @patch('multimodel_chat.os.path.exists', return_value=False)
    @patch('multimodel_chat.load_api_keys', return_value={})
    def test_full_initialization_flow(self, mock_load_keys, mock_exists):
        """Test complete initialization flow"""
        chat = MultiModelChat()
        
        # Verify all session state variables are initialized
        required_keys = [
            "models", "multimodel_chat_history", "selected_models",
            "comparison_mode", "model_settings", "shared_context",
            "total_tokens", "api_keys", "response_comparisons"
        ]
        
        for key in required_keys:
            assert key in st.session_state
        
        # Verify defaults
        assert st.session_state.comparison_mode == "side-by-side"
        assert st.session_state.shared_context is True
        assert isinstance(st.session_state.total_tokens, dict)
        assert isinstance(st.session_state.model_settings, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])