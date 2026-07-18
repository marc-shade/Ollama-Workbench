"""
Test suite for voice_interface.py - Voice interface UI components
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock, call
import streamlit as st

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AttrDict(dict):
    """Minimal stand-in for streamlit's session_state: dict + attribute access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TestVoiceInterface:
    """Test voice interface UI components"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Clear any session state
        if hasattr(st, 'session_state'):
            st.session_state.clear()
    
    @patch('ollama_workbench.chat.voice_interface.voice_utils')
    @patch('ollama_workbench.chat.voice_interface.st')
    def test_voice_settings_ui_basic_structure(self, mock_st, mock_voice_utils):
        """Test basic structure of voice settings UI"""
        # Mock available voices
        mock_voice_utils.get_available_voices.return_value = ['default', 'male', 'female']
        mock_voice_utils.get_voice_settings.return_value = {
            'provider': 'gtts',
            'language': 'en',
            'voice_id': 'default',
            'speed': 1.0,
            'pitch': 1.0
        }
        
        # Mock streamlit components
        mock_st.tabs.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.return_value = 'default'
        mock_st.text_input.return_value = 'en'
        mock_st.slider.return_value = 1.0
        mock_st.button.return_value = False
        
        import ollama_workbench.chat.voice_interface as voice_interface

        
        # Test the function runs without error
        voice_interface.voice_settings_ui()
        
        # Verify key components were called
        mock_st.title.assert_called_with("Voice Settings")
        mock_st.tabs.assert_called_once()
        mock_voice_utils.get_available_voices.assert_called_once()
    
    @patch('ollama_workbench.chat.voice_interface.voice_utils')
    @patch('ollama_workbench.chat.voice_interface.st')
    def test_voice_settings_profile_management(self, mock_st, mock_voice_utils):
        """Test voice profile management functionality"""
        # Setup mocks
        mock_voice_utils.get_available_voices.return_value = ['default', 'test_profile']
        mock_voice_utils.get_voice_settings.return_value = {
            'provider': 'gtts',
            'language': 'en',
            'voice_id': 'default'
        }
        mock_voice_utils.remove_voice_profile.return_value = True
        mock_voice_utils.add_voice_profile.return_value = True
        
        # Mock streamlit components
        mock_tabs = [MagicMock(), MagicMock()]
        mock_st.tabs.return_value = mock_tabs
        mock_st.selectbox.side_effect = ['test_profile', 'gtts', 'elevenlabs']
        # Call order: language_edit, voice_id_edit, test_text_edit,
        # profile_name_new, language_new, voice_id_new
        mock_st.text_input.side_effect = ['en', 'test_voice', 'A test sentence',
                                          'TestProfile', 'en', 'test_voice']
        mock_st.slider.return_value = 1.0
        mock_st.button.side_effect = [True, False, False, True]  # Delete, Test, Save, Create
        
        import ollama_workbench.chat.voice_interface as voice_interface

        voice_interface.voice_settings_ui()
        
        # Verify profile operations
        mock_voice_utils.remove_voice_profile.assert_called_with('test_profile')
        mock_voice_utils.add_voice_profile.assert_called()
    
    @patch('ollama_workbench.chat.voice_interface.voice_utils')
    @patch('ollama_workbench.chat.voice_interface.st')
    def test_voice_settings_test_functionality(self, mock_st, mock_voice_utils):
        """Test voice testing functionality"""
        # Setup mocks
        mock_voice_utils.get_available_voices.return_value = ['default']
        mock_voice_utils.get_voice_settings.return_value = {
            'provider': 'gtts',
            'language': 'en',
            'voice_id': 'default'
        }
        mock_voice_utils.text_to_speech.return_value = '/tmp/test.mp3'
        mock_voice_utils.play_speech.return_value = None
        
        # Mock streamlit components
        mock_tabs = [MagicMock(), MagicMock()]
        mock_st.tabs.return_value = mock_tabs
        mock_st.selectbox.return_value = 'default'
        mock_st.text_input.return_value = 'Test speech'
        mock_st.slider.return_value = 1.0
        # The Delete button is skipped for the 'default' profile, so the
        # buttons rendered are: Test Voice, Save Changes, Create Profile
        mock_st.button.side_effect = [True, False, False]
        
        import ollama_workbench.chat.voice_interface as voice_interface

        voice_interface.voice_settings_ui()
        
        # Verify test functionality
        mock_voice_utils.text_to_speech.assert_called_with('Test speech', 'default')
        mock_voice_utils.play_speech.assert_called_with('/tmp/test.mp3')
    
    @patch('ollama_workbench.chat.voice_interface.voice_utils')
    @patch('ollama_workbench.chat.voice_interface.st')
    def test_voice_input_component_basic(self, mock_st, mock_voice_utils):
        """Test basic voice input component functionality"""
        # Mock session state
        mock_session_state = AttrDict()
        mock_st.session_state = mock_session_state

        # Mock streamlit components (columns are used as context managers)
        mock_columns = [MagicMock(), MagicMock()]
        mock_st.columns.return_value = mock_columns
        mock_st.button.return_value = False
        mock_st.text_input.return_value = ""
        
        import ollama_workbench.chat.voice_interface as voice_interface

        
        # Test with no callback
        voice_interface.voice_input_component()
        
        # Verify basic structure
        mock_st.columns.assert_called_with([1, 10])
        mock_st.button.assert_called()
        mock_st.text_input.assert_called()
    
    @patch('ollama_workbench.chat.voice_interface.voice_utils')
    @patch('ollama_workbench.chat.voice_interface.st')
    def test_voice_input_component_start_listening(self, mock_st, mock_voice_utils):
        """Test starting voice input listening"""
        # Mock session state
        mock_session_state = AttrDict({'is_listening': False})
        mock_st.session_state = mock_session_state

        # Mock streamlit components (columns are used as context managers)
        mock_columns = [MagicMock(), MagicMock()]
        mock_st.columns.return_value = mock_columns
        mock_st.button.return_value = True  # Button clicked
        mock_st.text_input.return_value = ""
        mock_st.rerun = Mock()
        
        import ollama_workbench.chat.voice_interface as voice_interface

        
        callback = Mock()
        voice_interface.voice_input_component(callback)
        
        # Verify listening started
        assert mock_session_state['is_listening'] is True
        mock_voice_utils.start_voice_input.assert_called_once()
        mock_st.rerun.assert_called_once()
    
    @patch('ollama_workbench.chat.voice_interface.voice_utils')
    @patch('ollama_workbench.chat.voice_interface.st')
    def test_voice_input_component_stop_listening(self, mock_st, mock_voice_utils):
        """Test stopping voice input listening"""
        # Mock session state
        mock_session_state = AttrDict({'is_listening': True})
        mock_st.session_state = mock_session_state

        # Mock streamlit components (columns are used as context managers)
        mock_columns = [MagicMock(), MagicMock()]
        mock_st.columns.return_value = mock_columns
        mock_st.button.return_value = True  # Button clicked
        mock_st.text_input.return_value = ""
        mock_st.rerun = Mock()
        
        # Mock voice utils
        mock_voice_utils.stop_voice_input.return_value = "Hello world"
        
        import ollama_workbench.chat.voice_interface as voice_interface

        
        callback = Mock()
        voice_interface.voice_input_component(callback)
        
        # Verify listening stopped and callback called
        assert mock_session_state['is_listening'] is False
        mock_voice_utils.stop_voice_input.assert_called_once()
        callback.assert_called_with("Hello world")
        mock_st.rerun.assert_called_once()
    
    @patch('ollama_workbench.chat.voice_interface.voice_utils')
    @patch('ollama_workbench.chat.voice_interface.st')
    @patch('ollama_workbench.providers.ollama_utils.call_ollama_endpoint')
    def test_voice_chat_interface_basic_structure(self, mock_ollama, mock_st, mock_voice_utils):
        """Test basic structure of voice chat interface"""
        # Mock session state
        mock_session_state = AttrDict({'voice_chat_history': []})
        mock_st.session_state = mock_session_state

        # Mock streamlit components
        mock_st.chat_input.return_value = None
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.button.return_value = False
        mock_st.tabs.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.return_value = "llama3"
        mock_st.slider.return_value = 0.7
        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock(return_value=False)  # do not suppress exceptions
        
        # Mock voice utils
        mock_voice_utils.get_available_voices.return_value = ['default']
        
        # Mock get_all_models
        with patch('ollama_workbench.providers.ollama_utils.get_all_models') as mock_get_models:
            mock_get_models.return_value = ['llama3', 'mixtral']
            
            import ollama_workbench.chat.voice_interface as voice_interface

            voice_interface.voice_chat_interface()
        
        # Verify key components
        mock_st.title.assert_called_with("Voice Chat")
        mock_st.chat_input.assert_called_once()
        mock_voice_utils.get_available_voices.assert_called_once()
    
    @patch('ollama_workbench.chat.voice_interface.voice_utils')
    @patch('ollama_workbench.chat.voice_interface.st')
    @patch('ollama_workbench.providers.ollama_utils.call_ollama_endpoint')
    def test_voice_chat_interface_message_handling(self, mock_ollama, mock_st, mock_voice_utils):
        """Test message handling in voice chat interface"""
        # Setup chat history
        mock_session_state = AttrDict({
            'voice_chat_history': [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        })
        mock_st.session_state = mock_session_state

        # Mock streamlit components
        mock_st.chat_input.return_value = "New message"
        mock_st.chat_message.return_value.__enter__ = Mock()
        mock_st.chat_message.return_value.__exit__ = Mock(return_value=False)  # do not suppress exceptions
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.button.return_value = False
        mock_st.tabs.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.return_value = "llama3"
        mock_st.slider.return_value = 0.7
        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock(return_value=False)  # do not suppress exceptions
        mock_st.rerun = Mock()
        
        # Mock voice and model utilities
        mock_voice_utils.get_available_voices.return_value = ['default']
        mock_voice_utils.text_to_speech.return_value = '/tmp/response.mp3'
        mock_voice_utils.play_speech.return_value = None
        # call_ollama_endpoint returns a 5-tuple:
        # (response, context, eval_count, eval_duration, metrics_dict)
        mock_ollama.return_value = ("Generated response", None, None, None, {})
        
        # Mock get_all_models
        with patch('ollama_workbench.providers.ollama_utils.get_all_models') as mock_get_models:
            mock_get_models.return_value = ['llama3', 'mixtral']
            
            import ollama_workbench.chat.voice_interface as voice_interface

            voice_interface.voice_chat_interface()
        
        # Verify message processing
        assert len(mock_session_state['voice_chat_history']) == 4  # 2 original + 2 new
        mock_ollama.assert_called_once()
        mock_voice_utils.text_to_speech.assert_called_once()
        mock_voice_utils.play_speech.assert_called_once()
    
    @patch('ollama_workbench.chat.voice_interface.voice_utils')
    @patch('ollama_workbench.chat.voice_interface.st')
    @patch('ollama_workbench.providers.groq_utils.call_groq_api')
    def test_voice_chat_interface_groq_provider(self, mock_groq, mock_st, mock_voice_utils):
        """Test voice chat with Groq provider"""
        # Mock session state with Groq model
        mock_session_state = AttrDict({
            'voice_chat_history': [],
            'voice_chat_model': '🚀 Groq Models mixtral-8x7b',
            'voice_chat_temperature': 0.7,
            'voice_chat_max_tokens': 500
        })
        mock_st.session_state = mock_session_state

        # Mock streamlit components
        mock_st.chat_input.return_value = "Test message"
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.button.return_value = False
        mock_st.tabs.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.return_value = "🚀 Groq Models mixtral-8x7b"
        mock_st.slider.return_value = 0.7
        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock(return_value=False)  # do not suppress exceptions
        mock_st.rerun = Mock()
        mock_st.spinner.return_value.__enter__ = Mock()
        mock_st.spinner.return_value.__exit__ = Mock(return_value=False)  # do not suppress exceptions
        
        # Mock APIs
        mock_voice_utils.get_available_voices.return_value = ['default']
        mock_voice_utils.text_to_speech.return_value = '/tmp/response.mp3'
        mock_voice_utils.play_speech.return_value = None
        mock_groq.return_value = {
            'choices': [{'message': {'content': 'Groq response'}}]
        }
        
        # Mock get_all_models
        with patch('ollama_workbench.providers.ollama_utils.get_all_models') as mock_get_models:
            mock_get_models.return_value = ['🚀 Groq Models mixtral-8x7b']
            
            import ollama_workbench.chat.voice_interface as voice_interface

            voice_interface.voice_chat_interface()
        
        # Verify Groq API was called
        mock_groq.assert_called_once()
        args, kwargs = mock_groq.call_args
        assert kwargs['model'] == 'mixtral-8x7b'
        assert kwargs['temperature'] == 0.7
        assert kwargs['max_tokens'] == 500
    
    @patch('ollama_workbench.chat.voice_interface.voice_utils')
    @patch('ollama_workbench.chat.voice_interface.st')
    @patch('ollama_workbench.providers.openai_utils.call_openai_api')
    def test_voice_chat_interface_openai_provider(self, mock_openai, mock_st, mock_voice_utils):
        """Test voice chat with OpenAI provider"""
        # Mock session state with OpenAI model
        mock_session_state = AttrDict({
            'voice_chat_history': [],
            'voice_chat_model': '🤖 OpenAI Models gpt-4',
            'voice_chat_temperature': 0.5,
            'voice_chat_max_tokens': 1000
        })
        mock_st.session_state = mock_session_state

        # Mock streamlit components
        mock_st.chat_input.return_value = "Test OpenAI message"
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.button.return_value = False
        mock_st.tabs.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.return_value = "🤖 OpenAI Models gpt-4"
        mock_st.slider.return_value = 0.5
        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock(return_value=False)  # do not suppress exceptions
        mock_st.rerun = Mock()
        mock_st.spinner.return_value.__enter__ = Mock()
        mock_st.spinner.return_value.__exit__ = Mock(return_value=False)  # do not suppress exceptions
        
        # Mock APIs
        mock_voice_utils.get_available_voices.return_value = ['default']
        mock_voice_utils.text_to_speech.return_value = '/tmp/response.mp3'
        mock_voice_utils.play_speech.return_value = None
        mock_openai.return_value = {
            'choices': [{'message': {'content': 'OpenAI response'}}]
        }
        
        # Mock get_all_models
        with patch('ollama_workbench.providers.ollama_utils.get_all_models') as mock_get_models:
            mock_get_models.return_value = ['🤖 OpenAI Models gpt-4']
            
            import ollama_workbench.chat.voice_interface as voice_interface

            voice_interface.voice_chat_interface()
        
        # Verify OpenAI API was called
        mock_openai.assert_called_once()
        args, kwargs = mock_openai.call_args
        assert kwargs['model'] == 'gpt-4'
        assert kwargs['temperature'] == 0.5
        assert kwargs['max_tokens'] == 1000
    
    @patch('ollama_workbench.chat.voice_interface.voice_utils')
    @patch('ollama_workbench.chat.voice_interface.st')
    def test_voice_chat_interface_error_handling(self, mock_st, mock_voice_utils):
        """Test error handling in voice chat interface"""
        # Mock session state
        mock_session_state = AttrDict({
            'voice_chat_history': [],
            'voice_chat_model': 'llama3'
        })
        mock_st.session_state = mock_session_state

        # Mock streamlit components
        mock_st.chat_input.return_value = "Test error message"
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.button.return_value = False
        mock_st.tabs.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.return_value = "llama3"
        mock_st.slider.return_value = 0.7
        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock(return_value=False)  # do not suppress exceptions
        mock_st.rerun = Mock()
        mock_st.spinner.return_value.__enter__ = Mock()
        mock_st.spinner.return_value.__exit__ = Mock(return_value=False)  # do not suppress exceptions
        
        # Mock voice utils
        mock_voice_utils.get_available_voices.return_value = ['default']
        mock_voice_utils.text_to_speech.return_value = '/tmp/response.mp3'
        mock_voice_utils.play_speech.return_value = None
        
        # Mock get_all_models
        with patch('ollama_workbench.providers.ollama_utils.get_all_models') as mock_get_models:
            mock_get_models.return_value = ['llama3']
            
            # Mock ollama call to raise exception
            with patch('ollama_workbench.providers.ollama_utils.call_ollama_endpoint') as mock_ollama:
                mock_ollama.side_effect = Exception("Test error")
                
                import ollama_workbench.chat.voice_interface as voice_interface

                voice_interface.voice_chat_interface()
        
        # Verify error handling
        mock_st.error.assert_called_once()
        assert "Test error" in str(mock_st.error.call_args)
        
        # Verify error message was added to history
        assert len(mock_session_state['voice_chat_history']) == 2
        assert "error" in mock_session_state['voice_chat_history'][1]['content'].lower()
    
    @patch('ollama_workbench.chat.voice_interface.st')
    def test_main_function_structure(self, mock_st):
        """Test main function structure when run as script"""
        # Mock streamlit components
        mock_st.set_page_config.return_value = None
        mock_st.tabs.return_value = [MagicMock(), MagicMock()]
        
        # Mock the main functions to avoid running full UI
        with patch('ollama_workbench.chat.voice_interface.voice_chat_interface') as mock_chat:
            with patch('ollama_workbench.chat.voice_interface.voice_settings_ui') as mock_settings:
                # Import and run the main block
                import ollama_workbench.chat.voice_interface as voice_interface

                
                # Simulate running as main
                if __name__ == "__main__":
                    # This would normally be executed
                    pass
        
        # Test would verify page config if main block was executed
        # For now, just verify the functions exist
        assert hasattr(voice_interface, 'voice_chat_interface')
        assert hasattr(voice_interface, 'voice_settings_ui')
    
    @patch('ollama_workbench.chat.voice_interface.voice_utils')
    @patch('ollama_workbench.chat.voice_interface.st')
    def test_voice_chat_conversation_context(self, mock_st, mock_voice_utils):
        """Test conversation context handling in voice chat"""
        # Setup conversation history
        mock_session_state = AttrDict({
            'voice_chat_history': [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is artificial intelligence."},
                {"role": "user", "content": "Tell me more"}
            ]
        })
        mock_st.session_state = mock_session_state

        # Mock streamlit components
        mock_st.chat_input.return_value = "Can you explain machine learning?"
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.button.return_value = False
        mock_st.tabs.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.return_value = "llama3"
        mock_st.slider.return_value = 0.7
        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock(return_value=False)  # do not suppress exceptions
        mock_st.rerun = Mock()
        mock_st.spinner.return_value.__enter__ = Mock()
        mock_st.spinner.return_value.__exit__ = Mock(return_value=False)  # do not suppress exceptions
        
        # Mock voice and model utilities
        mock_voice_utils.get_available_voices.return_value = ['default']
        mock_voice_utils.text_to_speech.return_value = '/tmp/response.mp3'
        mock_voice_utils.play_speech.return_value = None
        
        # Mock get_all_models
        with patch('ollama_workbench.providers.ollama_utils.get_all_models') as mock_get_models:
            mock_get_models.return_value = ['llama3']
            
            # Mock ollama call (5-tuple contract)
            with patch('ollama_workbench.providers.ollama_utils.call_ollama_endpoint') as mock_ollama:
                mock_ollama.return_value = ("Machine learning is...", None, None, None, {})
                
                import ollama_workbench.chat.voice_interface as voice_interface

                voice_interface.voice_chat_interface()
        
        # Verify context was passed correctly
        mock_ollama.assert_called_once()
        args, kwargs = mock_ollama.call_args
        assert kwargs['prompt'] == "Can you explain machine learning?"


class TestVoiceInterfaceIntegration:
    """Integration tests for voice interface components"""
    
    @patch('ollama_workbench.chat.voice_interface.voice_utils')
    @patch('ollama_workbench.chat.voice_interface.st')
    def test_complete_voice_workflow(self, mock_st, mock_voice_utils):
        """Test complete voice workflow from input to output"""
        # Mock session state
        mock_session_state = AttrDict({'voice_chat_history': []})
        mock_st.session_state = mock_session_state

        # Mock voice utils
        mock_voice_utils.get_available_voices.return_value = ['default', 'female']
        mock_voice_utils.get_voice_settings.return_value = {
            'provider': 'gtts',
            'language': 'en'
        }
        mock_voice_utils.text_to_speech.return_value = '/tmp/test.mp3'
        mock_voice_utils.play_speech.return_value = None
        # start_voice_input(speech_callback, error_callback) delivers recognized
        # speech through the callback; simulate a successful recognition.
        mock_voice_utils.start_voice_input.side_effect = \
            lambda speech_cb, error_cb: speech_cb("Hello AI")
        mock_voice_utils.stop_voice_input.return_value = "Hello AI"

        # Mock streamlit (columns are used as context managers)
        mock_st.button.side_effect = [True, False]  # Voice button clicked, then not
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.text_input.return_value = ""
        mock_st.rerun = Mock()
        
        import ollama_workbench.chat.voice_interface as voice_interface

        
        # Test voice input component
        received_text = None
        def callback(text):
            nonlocal received_text
            received_text = text
        
        voice_interface.voice_input_component(callback)
        
        # Verify workflow
        mock_voice_utils.start_voice_input.assert_called_once()
        assert received_text == "Hello AI"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
