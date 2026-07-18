"""
Test suite for voice_utils.py - Voice chat functionality
"""

import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, call, mock_open
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVoiceUtils:
    """Test voice utilities"""
    
    @patch('ollama_workbench.chat.voice_utils.pygame.mixer.init')
    @patch('ollama_workbench.chat.voice_utils.pyaudio.PyAudio')
    @patch('ollama_workbench.chat.voice_utils.sr.Recognizer')
    def test_voice_manager_import(self, mock_recognizer, mock_pyaudio, mock_pygame):
        """Test that voice_utils can be imported"""
        import ollama_workbench.chat.voice_utils as voice_utils

        
        # Verify the module loaded
        assert hasattr(voice_utils, 'VoiceManager')
        assert hasattr(voice_utils, 'voice_manager')
        assert hasattr(voice_utils, 'DEFAULT_VOICE')
    
    def test_module_functions_exist(self):
        """Test that all expected functions exist"""
        import ollama_workbench.chat.voice_utils as voice_utils

        
        # Check all expected functions
        functions = [
            'start_voice_input', 'stop_voice_input', 'text_to_speech',
            'play_speech', 'stop_speech', 'get_available_voices',
            'get_voice_settings', 'add_voice_profile', 'remove_voice_profile',
            'cleanup'
        ]
        
        for func in functions:
            assert hasattr(voice_utils, func), f"Missing function: {func}"
    
    @patch('ollama_workbench.chat.voice_utils.voice_manager')
    def test_text_to_speech_function(self, mock_manager):
        """Test text_to_speech convenience function"""
        import ollama_workbench.chat.voice_utils as voice_utils

        
        mock_manager.text_to_speech.return_value = "audio.mp3"
        
        result = voice_utils.text_to_speech("Hello", voice_name='test')
        
        mock_manager.text_to_speech.assert_called_once_with("Hello", 'test', None)
        assert result == "audio.mp3"
    
    @patch('ollama_workbench.chat.voice_utils.voice_manager')
    def test_get_available_voices(self, mock_manager):
        """Test get_available_voices function"""
        import ollama_workbench.chat.voice_utils as voice_utils

        
        mock_manager.get_available_voices.return_value = ['default', 'custom']
        
        voices = voice_utils.get_available_voices()
        
        assert voices == ['default', 'custom']
        mock_manager.get_available_voices.assert_called_once()
    
    @patch('ollama_workbench.chat.voice_utils.voice_manager')
    def test_start_stop_voice_input(self, mock_manager):
        """Test start and stop voice input functions"""
        import ollama_workbench.chat.voice_utils as voice_utils

        
        callback = Mock()
        error_callback = Mock()
        
        # Test start
        voice_utils.start_voice_input(callback, error_callback)
        mock_manager.start_listening.assert_called_once_with(callback, error_callback)
        
        # Test stop
        voice_utils.stop_voice_input()
        mock_manager.stop_listening.assert_called_once()
    
    @patch('ollama_workbench.chat.voice_utils.voice_manager')
    def test_voice_profile_management(self, mock_manager):
        """Test voice profile management functions"""
        import ollama_workbench.chat.voice_utils as voice_utils

        
        # Test add profile
        mock_manager.add_voice_profile.return_value = True
        settings = {'provider': 'gtts', 'language': 'es'}
        
        result = voice_utils.add_voice_profile('spanish', settings)
        assert result is True
        mock_manager.add_voice_profile.assert_called_once_with('spanish', settings)
        
        # Test remove profile
        mock_manager.remove_voice_profile.return_value = True
        
        result = voice_utils.remove_voice_profile('spanish')
        assert result is True
        mock_manager.remove_voice_profile.assert_called_once_with('spanish')
    
    @patch('ollama_workbench.chat.voice_utils.voice_manager')
    def test_play_and_stop_speech(self, mock_manager):
        """Test play and stop speech functions"""
        import ollama_workbench.chat.voice_utils as voice_utils

        
        # Test play
        voice_utils.play_speech('/tmp/audio.mp3', block=False)
        mock_manager.play_speech.assert_called_once_with('/tmp/audio.mp3', False)
        
        # Test stop
        voice_utils.stop_speech()
        mock_manager.stop_playback.assert_called_once()


class TestVoiceManagerClass:
    """Test VoiceManager class directly"""
    
    @patch('ollama_workbench.chat.voice_utils.pygame.mixer.init')
    @patch('ollama_workbench.chat.voice_utils.pyaudio.PyAudio')
    @patch('ollama_workbench.chat.voice_utils.sr.Recognizer')
    @patch('ollama_workbench.chat.voice_utils.os.path.exists', return_value=False)
    @patch('builtins.open', create=True)
    @patch('ollama_workbench.chat.voice_utils.json.dump')
    def test_voice_manager_initialization(self, mock_dump, mock_open, mock_exists, 
                                        mock_recognizer, mock_pyaudio, mock_pygame):
        """Test VoiceManager initialization"""
        # Import here to ensure patches are applied
        from ollama_workbench.chat.voice_utils import VoiceManager
        
        manager = VoiceManager()
        
        # Verify initialization
        assert manager.audio is not None
        assert manager.recognizer is not None
        assert manager.is_listening is False
        assert 'default' in manager.voices
        
        # When no settings file exists, it creates default profiles
        assert 'male' in manager.voices
        assert 'female' in manager.voices
    
    @patch('ollama_workbench.chat.voice_utils.pygame.mixer.init')
    @patch('ollama_workbench.chat.voice_utils.pyaudio.PyAudio')
    @patch('ollama_workbench.chat.voice_utils.sr.Recognizer')
    @patch('ollama_workbench.chat.voice_utils.os.path.exists', return_value=False)
    def test_voice_settings_methods(self, mock_exists, mock_recognizer, 
                                   mock_pyaudio, mock_pygame):
        """Test voice settings methods"""
        from ollama_workbench.chat.voice_utils import VoiceManager
        
        manager = VoiceManager()
        
        # Test get_available_voices
        voices = manager.get_available_voices()
        assert 'default' in voices
        assert len(voices) >= 1
        
        # Test get_voice_settings
        settings = manager.get_voice_settings('default')
        assert settings['provider'] == 'gtts'
        assert 'language' in settings
        
        # Test add_voice_profile
        new_profile = {'provider': 'tts_server', 'voice_id': 'test'}
        result = manager.add_voice_profile('test_voice', new_profile)
        assert result is True
        assert 'test_voice' in manager.voices
        
        # Test remove_voice_profile
        result = manager.remove_voice_profile('test_voice')
        assert result is True
        assert 'test_voice' not in manager.voices
        
        # Cannot remove default
        result = manager.remove_voice_profile('default')
        assert result is False
    
    @patch('ollama_workbench.chat.voice_utils.pygame.mixer.init')
    @patch('ollama_workbench.chat.voice_utils.pygame.mixer.get_init', return_value=True)
    @patch('ollama_workbench.chat.voice_utils.pygame.mixer.quit')
    @patch('ollama_workbench.chat.voice_utils.pyaudio.PyAudio')
    @patch('ollama_workbench.chat.voice_utils.sr.Recognizer')
    @patch('ollama_workbench.chat.voice_utils.os.path.exists', return_value=False)
    def test_cleanup(self, mock_exists, mock_recognizer, mock_pyaudio,
                    mock_quit, mock_get_init, mock_pygame):
        """Test cleanup method"""
        from ollama_workbench.chat.voice_utils import VoiceManager
        
        mock_audio_instance = Mock()
        mock_pyaudio.return_value = mock_audio_instance
        
        manager = VoiceManager()
        manager.stream = Mock()
        
        manager.cleanup()
        
        manager.stream.stop_stream.assert_called_once()
        manager.stream.close.assert_called_once()
        mock_audio_instance.terminate.assert_called_once()
        mock_quit.assert_called_once()
    
    @patch('ollama_workbench.chat.voice_utils.pygame.mixer.init')
    @patch('ollama_workbench.chat.voice_utils.pyaudio.PyAudio')
    @patch('ollama_workbench.chat.voice_utils.sr.Recognizer')
    @patch('ollama_workbench.chat.voice_utils.os.path.exists', return_value=False)
    @patch('ollama_workbench.chat.voice_utils.gTTS')
    def test_text_to_speech_gtts(self, mock_gtts, mock_exists,
                                 mock_recognizer, mock_pyaudio, mock_pygame):
        """Test text-to-speech with gTTS"""
        from ollama_workbench.chat.voice_utils import VoiceManager

        # Setup mocks
        mock_tts_instance = Mock()
        mock_gtts.return_value = mock_tts_instance

        manager = VoiceManager()

        # Test TTS
        result = manager.text_to_speech("Hello world")

        # Verify: with no output_file given, a speech_<ts>.mp3 path is generated
        # in the temp directory, passed to gTTS.save, and returned.
        mock_gtts.assert_called_once_with(text="Hello world", lang='en', slow=False)
        mock_tts_instance.save.assert_called_once_with(result)
        assert result.startswith(tempfile.gettempdir())
        assert os.path.basename(result).startswith('speech_')
        assert result.endswith('.mp3')
    
    @patch('ollama_workbench.chat.voice_utils.pygame.mixer.init')
    @patch('ollama_workbench.chat.voice_utils.pyaudio.PyAudio')
    @patch('ollama_workbench.chat.voice_utils.sr.Recognizer')
    @patch('ollama_workbench.chat.voice_utils.os.path.exists', return_value=False)
    @patch('ollama_workbench.chat.voice_utils.pygame.mixer.get_init', return_value=True)
    @patch('ollama_workbench.chat.voice_utils.pygame.mixer.music.load')
    @patch('ollama_workbench.chat.voice_utils.pygame.mixer.music.play')
    @patch('ollama_workbench.chat.voice_utils.pygame.mixer.music.get_busy')
    def test_play_speech(self, mock_get_busy, mock_play, mock_load, mock_get_init,
                        mock_exists, mock_recognizer, mock_pyaudio, mock_pygame):
        """Test play speech functionality"""
        from ollama_workbench.chat.voice_utils import VoiceManager

        mock_get_busy.side_effect = [True, False]  # Busy then not busy

        manager = VoiceManager()

        # The audio file must "exist" for playback (init above needed it absent
        # so the manager creates default profiles instead of reading a file)
        mock_exists.return_value = True

        # Test blocking play
        result = manager.play_speech('/tmp/audio.mp3', block=True)

        assert result is True
        mock_load.assert_called_once_with('/tmp/audio.mp3')
        mock_play.assert_called_once()
        assert mock_get_busy.call_count >= 1
    
    @patch('ollama_workbench.chat.voice_utils.pygame.mixer.init')
    @patch('ollama_workbench.chat.voice_utils.pyaudio.PyAudio')
    @patch('ollama_workbench.chat.voice_utils.sr.Recognizer')
    @patch('ollama_workbench.chat.voice_utils.os.path.exists', return_value=False)
    def test_listening_methods(self, mock_exists, mock_recognizer, mock_pyaudio, mock_pygame):
        """Test start/stop listening methods"""
        from ollama_workbench.chat.voice_utils import VoiceManager
        
        mock_audio = Mock()
        mock_stream = Mock()
        mock_pyaudio.return_value = mock_audio
        mock_audio.open.return_value = mock_stream
        
        manager = VoiceManager()
        
        # Test start listening
        with patch('threading.Thread') as mock_thread:
            callback = Mock()
            error_callback = Mock()

            manager.start_listening(callback, error_callback)

            assert manager.is_listening is True
            assert manager.speech_callback == callback
            assert manager.error_callback == error_callback
            # Recording is driven by the PyAudio stream callback; the only
            # thread the manager spawns is the audio-queue processor.
            mock_thread.assert_called_once_with(target=manager._process_audio_queue)
            mock_thread.return_value.start.assert_called_once()
        
        # Test stop listening
        manager.stop_listening()
        
        assert manager.is_listening is False
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()


class TestConstants:
    """Test module constants"""
    
    def test_default_voice_constant(self):
        """Test DEFAULT_VOICE constant"""
        import ollama_workbench.chat.voice_utils as voice_utils

        
        assert isinstance(voice_utils.DEFAULT_VOICE, dict)
        assert 'provider' in voice_utils.DEFAULT_VOICE
        assert 'language' in voice_utils.DEFAULT_VOICE
        assert voice_utils.DEFAULT_VOICE['provider'] == 'gtts'
    
    def test_audio_constants(self):
        """Test audio-related constants"""
        import ollama_workbench.chat.voice_utils as voice_utils

        
        assert hasattr(voice_utils, 'CHUNK_SIZE')
        assert hasattr(voice_utils, 'FORMAT')
        assert hasattr(voice_utils, 'CHANNELS')
        assert hasattr(voice_utils, 'RATE')
        assert hasattr(voice_utils, 'SILENCE_THRESHOLD')
        assert hasattr(voice_utils, 'MAX_SILENCE_DURATION')
        
        # Verify reasonable values
        assert voice_utils.CHUNK_SIZE > 0
        assert voice_utils.CHANNELS in [1, 2]
        assert voice_utils.RATE > 0
        assert voice_utils.SILENCE_THRESHOLD > 0
        assert voice_utils.MAX_SILENCE_DURATION > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])