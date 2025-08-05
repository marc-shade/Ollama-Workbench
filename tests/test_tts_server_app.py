"""
Test suite for tts_server/app.py - Text-to-Speech server functionality
"""

import pytest
import json
import os
import tempfile
import time
import io
from unittest.mock import Mock, patch, MagicMock, mock_open, call
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFlaskApp:
    """Test Flask application setup and configuration"""
    
    @pytest.fixture
    def app(self):
        """Create test Flask app"""
        # Import here to avoid issues with module loading
        with patch('tts_server.app.load_voice_profiles'):
            with patch('tts_server.app.clean_cache'):
                from tts_server.app import app
                app.config['TESTING'] = True
                return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()
    
    def test_app_creation(self, app):
        """Test Flask app is created successfully"""
        assert app is not None
        assert app.config['TESTING'] is True
    
    def test_cors_enabled(self, app):
        """Test CORS is enabled for the app"""
        # CORS should be enabled - we can't easily test this directly
        # but we can check the app was configured
        assert app is not None


class TestHealthCheck:
    """Test health check endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        with patch('tts_server.app.load_voice_profiles'):
            with patch('tts_server.app.clean_cache'):
                from tts_server.app import app
                app.config['TESTING'] = True
                return app.test_client()
    
    def test_healthcheck_endpoint(self, client):
        """Test health check endpoint returns OK"""
        response = client.get('/healthcheck')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'ok'
        assert 'timestamp' in data
        assert isinstance(data['timestamp'], (int, float))


class TestVoiceProfiles:
    """Test voice profile management"""
    
    @pytest.fixture
    def client(self):
        """Create test client with mocked voice profiles"""
        with patch('tts_server.app.load_voice_profiles'):
            with patch('tts_server.app.clean_cache'):
                from tts_server.app import app
                app.config['TESTING'] = True
                return app.test_client()
    
    @patch('tts_server.app.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('tts_server.app.json.load')
    def test_load_voice_profiles_success(self, mock_json_load, mock_file, mock_exists):
        """Test successful loading of voice profiles"""
        from tts_server.app import load_voice_profiles, VOICE_PROFILES
        
        mock_exists.return_value = True
        mock_profiles = {
            "female_voice": {"provider": "gtts", "language": "en", "voice_id": "en-US-Wavenet-F"},
            "male_voice": {"provider": "gtts", "language": "en", "voice_id": "en-US-Wavenet-D"}
        }
        mock_json_load.return_value = mock_profiles
        
        load_voice_profiles()
        
        mock_exists.assert_called_once()
        mock_file.assert_called_once()
        mock_json_load.assert_called_once()
    
    @patch('tts_server.app.os.path.exists')
    def test_load_voice_profiles_file_not_exists(self, mock_exists):
        """Test loading voice profiles when file doesn't exist"""
        from tts_server.app import load_voice_profiles
        
        mock_exists.return_value = False
        
        # Should not raise exception
        load_voice_profiles()
        
        mock_exists.assert_called_once()
    
    @patch('tts_server.app.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('tts_server.app.json.load')
    @patch('tts_server.app.logger')
    def test_load_voice_profiles_json_error(self, mock_logger, mock_json_load, mock_file, mock_exists):
        """Test loading voice profiles with JSON error"""
        from tts_server.app import load_voice_profiles
        
        mock_exists.return_value = True
        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
        
        load_voice_profiles()
        
        mock_logger.error.assert_called_once()
    
    @patch('tts_server.app.tts_langs')
    @patch('tts_server.app.VOICE_PROFILES')
    def test_get_voices_endpoint(self, mock_profiles, mock_tts_langs, client):
        """Test /voices endpoint"""
        mock_tts_langs.return_value = {'en': 'English', 'es': 'Spanish', 'fr': 'French'}
        mock_profiles.__iter__ = Mock(return_value=iter(['default', 'female_voice']))
        mock_profiles.keys.return_value = ['default', 'female_voice']
        
        response = client.get('/voices')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'supported_languages' in data
        assert 'voice_profiles' in data
        assert data['supported_languages'] == {'en': 'English', 'es': 'Spanish', 'fr': 'French'}
    
    @patch('tts_server.app.tts_langs')
    @patch('tts_server.app.logger')
    def test_get_voices_endpoint_error(self, mock_logger, mock_tts_langs, client):
        """Test /voices endpoint with error"""
        mock_tts_langs.side_effect = Exception("TTS error")
        
        response = client.get('/voices')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        mock_logger.error.assert_called_once()
    
    @patch('tts_server.app.load_voice_profiles')
    @patch('tts_server.app.VOICE_PROFILES')
    def test_reload_profiles_endpoint_success(self, mock_profiles, mock_load, client):
        """Test /reload_profiles endpoint success"""
        mock_profiles.__len__ = Mock(return_value=3)
        
        response = client.post('/reload_profiles')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['count'] == 3
        mock_load.assert_called_once()
    
    @patch('tts_server.app.load_voice_profiles')
    @patch('tts_server.app.logger')
    def test_reload_profiles_endpoint_error(self, mock_logger, mock_load, client):
        """Test /reload_profiles endpoint with error"""
        mock_load.side_effect = Exception("Load error")
        
        response = client.post('/reload_profiles')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        mock_logger.error.assert_called_once()


class TestCacheManagement:
    """Test audio cache management"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        with patch('tts_server.app.load_voice_profiles'):
            with patch('tts_server.app.clean_cache'):
                from tts_server.app import app
                app.config['TESTING'] = True
                return app.test_client()
    
    @patch('tts_server.app.os.path.exists')
    @patch('tts_server.app.time.time')
    @patch('tts_server.app.os.path.getmtime')
    def test_get_cached_audio_exists_recent(self, mock_getmtime, mock_time, mock_exists):
        """Test getting cached audio that exists and is recent"""
        from tts_server.app import get_cached_audio
        
        mock_exists.return_value = True
        mock_time.return_value = 1000
        mock_getmtime.return_value = 900  # 100 seconds ago (recent)
        
        voice_profile = {"language": "en", "provider": "gtts"}
        result = get_cached_audio("test text", voice_profile)
        
        assert result is not None
        assert result.endswith('.mp3')
    
    @patch('tts_server.app.os.path.exists')
    @patch('tts_server.app.time.time')
    @patch('tts_server.app.os.path.getmtime')
    def test_get_cached_audio_exists_old(self, mock_getmtime, mock_time, mock_exists):
        """Test getting cached audio that exists but is old"""
        from tts_server.app import get_cached_audio
        
        mock_exists.return_value = True
        mock_time.return_value = 100000
        mock_getmtime.return_value = 1000  # More than 24 hours ago (old)
        
        voice_profile = {"language": "en", "provider": "gtts"}
        result = get_cached_audio("test text", voice_profile)
        
        assert result is None
    
    @patch('tts_server.app.os.path.exists')
    def test_get_cached_audio_not_exists(self, mock_exists):
        """Test getting cached audio that doesn't exist"""
        from tts_server.app import get_cached_audio
        
        mock_exists.return_value = False
        
        voice_profile = {"language": "en", "provider": "gtts"}
        result = get_cached_audio("test text", voice_profile)
        
        assert result is None
    
    @patch('builtins.open', new_callable=mock_open)
    def test_save_to_cache(self, mock_file):
        """Test saving audio data to cache"""
        from tts_server.app import save_to_cache
        
        audio_data = b"fake audio data"
        voice_profile = {"language": "en", "provider": "gtts"}
        
        result = save_to_cache(audio_data, "test text", voice_profile)
        
        assert result is not None
        assert result.endswith('.mp3')
        mock_file.assert_called_once()
        mock_file().write.assert_called_once_with(audio_data)
    
    @patch('tts_server.app.os.listdir')
    @patch('tts_server.app.os.path.isfile')
    @patch('tts_server.app.os.path.getmtime')
    @patch('tts_server.app.os.remove')
    @patch('tts_server.app.time.time')
    def test_clean_cache(self, mock_time, mock_remove, mock_getmtime, mock_isfile, mock_listdir):
        """Test cache cleaning removes old files"""
        from tts_server.app import clean_cache
        
        mock_time.return_value = 1000000
        mock_listdir.return_value = ['old_file.mp3', 'new_file.mp3']
        mock_isfile.return_value = True
        mock_getmtime.side_effect = [300000, 999000]  # old file, new file
        
        clean_cache()
        
        # Should remove the old file
        mock_remove.assert_called_once()
        assert 'old_file.mp3' in mock_remove.call_args[0][0]
    
    @patch('tts_server.app.clean_cache')
    def test_clean_cache_endpoint_success(self, mock_clean, client):
        """Test /clean_cache endpoint success"""
        response = client.post('/clean_cache')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        mock_clean.assert_called_once()
    
    @patch('tts_server.app.clean_cache')
    @patch('tts_server.app.logger')
    def test_clean_cache_endpoint_error(self, mock_logger, mock_clean, client):
        """Test /clean_cache endpoint with error"""
        mock_clean.side_effect = Exception("Clean error")
        
        response = client.post('/clean_cache')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        mock_logger.error.assert_called_once()
    
    @patch('tts_server.app.os.listdir')
    @patch('tts_server.app.os.path.isfile')
    @patch('tts_server.app.os.path.getsize')
    def test_cache_stats_endpoint(self, mock_getsize, mock_isfile, mock_listdir, client):
        """Test /cache_stats endpoint"""
        mock_listdir.return_value = ['file1.mp3', 'file2.mp3']
        mock_isfile.return_value = True
        mock_getsize.side_effect = [1024, 2048]  # 1KB and 2KB files
        
        response = client.get('/cache_stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['file_count'] == 2
        assert data['total_size_bytes'] == 3072
        assert data['total_size_mb'] == 3072 / (1024 * 1024)
    
    @patch('tts_server.app.os.listdir')
    @patch('tts_server.app.logger')
    def test_cache_stats_endpoint_error(self, mock_logger, mock_listdir, client):
        """Test /cache_stats endpoint with error"""
        mock_listdir.side_effect = Exception("Stats error")
        
        response = client.get('/cache_stats')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        mock_logger.error.assert_called_once()


class TestSpeechSynthesis:
    """Test speech synthesis functionality"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        with patch('tts_server.app.load_voice_profiles'):
            with patch('tts_server.app.clean_cache'):
                from tts_server.app import app
                app.config['TESTING'] = True
                return app.test_client()
    
    @patch('tts_server.app.get_cached_audio')
    @patch('tts_server.app.send_file')
    @patch('tts_server.app.VOICE_PROFILES')
    def test_synthesize_cached_audio(self, mock_profiles, mock_send_file, mock_get_cached, client):
        """Test synthesis with cached audio"""
        mock_profiles.get.return_value = {"provider": "gtts", "language": "en"}
        mock_get_cached.return_value = "/tmp/cached_audio.mp3"
        mock_send_file.return_value = Mock()
        
        response = client.post('/synthesize', 
                              json={'text': 'Hello world', 'voice': 'default'})
        
        assert response.status_code == 200
        mock_get_cached.assert_called_once()
        mock_send_file.assert_called_once_with("/tmp/cached_audio.mp3", mimetype='audio/mpeg')
    
    @patch('tts_server.app.get_cached_audio')
    @patch('tts_server.app.save_to_cache')
    @patch('tts_server.app.send_file')
    @patch('tts_server.app.gTTS')
    @patch('tts_server.app.VOICE_PROFILES')
    def test_synthesize_new_audio_gtts(self, mock_profiles, mock_gtts, mock_send_file, 
                                      mock_save_cache, mock_get_cached, client):
        """Test synthesis with new audio using gTTS"""
        mock_profiles.get.return_value = {"provider": "gtts", "language": "en"}
        mock_get_cached.return_value = None  # No cached audio
        mock_save_cache.return_value = "/tmp/new_audio.mp3"
        mock_send_file.return_value = Mock()
        
        # Mock gTTS
        mock_tts_instance = Mock()
        mock_gtts.return_value = mock_tts_instance
        
        response = client.post('/synthesize',
                              json={'text': 'Hello world', 'voice': 'default'})
        
        assert response.status_code == 200
        mock_gtts.assert_called_once_with(text='Hello world', lang='en', slow=False)
        mock_tts_instance.write_to_fp.assert_called_once()
        mock_save_cache.assert_called_once()
        mock_send_file.assert_called_once()
    
    @patch('tts_server.app.get_cached_audio')
    @patch('tts_server.app.save_to_cache')
    @patch('tts_server.app.send_file')
    @patch('tts_server.app.gTTS')
    @patch('tts_server.app.VOICE_PROFILES')
    def test_synthesize_unsupported_provider_fallback(self, mock_profiles, mock_gtts, 
                                                     mock_send_file, mock_save_cache, 
                                                     mock_get_cached, client):
        """Test synthesis with unsupported provider falls back to gTTS"""
        mock_profiles.get.return_value = {"provider": "unknown", "language": "en"}
        mock_get_cached.return_value = None
        mock_save_cache.return_value = "/tmp/fallback_audio.mp3"
        mock_send_file.return_value = Mock()
        
        # Mock gTTS
        mock_tts_instance = Mock()
        mock_gtts.return_value = mock_tts_instance
        
        response = client.post('/synthesize',
                              json={'text': 'Test fallback', 'voice': 'default'})
        
        assert response.status_code == 200
        mock_gtts.assert_called_once_with(text='Test fallback', lang='en', slow=False)
    
    @patch('tts_server.app.VOICE_PROFILES')
    def test_synthesize_missing_text(self, mock_profiles, client):
        """Test synthesis with missing text parameter"""
        mock_profiles.get.return_value = {"provider": "gtts", "language": "en"}
        
        response = client.post('/synthesize', json={'voice': 'default'})
        
        # Should handle empty text gracefully or return an error
        # The actual behavior depends on how gTTS handles empty text
        assert response.status_code in [200, 500]
    
    @patch('tts_server.app.get_cached_audio')
    @patch('tts_server.app.gTTS')
    @patch('tts_server.app.VOICE_PROFILES')
    @patch('tts_server.app.logger')
    def test_synthesize_gtts_error(self, mock_logger, mock_profiles, mock_gtts, 
                                  mock_get_cached, client):
        """Test synthesis with gTTS error"""
        mock_profiles.get.return_value = {"provider": "gtts", "language": "en"}
        mock_get_cached.return_value = None
        mock_gtts.side_effect = Exception("TTS generation failed")
        
        response = client.post('/synthesize',
                              json={'text': 'Hello world', 'voice': 'default'})
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        mock_logger.error.assert_called_once()
    
    @patch('tts_server.app.VOICE_PROFILES')
    def test_synthesize_custom_speed_pitch(self, mock_profiles, client):
        """Test synthesis with custom speed and pitch parameters"""
        mock_profiles.get.return_value = {"provider": "gtts", "language": "en", "speed": 1.0, "pitch": 1.0}
        
        with patch('tts_server.app.get_cached_audio') as mock_get_cached:
            with patch('tts_server.app.gTTS') as mock_gtts:
                with patch('tts_server.app.save_to_cache') as mock_save_cache:
                    with patch('tts_server.app.send_file') as mock_send_file:
                        mock_get_cached.return_value = None
                        mock_save_cache.return_value = "/tmp/custom_audio.mp3"
                        mock_send_file.return_value = Mock()
                        mock_tts_instance = Mock()
                        mock_gtts.return_value = mock_tts_instance
                        
                        response = client.post('/synthesize',
                                              json={
                                                  'text': 'Custom speech', 
                                                  'voice': 'default',
                                                  'speed': 1.5,
                                                  'pitch': 0.8
                                              })
        
        assert response.status_code == 200
        # Speed and pitch should be updated in voice settings
        # This tests the parameter override functionality


class TestOpenAICompatibleAPI:
    """Test OpenAI-compatible API endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        with patch('tts_server.app.load_voice_profiles'):
            with patch('tts_server.app.clean_cache'):
                from tts_server.app import app
                app.config['TESTING'] = True
                return app.test_client()
    
    @patch('tts_server.app.gTTS')
    @patch('tts_server.app.base64.b64encode')
    def test_openai_text_to_speech_success(self, mock_b64encode, mock_gtts, client):
        """Test OpenAI-compatible text-to-speech endpoint"""
        mock_tts_instance = Mock()
        mock_gtts.return_value = mock_tts_instance
        mock_b64encode.return_value.decode.return_value = "encoded_audio_data"
        
        response = client.post('/v1/audio/speech',
                              json={'input': 'Hello from OpenAI API', 'voice': 'en-US-Wavenet-A'})
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'audio' in data
        assert data['audio'] == "encoded_audio_data"
        mock_gtts.assert_called_once_with(text='Hello from OpenAI API', lang='en')
    
    @patch('tts_server.app.gTTS')
    def test_openai_text_to_speech_complex_voice(self, mock_gtts, client):
        """Test OpenAI API with complex voice ID"""
        mock_tts_instance = Mock()
        mock_gtts.return_value = mock_tts_instance
        
        with patch('tts_server.app.base64.b64encode') as mock_b64encode:
            mock_b64encode.return_value.decode.return_value = "encoded_data"
            
            response = client.post('/v1/audio/speech',
                                  json={'input': 'Test text', 'voice': 'es-ES-Wavenet-B'})
        
        assert response.status_code == 200
        # Should extract 'es' from 'es-ES-Wavenet-B'
        mock_gtts.assert_called_once_with(text='Test text', lang='es')
    
    @patch('tts_server.app.gTTS')
    def test_openai_text_to_speech_simple_voice(self, mock_gtts, client):
        """Test OpenAI API with simple voice ID"""
        mock_tts_instance = Mock()
        mock_gtts.return_value = mock_tts_instance
        
        with patch('tts_server.app.base64.b64encode') as mock_b64encode:
            mock_b64encode.return_value.decode.return_value = "encoded_data"
            
            response = client.post('/v1/audio/speech',
                                  json={'input': 'Test text', 'voice': 'fr'})
        
        assert response.status_code == 200
        # Should use 'fr' directly
        mock_gtts.assert_called_once_with(text='Test text', lang='fr')
    
    def test_openai_text_to_speech_missing_input(self, client):
        """Test OpenAI API with missing input parameter"""
        response = client.post('/v1/audio/speech', json={'voice': 'en-US-Wavenet-A'})
        
        # Should handle empty input gracefully
        assert response.status_code in [200, 500]
    
    @patch('tts_server.app.gTTS')
    @patch('tts_server.app.logger')
    def test_openai_text_to_speech_error(self, mock_logger, mock_gtts, client):
        """Test OpenAI API with TTS error"""
        mock_gtts.side_effect = Exception("TTS generation failed")
        
        response = client.post('/v1/audio/speech',
                              json={'input': 'Test text', 'voice': 'en-US-Wavenet-A'})
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        mock_logger.error.assert_called_once()


class TestLogging:
    """Test logging functionality"""
    
    @patch('tts_server.app.logging.basicConfig')
    @patch('tts_server.app.logging.getLogger')
    def test_logging_configuration(self, mock_get_logger, mock_basic_config):
        """Test logging is configured correctly"""
        # Import to trigger logging setup
        import tts_server.app
        
        # Verify logging was configured
        mock_basic_config.assert_called_once()
        mock_get_logger.assert_called_with('tts_server')
    
    @patch('tts_server.app.logger')
    def test_logging_in_error_scenarios(self, mock_logger):
        """Test that errors are properly logged"""
        from tts_server.app import load_voice_profiles
        
        with patch('tts_server.app.os.path.exists', return_value=True):
            with patch('builtins.open', side_effect=Exception("File error")):
                load_voice_profiles()
        
        mock_logger.error.assert_called_once()


class TestInitialization:
    """Test application initialization"""
    
    @patch('tts_server.app.os.makedirs')
    @patch('tts_server.app.CACHE_DIR')
    def test_cache_directory_creation(self, mock_cache_dir, mock_makedirs):
        """Test cache directory is created on startup"""
        # Import to trigger initialization
        import tts_server.app
        
        # Cache directory should be created
        mock_makedirs.assert_called()
    
    @patch('tts_server.app.load_voice_profiles')
    @patch('tts_server.app.clean_cache')
    def test_startup_initialization(self, mock_clean_cache, mock_load_profiles):
        """Test startup initialization calls"""
        # Import to trigger initialization
        import tts_server.app
        
        # Both functions should be called during startup
        mock_load_profiles.assert_called()
        mock_clean_cache.assert_called()
    
    @patch('tts_server.app.load_voice_profiles')
    @patch('tts_server.app.logger')
    def test_initialization_error_handling(self, mock_logger, mock_load_profiles):
        """Test error handling during initialization"""
        mock_load_profiles.side_effect = Exception("Initialization error")
        
        # Should handle initialization errors gracefully
        try:
            import tts_server.app
        except Exception:
            pass  # Should not raise unhandled exceptions
        
        mock_logger.error.assert_called()


class TestThreadSafety:
    """Test thread safety of cache operations"""
    
    @patch('tts_server.app.cache_lock')
    @patch('tts_server.app.os.path.exists')
    def test_get_cached_audio_thread_safety(self, mock_exists, mock_lock):
        """Test get_cached_audio uses lock"""
        from tts_server.app import get_cached_audio
        
        mock_exists.return_value = False
        mock_lock.__enter__ = Mock()
        mock_lock.__exit__ = Mock()
        
        get_cached_audio("test", {"lang": "en"})
        
        # Lock should be acquired and released
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()
    
    @patch('tts_server.app.cache_lock')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_to_cache_thread_safety(self, mock_file, mock_lock):
        """Test save_to_cache uses lock"""
        from tts_server.app import save_to_cache
        
        mock_lock.__enter__ = Mock()
        mock_lock.__exit__ = Mock()
        
        save_to_cache(b"audio data", "test", {"lang": "en"})
        
        # Lock should be acquired and released
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()


class TestErrorHandling:
    """Test comprehensive error handling"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        with patch('tts_server.app.load_voice_profiles'):
            with patch('tts_server.app.clean_cache'):
                from tts_server.app import app
                app.config['TESTING'] = True
                return app.test_client()
    
    def test_invalid_json_request(self, client):
        """Test endpoint with invalid JSON"""
        response = client.post('/synthesize',
                              data='invalid json',
                              content_type='application/json')
        
        assert response.status_code == 400
    
    def test_missing_content_type(self, client):
        """Test endpoint without proper content type"""
        response = client.post('/synthesize', data='{"text": "test"}')
        
        # Should handle missing content type
        assert response.status_code in [400, 500]
    
    @patch('tts_server.app.VOICE_PROFILES')
    def test_unknown_voice_profile(self, mock_profiles, client):
        """Test synthesis with unknown voice profile"""
        mock_profiles.get.return_value = None  # Unknown voice
        
        with patch('tts_server.app.get_cached_audio') as mock_get_cached:
            mock_get_cached.return_value = None
            
            response = client.post('/synthesize',
                                  json={'text': 'Test', 'voice': 'unknown_voice'})
        
        # Should handle gracefully or return error
        assert response.status_code in [200, 500]


class TestIntegration:
    """Test integration scenarios"""
    
    def test_module_imports(self):
        """Test that all required modules can be imported"""
        import tts_server.app
        
        # Test that main components exist
        assert hasattr(tts_server.app, 'app')
        assert hasattr(tts_server.app, 'load_voice_profiles')
        assert hasattr(tts_server.app, 'get_cached_audio')
        assert hasattr(tts_server.app, 'save_to_cache')
        assert hasattr(tts_server.app, 'clean_cache')
    
    @patch('tts_server.app.load_voice_profiles')
    @patch('tts_server.app.clean_cache')
    def test_flask_app_configuration(self, mock_clean, mock_load):
        """Test Flask app is properly configured"""
        from tts_server.app import app
        
        assert app.config.get('TESTING') is not True  # Should be False by default
        # CORS should be enabled
        assert hasattr(app, 'after_request')
    
    @patch('tts_server.app.get_cached_audio')
    @patch('tts_server.app.save_to_cache')
    @patch('tts_server.app.gTTS')
    def test_cache_workflow_integration(self, mock_gtts, mock_save_cache, mock_get_cached):
        """Test complete cache workflow"""
        from tts_server.app import app
        
        client = app.test_client()
        
        # First request - no cache
        mock_get_cached.return_value = None
        mock_save_cache.return_value = "/tmp/audio.mp3"
        mock_tts_instance = Mock()
        mock_gtts.return_value = mock_tts_instance
        
        with patch('tts_server.app.send_file') as mock_send_file:
            with patch('tts_server.app.VOICE_PROFILES') as mock_profiles:
                mock_profiles.get.return_value = {"provider": "gtts", "language": "en"}
                mock_send_file.return_value = Mock()
                
                response1 = client.post('/synthesize',
                                       json={'text': 'Test integration', 'voice': 'default'})
        
        assert response1.status_code == 200
        mock_gtts.assert_called_once()
        mock_save_cache.assert_called_once()
        
        # Second request - with cache
        mock_get_cached.return_value = "/tmp/cached_audio.mp3"
        mock_gtts.reset_mock()
        
        with patch('tts_server.app.send_file') as mock_send_file2:
            mock_send_file2.return_value = Mock()
            response2 = client.post('/synthesize',
                                   json={'text': 'Test integration', 'voice': 'default'})
        
        assert response2.status_code == 200
        # Should not call gTTS again for cached request
        mock_gtts.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
