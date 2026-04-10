"""
Test suite for ollama_utils.py - Core Ollama API functionality
"""

import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, call
import requests
import numpy as np

# Import the module to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ollama_workbench.providers.ollama_utils import (
    load_api_keys, save_api_keys, load_model_settings, save_model_settings,
    get_ollama_url, get_ollama_client, get_ollama_resource_usage,
    get_available_models, call_ollama_endpoint, check_json_handling,
    get_token_embeddings, check_function_calling, run_tool_test,
    pull_model, show_model_info, remove_model, save_chat_history,
    load_chat_history, update_model_selection, preload_model,
    stop_server, apply_server_settings, start_server,
    apply_model_keep_alive, get_log_file_path, get_new_logs,
    view_last_logs, get_server_logs, get_resource_usage,
    generate_embeddings, get_local_models, log_model_stats,
    get_all_models, call_ollama_cli_verbose, _call_ollama_endpoint_impl
)


class TestFileOperations:
    """Test file loading and saving operations"""
    
    def test_load_api_keys_exists(self, tmp_path):
        """Test loading API keys when file exists"""
        # Create a test file
        test_keys = {"openai": "test-key-123", "groq": "test-key-456"}
        api_file = tmp_path / "api_keys.json"
        with open(api_file, "w") as f:
            json.dump(test_keys, f)

        # Mock the file path and clear cache
        with patch('ollama_workbench.providers.ollama_utils.API_KEYS_FILE', str(api_file)), \
             patch('ollama_workbench.providers.ollama_utils._api_keys_cache', None), \
             patch('ollama_workbench.providers.ollama_utils._api_keys_cache_time', 0):
            loaded_keys = load_api_keys()
            assert loaded_keys == test_keys

    def test_load_api_keys_not_exists(self):
        """Test loading API keys when file doesn't exist"""
        with patch('ollama_workbench.providers.ollama_utils.API_KEYS_FILE', '/nonexistent/api_keys.json'), \
             patch('ollama_workbench.providers.ollama_utils._api_keys_cache', None), \
             patch('ollama_workbench.providers.ollama_utils._api_keys_cache_time', 0):
            loaded_keys = load_api_keys()
            assert loaded_keys == {}

    def test_save_api_keys(self, tmp_path):
        """Test saving API keys"""
        test_keys = {"openai": "test-key-123"}
        api_file = tmp_path / "api_keys.json"

        with patch('ollama_workbench.providers.ollama_utils.API_KEYS_FILE', str(api_file)):
            save_api_keys(test_keys)

            # Verify file was saved correctly
            with open(api_file, "r") as f:
                saved_keys = json.load(f)
            assert saved_keys == test_keys
    
    def test_load_model_settings_exists(self, tmp_path):
        """Test loading model settings when file exists"""
        test_settings = {"model1": {"temp": 0.7}, "model2": {"temp": 0.5}}
        settings_file = tmp_path / "model_settings.json"
        with open(settings_file, "w") as f:
            json.dump(test_settings, f)
        
        with patch('ollama_workbench.providers.ollama_utils.MODEL_SETTINGS_FILE', str(settings_file)):
            loaded_settings = load_model_settings()
            assert loaded_settings == test_settings
    
    def test_save_model_settings(self, tmp_path):
        """Test saving model settings"""
        test_settings = {"model1": {"temp": 0.7}}
        settings_file = tmp_path / "model_settings.json"
        
        with patch('ollama_workbench.providers.ollama_utils.MODEL_SETTINGS_FILE', str(settings_file)):
            save_model_settings(test_settings)
            
            with open(settings_file, "r") as f:
                saved_settings = json.load(f)
            assert saved_settings == test_settings


class TestOllamaConfiguration:
    """Test Ollama configuration functions"""
    
    def test_get_ollama_url_default(self):
        """Test getting default Ollama URL"""
        with patch('ollama_workbench.providers.ollama_utils.get_config', return_value={}):
            url = get_ollama_url()
            assert url == "http://localhost:11434/api"
    
    def test_get_ollama_url_custom_host(self):
        """Test getting Ollama URL with custom host"""
        with patch('ollama_workbench.providers.ollama_utils.get_config', return_value={"OLLAMA_HOST": "http://custom:8080"}):
            url = get_ollama_url()
            assert url == "http://custom:8080/api"
    
    def test_get_ollama_url_adds_http_prefix(self):
        """Test that http:// is added if missing"""
        with patch('ollama_workbench.providers.ollama_utils.get_config', return_value={"OLLAMA_HOST": "localhost:8080"}):
            url = get_ollama_url()
            assert url == "http://localhost:8080/api"
    
    def test_get_ollama_url_adds_default_port(self):
        """Test that default port is added if missing"""
        with patch('ollama_workbench.providers.ollama_utils.get_config', return_value={"OLLAMA_HOST": "http://custom"}):
            url = get_ollama_url()
            assert url == "http://custom:11434/api"


class TestOllamaClient:
    """Test Ollama client initialization"""
    
    @patch('ollama_workbench.providers.ollama_utils.get_config')
    @patch('ollama_workbench.providers.ollama_utils.ollama')
    def test_get_ollama_client_with_client_class(self, mock_ollama, mock_config):
        """Test getting client when Client class is available"""
        mock_config.return_value = {"OLLAMA_HOST": "http://localhost:11434"}
        mock_client = Mock()
        mock_client.list.return_value = {"models": []}
        mock_ollama.Client = Mock(return_value=mock_client)
        
        client = get_ollama_client()
        assert client == mock_client
        mock_ollama.Client.assert_called_once_with(host="http://localhost:11434")
    
    @patch('ollama_workbench.providers.ollama_utils.get_config')
    @patch('ollama_workbench.providers.ollama_utils.ollama')
    def test_get_ollama_client_fallback_module_level(self, mock_ollama, mock_config):
        """Test fallback to module-level functions when Client not available"""
        mock_config.return_value = {"OLLAMA_HOST": "http://localhost:11434"}
        mock_ollama.Client = None  # Simulate older version
        mock_ollama.base_url = None
        mock_ollama.list = Mock(return_value={"models": []})
        
        client = get_ollama_client()
        assert client is None  # Should return None to indicate module-level usage
    
    @patch('ollama_workbench.providers.ollama_utils.get_config')
    @patch('ollama_workbench.providers.ollama_utils.ollama')
    def test_get_ollama_client_error_handling(self, mock_ollama, mock_config):
        """Test client error handling"""
        mock_config.return_value = {"OLLAMA_HOST": "http://localhost:11434"}
        mock_ollama.Client = Mock(side_effect=Exception("Connection failed"))
        
        client = get_ollama_client()
        assert client is None


class TestModelOperations:
    """Test model listing and information operations"""
    
    @patch('ollama_workbench.providers.ollama_utils.get_ollama_client')
    def test_get_available_models_with_client(self, mock_get_client):
        """Test getting models using client"""
        mock_client = Mock()
        mock_client.list.return_value = {
            "models": [
                {"name": "llama2"},
                {"name": "codellama"},
                {"name": "embed-model"}  # Should be filtered out
            ]
        }
        mock_get_client.return_value = mock_client
        
        models = get_available_models()
        assert models == ["llama2", "codellama"]
        assert "embed-model" not in models
    
    @patch('ollama_workbench.providers.ollama_utils.get_ollama_client')
    @patch('ollama_workbench.providers.ollama_utils.ollama')
    def test_get_available_models_module_fallback(self, mock_ollama, mock_get_client):
        """Test getting models using module-level functions"""
        # Clear the cache before testing
        get_available_models.clear()
        
        mock_get_client.return_value = None
        mock_ollama.list = Mock(return_value={
            "models": [
                {"name": "llama2"},
                {"name": "mistral"}
            ]
        })
        
        models = get_available_models()
        assert models == ["llama2", "mistral"]
    
    @patch('ollama_workbench.providers.ollama_utils.get_ollama_client')
    @patch('ollama_workbench.providers.ollama_utils.ollama')
    @patch('requests.get')
    def test_get_available_models_api_fallback(self, mock_requests, mock_ollama, mock_get_client):
        """Test getting models using direct API call"""
        # Clear the cache before testing
        get_available_models.clear()
        
        mock_get_client.return_value = None
        mock_ollama.list = Mock(side_effect=Exception("Module failed"))
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2"},
                {"name": "phi"}
            ]
        }
        mock_requests.return_value = mock_response
        
        models = get_available_models()
        assert models == ["llama2", "phi"]
    
    @patch('ollama_workbench.providers.ollama_utils.get_ollama_client')
    @patch('ollama_workbench.providers.ollama_utils.ollama')
    @patch('requests.get')
    @patch('subprocess.run')
    def test_get_available_models_cli_fallback(self, mock_subprocess, mock_requests, mock_ollama, mock_get_client):
        """Test getting models using CLI as last resort"""
        # Clear the cache before testing
        get_available_models.clear()
        
        mock_get_client.return_value = None
        mock_ollama.list = Mock(side_effect=Exception("Module failed"))
        mock_requests.side_effect = Exception("API failed")
        
        # Mock CLI output
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "NAME\tTAG\tSIZE\nllama2\tlatest\t3.8GB\ncodellama\t13b\t7.4GB"
        mock_subprocess.return_value = mock_result
        
        models = get_available_models()
        assert "llama2" in models
        assert "codellama:13b" in models


class TestCallOllamaEndpoint:
    """Test the main endpoint calling function"""
    
    @patch('ollama_workbench.providers.ollama_utils.get_ollama_client')
    def test_call_endpoint_text_generation(self, mock_get_client):
        """Test basic text generation"""
        mock_client = Mock()
        mock_stream = [
            {"response": "Hello"},
            {"response": " world"},
            {"done": True, "eval_count": 2, "eval_duration": 1000000000}
        ]
        mock_client.generate.return_value = mock_stream
        mock_get_client.return_value = mock_client

        response, context, eval_count, eval_duration, metrics = call_ollama_endpoint(
            model="llama2",
            prompt="Say hello",
            temperature=0.5,
            max_tokens=10
        )

        assert response == "Hello world"
        assert eval_count == 2
        assert eval_duration == 1000000000

    @patch('ollama_workbench.providers.ollama_utils.get_ollama_client')
    def test_call_endpoint_with_image(self, mock_get_client):
        """Test multimodal processing with image"""
        mock_client = Mock()
        mock_client.chat.return_value = {
            "message": {"content": "I see an image"},
            "eval_count": 10,
            "eval_duration": 2000000000
        }
        mock_get_client.return_value = mock_client

        response, context, eval_count, eval_duration, metrics = call_ollama_endpoint(
            model="llava",
            prompt="What's in this image?",
            image="base64encodedimage",
            temperature=0.5
        )

        assert response == "I see an image"
        assert eval_count == 10
    
    @patch('ollama_workbench.providers.ollama_utils.call_ollama_cli_verbose')
    def test_call_endpoint_capture_metrics(self, mock_cli):
        """Test calling with capture_metrics flag"""
        mock_cli.return_value = ("Response", None, 5, 1000000000, {"detailed": "metrics"})
        
        response, context, eval_count, eval_duration, metrics = call_ollama_endpoint(
            model="llama2",
            prompt="Test",
            capture_metrics=True
        )
        
        assert response == "Response"
        assert metrics == {"detailed": "metrics"}
        mock_cli.assert_called_once()


class TestEmbeddings:
    """Test embedding generation functions"""
    
    @patch('ollama_workbench.providers.ollama_utils.get_ollama_client')
    def test_generate_embeddings_ollama(self, mock_get_client):
        """Test generating embeddings with Ollama"""
        mock_client = Mock()
        mock_client.embeddings.return_value = {
            "embedding": [0.1, 0.2, 0.3],
            "total_duration": 1000000000,
            "load_duration": 500000000,
            "prompt_eval_count": 5
        }
        mock_get_client.return_value = mock_client
        
        embedding, total_duration, load_duration, eval_count = generate_embeddings(
            "llama2", "test text"
        )
        
        assert embedding == [0.1, 0.2, 0.3]
        assert total_duration == 1000000000
        assert eval_count == 5
    
    @patch('ollama_workbench.providers.ollama_utils.get_local_embeddings')
    def test_generate_embeddings_groq(self, mock_groq_embeddings):
        """Test generating embeddings with Groq"""
        mock_groq_embeddings.return_value = [0.4, 0.5, 0.6]
        
        from ollama_workbench.providers.ollama_utils import GROQ_MODELS
        test_model = GROQ_MODELS[0] if GROQ_MODELS else "groq-model"
        with patch('ollama_workbench.providers.ollama_utils.GROQ_MODELS', [test_model]):
            embedding = generate_embeddings(test_model, "test text")
            assert embedding == [0.4, 0.5, 0.6]


class TestModelManagement:
    """Test model pulling, showing, and removing"""
    
    @patch('streamlit.progress')
    @patch('streamlit.empty')
    @patch('streamlit.write')
    @patch('ollama_workbench.providers.ollama_utils.get_ollama_client')
    def test_pull_model_with_client(self, mock_get_client, mock_write, mock_empty, mock_progress):
        """Test pulling model with client"""
        mock_client = Mock()
        mock_stream = [
            {"status": "pulling manifest"},
            {"status": "downloading", "completed": 50, "total": 100},
            {"status": "success"}
        ]
        mock_client.pull.return_value = mock_stream
        mock_get_client.return_value = mock_client
        
        mock_progress_bar = Mock()
        mock_progress.return_value = mock_progress_bar
        mock_status = Mock()
        mock_empty.return_value = mock_status
        
        results = pull_model("llama2")
        
        assert len(results) == 3
        assert results[0]["status"] == "pulling manifest"
        mock_progress_bar.progress.assert_any_call(0.5)
        mock_progress_bar.progress.assert_any_call(1.0)
    
    @patch('ollama_workbench.providers.ollama_utils.get_ollama_client')
    def test_show_model_info(self, mock_get_client):
        """Test showing model information"""
        mock_client = Mock()
        mock_info = {
            "name": "llama2",
            "modified_at": "2023-08-01",
            "size": 3825819519
        }
        mock_client.show.return_value = mock_info
        mock_get_client.return_value = mock_client
        
        info = show_model_info("llama2")
        
        assert info["name"] == "llama2"
        assert info["size"] == 3825819519
    
    @patch('ollama_workbench.providers.ollama_utils.get_ollama_client')
    def test_remove_model(self, mock_get_client):
        """Test removing a model"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        result = remove_model("llama2")
        
        assert result["status"] == "success"
        assert "llama2" in result["message"]
        mock_client.delete.assert_called_once_with(model="llama2")


class TestUtilityFunctions:
    """Test various utility functions"""
    
    def test_check_json_handling(self):
        """Test JSON handling check"""
        with patch('ollama_workbench.providers.ollama_utils.call_ollama_endpoint') as mock_call:
            # Valid JSON response
            mock_call.return_value = ('{"name": "John", "age": 30, "city": "New York"}', None, None, None, {})
            assert check_json_handling("llama2", 0.5, 100, 0, 0) is True

            # Invalid JSON response
            mock_call.return_value = ('This is not JSON', None, None, None, {})
            assert check_json_handling("llama2", 0.5, 100, 0, 0) is False

    def test_check_function_calling(self):
        """Test function calling check"""
        with patch('ollama_workbench.providers.ollama_utils.call_ollama_endpoint') as mock_call:
            # Response contains "8" (5 + 3)
            mock_call.return_value = ('def add(a, b): return a + b\nresult = add(5, 3)\nprint(result)  # 8', None, None, None, {})
            assert check_function_calling("llama2", 0.5, 100, 0, 0) is True

            # Response doesn't contain "8"
            mock_call.return_value = ('def add(a, b): return a + b', None, None, None, {})
            assert check_function_calling("llama2", 0.5, 100, 0, 0) is False
    
    def test_save_and_load_chat_history(self, tmp_path):
        """Test saving and loading chat history"""
        test_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        history_file = tmp_path / "chat_history.json"
        
        # Save history
        save_chat_history(test_history, str(history_file))
        
        # Load history
        loaded_history = load_chat_history(str(history_file))
        
        assert loaded_history == test_history
    
    @patch('streamlit.session_state', {})
    def test_update_model_selection(self):
        """Test updating model selection in session state"""
        import streamlit as st
        selected_models = ["llama2", "codellama"]
        update_model_selection(selected_models, "test_key")
        
        assert st.session_state["test_key"] == selected_models
    
    def test_get_log_file_path(self):
        """Test getting platform-specific log file path"""
        # Test macOS
        with patch('platform.system', return_value="Darwin"):
            path = get_log_file_path()
            assert "/.ollama/logs/server.log" in path
        
        # Test Linux
        with patch('platform.system', return_value="Linux"):
            path = get_log_file_path()
            assert path == "/var/log/ollama/server.log"
        
        # Test Windows
        with patch('platform.system', return_value="Windows"):
            with patch.dict(os.environ, {"LOCALAPPDATA": "C:\\Users\\Test\\AppData\\Local"}):
                path = get_log_file_path()
                # Windows path might use forward or backward slashes
                assert "Ollama" in path and "server.log" in path
    
    def test_log_model_stats(self):
        """Test logging model statistics"""
        # Patch at the model_management module level since it's imported inside the function
        with patch('ollama_workbench.models.model_management.log_model_usage') as mock_log:
            result = log_model_stats("llama2", 100, 2.5, "generate")
            assert result is True
            mock_log.assert_called_once_with(
                model_name="llama2",
                tokens_generated=100,
                response_time=2.5,
                operation_type="generate"
            )
        
        # Test when logging module is not available
        with patch.dict('sys.modules', {'ollama_workbench.models.model_management': None}):
            result = log_model_stats("llama2", 100, 2.5, "generate")
            assert result is False


class TestServerOperations:
    """Test server-related operations"""
    
    @patch('subprocess.run')
    def test_stop_server(self, mock_subprocess):
        """Test stopping Ollama server"""
        with patch('streamlit.success') as mock_success:
            stop_server()
            mock_subprocess.assert_called_once()
            mock_success.assert_called_once()
    
    @patch('subprocess.Popen')
    def test_start_server(self, mock_popen):
        """Test starting Ollama server"""
        with patch('streamlit.success') as mock_success:
            start_server()
            mock_popen.assert_called_once_with(["ollama", "serve"])
            mock_success.assert_called_once()
    
    @patch('os.environ', {})
    def test_apply_server_settings(self):
        """Test applying server settings"""
        with patch('streamlit.success') as mock_success:
            apply_server_settings(
                host="http://localhost:8080",
                origins="localhost,127.0.0.1",
                model_dir="/models",
                global_keep_alive="5m",
                max_loaded_models=2,
                num_parallel=4,
                max_queue=10
            )
            
            assert os.environ.get("OLLAMA_HOST") == "http://localhost:8080"
            assert os.environ.get("OLLAMA_ORIGINS") == "http://localhost http://127.0.0.1"
            assert os.environ.get("OLLAMA_MODELS") == "/models"
            assert os.environ.get("OLLAMA_KEEP_ALIVE") == "5m"
            assert os.environ.get("OLLAMA_MAX_LOADED_MODELS") == "2"
            assert os.environ.get("OLLAMA_NUM_PARALLEL") == "4"
            assert os.environ.get("OLLAMA_MAX_QUEUE") == "10"
            mock_success.assert_called_once()


class TestResourceMonitoring:
    """Test resource monitoring functions"""
    
    @patch('psutil.process_iter')
    @patch('requests.get')
    def test_get_ollama_resource_usage(self, mock_requests, mock_process_iter):
        """Test getting Ollama resource usage"""
        # Mock process
        mock_process = Mock()
        mock_process.info = {
            'name': 'ollama',
            'cpu_percent': 25.5,
            'memory_percent': 10.2
        }
        mock_process_iter.return_value = [mock_process]
        
        # Mock server response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests.return_value = mock_response
        
        usage = get_ollama_resource_usage()
        
        assert usage["status"] == "Running"
        assert usage["cpu_usage"] == "25.50%"
        assert usage["memory_usage"] == "10.20%"
    
    @patch('psutil.process_iter')
    def test_get_ollama_resource_usage_not_running(self, mock_process_iter):
        """Test resource usage when Ollama not running"""
        mock_process_iter.return_value = []
        
        usage = get_ollama_resource_usage()
        
        assert usage["status"] == "Not Running"
        assert usage["cpu_usage"] == "N/A"


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_get_all_models(self):
        """Test getting all models from all providers"""
        with patch('ollama_workbench.providers.ollama_utils.get_available_models', return_value=["llama2", "codellama"]):
            all_models = get_all_models()
            
            # Should include Ollama models
            assert "llama2" in all_models
            assert "codellama" in all_models
            
            # Should include other provider models (imported from their modules)
            # Just verify the function returns a list
            assert isinstance(all_models, list)
            assert len(all_models) > 2  # More than just Ollama models


if __name__ == "__main__":
    pytest.main([__file__, "-v"])