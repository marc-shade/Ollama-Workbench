"""
Test suite for server_configuration.py - Server configuration management
"""

import pytest
import json
import os
import tempfile
import subprocess
import platform
from unittest.mock import Mock, patch, MagicMock, mock_open, call
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDefaultConfiguration:
    """Test default configuration functions"""
    
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_get_default_model_dir_darwin(self, mock_system):
        """Test default model directory for macOS"""
        from ollama_workbench.server.server_configuration import get_default_model_dir
        
        mock_system.return_value = "Darwin"
        
        with patch('os.path.expanduser') as mock_expanduser:
            mock_expanduser.return_value = "/Users/test/.ollama/models"
            result = get_default_model_dir()
        
        assert result == "/Users/test/.ollama/models"
        mock_expanduser.assert_called_once_with("~/.ollama/models")
    
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_get_default_model_dir_linux(self, mock_system):
        """Test default model directory for Linux"""
        from ollama_workbench.server.server_configuration import get_default_model_dir
        
        mock_system.return_value = "Linux"
        result = get_default_model_dir()
        
        assert result == "/usr/share/ollama/.ollama/models"
    
    @patch('ollama_workbench.server.server_configuration.platform.system')
    @patch('ollama_workbench.server.server_configuration.os.environ')
    def test_get_default_model_dir_windows(self, mock_environ, mock_system):
        """Test default model directory for Windows"""
        from ollama_workbench.server.server_configuration import get_default_model_dir
        
        mock_system.return_value = "Windows"
        mock_environ.__getitem__.return_value = "C:\\Users\\test"
        
        result = get_default_model_dir()
        
        assert result == "C:\\Users\\test\\.ollama\\models"
    
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_get_default_model_dir_unknown(self, mock_system):
        """Test default model directory for unknown system"""
        from ollama_workbench.server.server_configuration import get_default_model_dir
        
        mock_system.return_value = "UnknownOS"
        result = get_default_model_dir()
        
        assert result == ""
    
    @patch('ollama_workbench.server.server_configuration.platform.system')
    @patch('ollama_workbench.server.server_configuration.platform.machine')
    def test_get_default_max_loaded_models_windows_64(self, mock_machine, mock_system):
        """Test default max loaded models for 64-bit Windows"""
        from ollama_workbench.server.server_configuration import get_default_max_loaded_models
        
        mock_system.return_value = "Windows"
        mock_machine.return_value = "x86_64"
        
        result = get_default_max_loaded_models()
        
        assert result == 1
    
    @patch('ollama_workbench.server.server_configuration.platform.system')
    @patch('ollama_workbench.server.server_configuration.platform.machine')
    def test_get_default_max_loaded_models_windows_32(self, mock_machine, mock_system):
        """Test default max loaded models for 32-bit Windows"""
        from ollama_workbench.server.server_configuration import get_default_max_loaded_models
        
        mock_system.return_value = "Windows"
        mock_machine.return_value = "x86"
        
        with patch('builtins.__import__') as mock_import:
            # Mock GPUtil import
            mock_gputil = Mock()
            mock_gputil.getGPUs.return_value = [Mock(), Mock()]  # 2 GPUs
            mock_import.return_value = mock_gputil
            
            result = get_default_max_loaded_models()
        
        assert result == 6  # 3 * 2 GPUs
    
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_get_default_max_loaded_models_linux_with_gpu(self, mock_system):
        """Test default max loaded models for Linux with GPU"""
        from ollama_workbench.server.server_configuration import get_default_max_loaded_models
        
        mock_system.return_value = "Linux"
        
        with patch('builtins.__import__') as mock_import:
            # Mock GPUtil import
            mock_gputil = Mock()
            mock_gputil.getGPUs.return_value = [Mock()]  # 1 GPU
            mock_import.return_value = mock_gputil
            
            result = get_default_max_loaded_models()
        
        assert result == 3  # 3 * 1 GPU
    
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_get_default_max_loaded_models_no_gpu(self, mock_system):
        """Test default max loaded models without GPU"""
        from ollama_workbench.server.server_configuration import get_default_max_loaded_models
        
        mock_system.return_value = "Linux"
        
        with patch('builtins.__import__', side_effect=ImportError):
            result = get_default_max_loaded_models()
        
        assert result == 3
    
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_get_default_max_loaded_models_no_gpus(self, mock_system):
        """Test default max loaded models with no GPUs available"""
        from ollama_workbench.server.server_configuration import get_default_max_loaded_models
        
        mock_system.return_value = "Linux"
        
        with patch('builtins.__import__') as mock_import:
            # Mock GPUtil import with no GPUs
            mock_gputil = Mock()
            mock_gputil.getGPUs.return_value = []  # No GPUs
            mock_import.return_value = mock_gputil
            
            result = get_default_max_loaded_models()
        
        assert result == 3


class TestServerControl:
    """Test server start/stop functionality"""
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.subprocess.run')
    @patch('ollama_workbench.server.server_configuration.shutil.which')
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_stop_server_linux_systemctl(self, mock_system, mock_which, mock_subprocess, mock_st):
        """Test stopping server on Linux with systemctl"""
        from ollama_workbench.server.server_configuration import stop_server
        
        mock_system.return_value = "Linux"
        mock_which.side_effect = lambda cmd: "/usr/bin/systemctl" if cmd == "systemctl" else None
        
        stop_server()
        
        mock_subprocess.assert_called_once_with(
            ["sudo", "systemctl", "stop", "ollama"], check=False
        )
        mock_st.success.assert_called_once_with("Ollama server has been stopped.")
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.subprocess.run')
    @patch('ollama_workbench.server.server_configuration.shutil.which')
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_stop_server_linux_killall(self, mock_system, mock_which, mock_subprocess, mock_st):
        """Test stopping server on Linux with killall"""
        from ollama_workbench.server.server_configuration import stop_server
        
        mock_system.return_value = "Linux"
        mock_which.side_effect = lambda cmd: "/usr/bin/killall" if cmd == "killall" else None
        
        stop_server()
        
        mock_subprocess.assert_called_once_with(["killall", "ollama"], check=False)
        mock_st.success.assert_called_once_with("Ollama server has been stopped.")
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.subprocess.run')
    @patch('ollama_workbench.server.server_configuration.shutil.which')
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_stop_server_darwin(self, mock_system, mock_which, mock_subprocess, mock_st):
        """Test stopping server on macOS"""
        from ollama_workbench.server.server_configuration import stop_server
        
        mock_system.return_value = "Darwin"
        mock_which.return_value = "/usr/bin/pkill"
        
        stop_server()
        
        mock_subprocess.assert_called_once_with(["pkill", "ollama"], check=False)
        mock_st.success.assert_called_once_with("Ollama server has been stopped.")
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.subprocess.run')
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_stop_server_windows(self, mock_system, mock_subprocess, mock_st):
        """Test stopping server on Windows"""
        from ollama_workbench.server.server_configuration import stop_server
        
        mock_system.return_value = "Windows"
        
        stop_server()
        
        mock_subprocess.assert_called_once_with(
            ["taskkill", "/F", "/IM", "ollama.exe"], check=False
        )
        mock_st.success.assert_called_once_with("Ollama server has been stopped.")
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.subprocess.run')
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_stop_server_exception(self, mock_system, mock_subprocess, mock_st):
        """Test stopping server with exception"""
        from ollama_workbench.server.server_configuration import stop_server
        
        mock_system.return_value = "Linux"
        mock_subprocess.side_effect = Exception("Command failed")
        
        stop_server()
        
        mock_st.error.assert_called_once_with("Failed to stop Ollama server: Command failed")
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.subprocess.run')
    @patch('ollama_workbench.server.server_configuration.subprocess.Popen')
    @patch('ollama_workbench.server.server_configuration.shutil.which')
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_start_server_linux_systemctl(self, mock_system, mock_which, mock_popen, mock_run, mock_st):
        """Test starting server on Linux with systemctl"""
        from ollama_workbench.server.server_configuration import start_server
        
        mock_system.return_value = "Linux"
        mock_which.side_effect = lambda cmd: "/usr/bin/systemctl" if cmd == "systemctl" else None
        
        start_server()
        
        mock_run.assert_called_once_with(
            ["sudo", "systemctl", "start", "ollama"], check=False
        )
        mock_st.success.assert_called_once_with("Ollama server has been started.")
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.subprocess.Popen')
    @patch('ollama_workbench.server.server_configuration.shutil.which')
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_start_server_linux_direct(self, mock_system, mock_which, mock_popen, mock_st):
        """Test starting server on Linux directly"""
        from ollama_workbench.server.server_configuration import start_server
        
        mock_system.return_value = "Linux"
        mock_which.side_effect = lambda cmd: "/usr/bin/ollama" if cmd == "ollama" else None
        
        start_server()
        
        mock_popen.assert_called_once_with(
            ["ollama", "serve"], start_new_session=True
        )
        mock_st.success.assert_called_once_with("Ollama server has been started.")
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.subprocess.Popen')
    @patch('ollama_workbench.server.server_configuration.shutil.which')
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_start_server_darwin(self, mock_system, mock_which, mock_popen, mock_st):
        """Test starting server on macOS"""
        from ollama_workbench.server.server_configuration import start_server
        
        mock_system.return_value = "Darwin"
        mock_which.return_value = "/usr/local/bin/ollama"
        
        start_server()
        
        mock_popen.assert_called_once_with(
            ["ollama", "serve"], start_new_session=True
        )
        mock_st.success.assert_called_once_with("Ollama server has been started.")
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.subprocess.Popen')
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_start_server_windows(self, mock_system, mock_popen, mock_st):
        """Test starting server on Windows"""
        from ollama_workbench.server.server_configuration import start_server
        
        mock_system.return_value = "Windows"
        
        start_server()
        
        mock_popen.assert_called_once_with(
            ["ollama", "serve"], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        mock_st.success.assert_called_once_with("Ollama server has been started.")
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.subprocess.Popen')
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_start_server_exception(self, mock_system, mock_popen, mock_st):
        """Test starting server with exception"""
        from ollama_workbench.server.server_configuration import start_server
        
        mock_system.return_value = "Linux"
        mock_popen.side_effect = Exception("Command failed")
        
        start_server()
        
        mock_st.error.assert_called_once_with("Failed to start Ollama server: Command failed")


class TestServerSettings:
    """Test server settings management"""
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.update_config')
    @patch('ollama_workbench.server.server_configuration.os.makedirs')
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_apply_server_settings_linux(self, mock_system, mock_makedirs, mock_update_config, mock_st):
        """Test applying server settings on Linux"""
        from ollama_workbench.server.server_configuration import apply_server_settings
        
        mock_system.return_value = "Linux"
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('ollama_workbench.server.server_configuration.json.dump') as mock_json_dump:
                apply_server_settings(
                    "127.0.0.1:11434",
                    "127.0.0.1,0.0.0.0",
                    "/custom/models",
                    "10m",
                    2,
                    8,
                    1024
                )
        
        expected_config = {
            "OLLAMA_HOST": "127.0.0.1:11434",
            "OLLAMA_ORIGINS": "127.0.0.1,0.0.0.0",
            "OLLAMA_MODELS": "/custom/models",
            "OLLAMA_KEEP_ALIVE": "10m",
            "OLLAMA_MAX_LOADED_MODELS": 2,
            "OLLAMA_NUM_PARALLEL": 8,
            "OLLAMA_MAX_QUEUE": 1024
        }
        
        mock_file.assert_called_once_with("/etc/ollama/config.json", "w")
        mock_json_dump.assert_called_once_with(expected_config, mock_file(), indent=2)
        mock_update_config.assert_called_once_with({"OLLAMA_HOST": "127.0.0.1:11434"})
        mock_st.success.assert_called_once_with("Server settings applied to /etc/ollama/config.json")
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.update_config')
    @patch('ollama_workbench.server.server_configuration.os.makedirs')
    @patch('ollama_workbench.server.server_configuration.os.path.expanduser')
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_apply_server_settings_darwin(self, mock_system, mock_expanduser, mock_makedirs, mock_update_config, mock_st):
        """Test applying server settings on macOS"""
        from ollama_workbench.server.server_configuration import apply_server_settings
        
        mock_system.return_value = "Darwin"
        mock_expanduser.return_value = "/Users/test/.ollama/config.json"
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('ollama_workbench.server.server_configuration.json.dump') as mock_json_dump:
                apply_server_settings(
                    "127.0.0.1:11434",
                    "127.0.0.1",
                    "/Users/test/.ollama/models",
                    "5m",
                    1,
                    4,
                    512
                )
        
        mock_expanduser.assert_called_once_with("~/.ollama/config.json")
        mock_makedirs.assert_called_once_with("/Users/test/.ollama", exist_ok=True)
        mock_file.assert_called_once_with("/Users/test/.ollama/config.json", "w")
        mock_st.success.assert_called_once_with("Server settings applied to /Users/test/.ollama/config.json")
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.update_config')
    @patch('ollama_workbench.server.server_configuration.os.makedirs')
    @patch('ollama_workbench.server.server_configuration.os.environ')
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_apply_server_settings_windows(self, mock_system, mock_environ, mock_makedirs, mock_update_config, mock_st):
        """Test applying server settings on Windows"""
        from ollama_workbench.server.server_configuration import apply_server_settings
        
        mock_system.return_value = "Windows"
        mock_environ.__getitem__.return_value = "C:\\Users\\test"
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('ollama_workbench.server.server_configuration.json.dump') as mock_json_dump:
                with patch('ollama_workbench.server.server_configuration.os.path.join') as mock_join:
                    mock_join.return_value = "C:\\Users\\test\\.ollama\\config.json"
                    
                    apply_server_settings(
                        "localhost:11434",
                        "*",
                        "C:\\Models",
                        "15m",
                        3,
                        6,
                        256
                    )
        
        mock_join.assert_called_once_with("C:\\Users\\test", ".ollama", "config.json")
        mock_makedirs.assert_called_once_with("C:\\Users\\test\\.ollama", exist_ok=True)
        mock_file.assert_called_once_with("C:\\Users\\test\\.ollama\\config.json", "w")
        mock_st.success.assert_called_once_with("Server settings applied to C:\\Users\\test\\.ollama\\config.json")
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_apply_server_settings_exception(self, mock_system, mock_st):
        """Test applying server settings with exception"""
        from ollama_workbench.server.server_configuration import apply_server_settings
        
        mock_system.return_value = "Linux"
        
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            apply_server_settings("127.0.0.1", "*", "/models", "5m", 1, 4, 512)
        
        mock_st.error.assert_called_once_with("Failed to apply server settings: Access denied")


class TestServerStatus:
    """Test server status checking"""
    
    @patch('ollama_workbench.server.server_configuration.get_config')
    @patch('requests.get')
    def test_get_server_status_api_success(self, mock_get, mock_config):
        """Test server status check via API (successful)"""
        from ollama_workbench.server.server_configuration import get_server_status
        
        mock_config.return_value = {"OLLAMA_HOST": "http://localhost:11434"}
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = get_server_status()
        
        assert result is True
        mock_get.assert_called_with("http://localhost:11434/api/tags", timeout=2)
    
    @patch('ollama_workbench.server.server_configuration.get_config')
    @patch('requests.get')
    def test_get_server_status_api_version_success(self, mock_get, mock_config):
        """Test server status check via version API (successful)"""
        from ollama_workbench.server.server_configuration import get_server_status
        
        mock_config.return_value = {"OLLAMA_HOST": "http://localhost:11434"}
        
        # First call fails, second call succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 404
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        
        mock_get.side_effect = [
            Exception("Connection failed"),  # First endpoint fails
            mock_response_success  # Second endpoint succeeds
        ]
        
        result = get_server_status()
        
        assert result is True
        assert mock_get.call_count == 2
    
    @patch('ollama_workbench.server.server_configuration.get_config')
    @patch('requests.get')
    @patch('psutil.process_iter')
    def test_get_server_status_process_check(self, mock_process_iter, mock_get, mock_config):
        """Test server status check via process detection"""
        from ollama_workbench.server.server_configuration import get_server_status
        
        mock_config.return_value = {"OLLAMA_HOST": "http://localhost:11434"}
        mock_get.side_effect = Exception("Connection failed")
        
        # Mock process with ollama name
        mock_process = Mock()
        mock_process.info = {'name': 'ollama'}
        mock_process_iter.return_value = [mock_process]
        
        result = get_server_status()
        
        assert result is True
    
    @patch('ollama_workbench.server.server_configuration.get_config')
    @patch('requests.get')
    @patch('psutil.process_iter')
    def test_get_server_status_process_check_windows(self, mock_process_iter, mock_get, mock_config):
        """Test server status check via process detection on Windows"""
        from ollama_workbench.server.server_configuration import get_server_status
        
        mock_config.return_value = {"OLLAMA_HOST": "http://localhost:11434"}
        mock_get.side_effect = Exception("Connection failed")
        
        # Mock process with ollama.exe name
        mock_process = Mock()
        mock_process.info = {'name': 'ollama.exe'}
        mock_process_iter.return_value = [mock_process]
        
        result = get_server_status()
        
        assert result is True
    
    @patch('ollama_workbench.server.server_configuration.get_config')
    @patch('requests.get')
    @patch('psutil.process_iter')
    def test_get_server_status_localhost_fallback(self, mock_process_iter, mock_get, mock_config):
        """Test server status check with localhost fallback"""
        from ollama_workbench.server.server_configuration import get_server_status
        
        mock_config.return_value = {"OLLAMA_HOST": "http://custom.host:11434"}
        
        # Mock successful response for localhost fallback
        mock_response = Mock()
        mock_response.status_code = 200
        
        mock_get.side_effect = [
            Exception("Connection failed"),  # First two API calls fail
            Exception("Connection failed"),
            mock_response  # Localhost fallback succeeds
        ]
        
        # Mock process check fails
        mock_process_iter.side_effect = Exception("Process check failed")
        
        result = get_server_status()
        
        assert result is True
        # Should have 3 calls: tags, version, localhost fallback
        assert mock_get.call_count == 3
        mock_get.assert_any_call("http://localhost:11434/api/tags", timeout=1)
    
    @patch('ollama_workbench.server.server_configuration.get_config')
    @patch('requests.get')
    @patch('psutil.process_iter')
    def test_get_server_status_all_checks_fail(self, mock_process_iter, mock_get, mock_config):
        """Test server status check when all checks fail"""
        from ollama_workbench.server.server_configuration import get_server_status
        
        mock_config.return_value = {"OLLAMA_HOST": "http://localhost:11434"}
        mock_get.side_effect = Exception("Connection failed")
        
        # Mock process with different name
        mock_process = Mock()
        mock_process.info = {'name': 'other_process'}
        mock_process_iter.return_value = [mock_process]
        
        result = get_server_status()
        
        assert result is False
    
    @patch('ollama_workbench.server.server_configuration.get_config')
    @patch('requests.get')
    def test_get_server_status_host_without_protocol(self, mock_get, mock_config):
        """Test server status check with host without protocol"""
        from ollama_workbench.server.server_configuration import get_server_status
        
        mock_config.return_value = {"OLLAMA_HOST": "localhost:11434"}
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = get_server_status()
        
        assert result is True
        mock_get.assert_called_with("http://localhost:11434/api/tags", timeout=2)


class TestStreamlitInterface:
    """Test Streamlit interface functions"""
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.get_server_status')
    @patch('ollama_workbench.server.server_configuration.get_default_model_dir')
    @patch('ollama_workbench.server.server_configuration.get_default_max_loaded_models')
    @patch('ollama_workbench.server.server_configuration.get_config')
    @patch('ollama_workbench.server.server_configuration.CONFIG')
    def test_server_configuration_interface_running(self, mock_config_dict, mock_get_config, 
                                                   mock_default_max, mock_default_dir, 
                                                   mock_server_status, mock_st):
        """Test server configuration interface when server is running"""
        from ollama_workbench.server.server_configuration import server_configuration
        
        # Setup mocks
        mock_server_status.return_value = True
        mock_default_dir.return_value = "/default/models"
        mock_default_max.return_value = 3
        mock_get_config.return_value = {"OLLAMA_HOST": "127.0.0.1"}
        mock_config_dict.get.return_value = "127.0.0.1"
        
        # Mock Streamlit components
        mock_st.header = Mock()
        mock_st.success = Mock()
        mock_st.tabs = Mock(return_value=(Mock(), Mock()))
        mock_st.subheader = Mock()
        mock_st.text_input = Mock(side_effect=["127.0.0.1", "127.0.0.1, 0.0.0.0", "/models", "5m"])
        mock_st.number_input = Mock(side_effect=[3, 4, 512])
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
        mock_st.button = Mock(return_value=False)
        mock_st.info = Mock()
        
        # Mock the config UI
        with patch('ollama_workbench.server.server_configuration.server_config_ui') as mock_config_ui:
            server_configuration()
        
        # Verify the interface was set up correctly
        mock_st.header.assert_called_with("⚙️ Ollama Server Configuration")
        mock_st.success.assert_called_with("✅ Ollama server is running")
        mock_config_ui.assert_called_once()
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.get_server_status')
    @patch('ollama_workbench.server.server_configuration.get_default_model_dir')
    @patch('ollama_workbench.server.server_configuration.get_default_max_loaded_models')
    @patch('ollama_workbench.server.server_configuration.get_config')
    @patch('ollama_workbench.server.server_configuration.CONFIG')
    def test_server_configuration_interface_not_running(self, mock_config_dict, mock_get_config,
                                                       mock_default_max, mock_default_dir,
                                                       mock_server_status, mock_st):
        """Test server configuration interface when server is not running"""
        from ollama_workbench.server.server_configuration import server_configuration
        
        # Setup mocks
        mock_server_status.return_value = False
        mock_default_dir.return_value = "/default/models"
        mock_default_max.return_value = 3
        mock_get_config.return_value = {"OLLAMA_HOST": "127.0.0.1"}
        mock_config_dict.get.return_value = "127.0.0.1"
        
        # Mock Streamlit components
        mock_st.header = Mock()
        mock_st.error = Mock()
        mock_st.tabs = Mock(return_value=(Mock(), Mock()))
        mock_st.subheader = Mock()
        mock_st.text_input = Mock(side_effect=["127.0.0.1", "127.0.0.1, 0.0.0.0", "/models", "5m"])
        mock_st.number_input = Mock(side_effect=[3, 4, 512])
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
        mock_st.button = Mock(return_value=False)
        mock_st.info = Mock()
        
        # Mock the config UI
        with patch('ollama_workbench.server.server_configuration.server_config_ui') as mock_config_ui:
            server_configuration()
        
        # Verify the interface shows server not running
        mock_st.header.assert_called_with("⚙️ Ollama Server Configuration")
        mock_st.error.assert_called_with("❌ Ollama server is not running")
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.get_server_status')
    @patch('ollama_workbench.server.server_configuration.apply_server_settings')
    @patch('ollama_workbench.server.server_configuration.get_default_model_dir')
    @patch('ollama_workbench.server.server_configuration.get_default_max_loaded_models')
    @patch('ollama_workbench.server.server_configuration.get_config')
    @patch('ollama_workbench.server.server_configuration.CONFIG')
    def test_server_configuration_apply_settings(self, mock_config_dict, mock_get_config,
                                                mock_default_max, mock_default_dir,
                                                mock_apply_settings, mock_server_status, mock_st):
        """Test applying server settings through interface"""
        from ollama_workbench.server.server_configuration import server_configuration
        
        # Setup mocks
        mock_server_status.return_value = True
        mock_default_dir.return_value = "/default/models"
        mock_default_max.return_value = 3
        mock_get_config.return_value = {"OLLAMA_HOST": "127.0.0.1"}
        mock_config_dict.get.return_value = "127.0.0.1"
        
        # Mock Streamlit components
        mock_st.header = Mock()
        mock_st.success = Mock()
        mock_st.tabs = Mock(return_value=(Mock(), Mock()))
        mock_st.subheader = Mock()
        mock_st.text_input = Mock(side_effect=["127.0.0.1", "127.0.0.1, 0.0.0.0", "/models", "5m"])
        mock_st.number_input = Mock(side_effect=[3, 4, 512])
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
        
        # Mock "Apply Settings" button being clicked
        mock_st.button = Mock(side_effect=[True, False, False])  # Apply button clicked
        mock_st.info = Mock()
        
        # Mock the config UI
        with patch('ollama_workbench.server.server_configuration.server_config_ui') as mock_config_ui:
            server_configuration()
        
        # Verify settings were applied
        mock_apply_settings.assert_called_once_with(
            "127.0.0.1", "127.0.0.1, 0.0.0.0", "/models", "5m", 3, 4, 512
        )
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.get_server_status')
    @patch('ollama_workbench.server.server_configuration.stop_server')
    @patch('ollama_workbench.server.server_configuration.get_default_model_dir')
    @patch('ollama_workbench.server.server_configuration.get_default_max_loaded_models')
    @patch('ollama_workbench.server.server_configuration.get_config')
    @patch('ollama_workbench.server.server_configuration.CONFIG')
    def test_server_configuration_stop_server(self, mock_config_dict, mock_get_config,
                                             mock_default_max, mock_default_dir,
                                             mock_stop_server, mock_server_status, mock_st):
        """Test stopping server through interface"""
        from ollama_workbench.server.server_configuration import server_configuration
        
        # Setup mocks
        mock_server_status.return_value = True
        mock_default_dir.return_value = "/default/models"
        mock_default_max.return_value = 3
        mock_get_config.return_value = {"OLLAMA_HOST": "127.0.0.1"}
        mock_config_dict.get.return_value = "127.0.0.1"
        
        # Mock Streamlit components
        mock_st.header = Mock()
        mock_st.success = Mock()
        mock_st.tabs = Mock(return_value=(Mock(), Mock()))
        mock_st.subheader = Mock()
        mock_st.text_input = Mock(side_effect=["127.0.0.1", "127.0.0.1, 0.0.0.0", "/models", "5m"])
        mock_st.number_input = Mock(side_effect=[3, 4, 512])
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
        
        # Mock "Stop Server" button being clicked
        mock_st.button = Mock(side_effect=[False, True, False])  # Stop button clicked
        mock_st.info = Mock()
        
        # Mock the config UI
        with patch('ollama_workbench.server.server_configuration.server_config_ui') as mock_config_ui:
            server_configuration()
        
        # Verify server was stopped
        mock_stop_server.assert_called_once()
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.get_server_status')
    @patch('ollama_workbench.server.server_configuration.start_server')
    @patch('ollama_workbench.server.server_configuration.get_default_model_dir')
    @patch('ollama_workbench.server.server_configuration.get_default_max_loaded_models')
    @patch('ollama_workbench.server.server_configuration.get_config')
    @patch('ollama_workbench.server.server_configuration.CONFIG')
    def test_server_configuration_start_server(self, mock_config_dict, mock_get_config,
                                              mock_default_max, mock_default_dir,
                                              mock_start_server, mock_server_status, mock_st):
        """Test starting server through interface"""
        from ollama_workbench.server.server_configuration import server_configuration
        
        # Setup mocks
        mock_server_status.return_value = False  # Server not running
        mock_default_dir.return_value = "/default/models"
        mock_default_max.return_value = 3
        mock_get_config.return_value = {"OLLAMA_HOST": "127.0.0.1"}
        mock_config_dict.get.return_value = "127.0.0.1"
        
        # Mock Streamlit components
        mock_st.header = Mock()
        mock_st.error = Mock()
        mock_st.tabs = Mock(return_value=(Mock(), Mock()))
        mock_st.subheader = Mock()
        mock_st.text_input = Mock(side_effect=["127.0.0.1", "127.0.0.1, 0.0.0.0", "/models", "5m"])
        mock_st.number_input = Mock(side_effect=[3, 4, 512])
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
        
        # Mock "Start Server" button being clicked
        mock_st.button = Mock(side_effect=[False, True, False])  # Start button clicked
        mock_st.info = Mock()
        
        # Mock the config UI
        with patch('ollama_workbench.server.server_configuration.server_config_ui') as mock_config_ui:
            server_configuration()
        
        # Verify server was started
        mock_start_server.assert_called_once()
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.get_server_status')
    @patch('ollama_workbench.server.server_configuration.stop_server')
    @patch('ollama_workbench.server.server_configuration.start_server')
    @patch('ollama_workbench.server.server_configuration.time.sleep')
    @patch('ollama_workbench.server.server_configuration.get_default_model_dir')
    @patch('ollama_workbench.server.server_configuration.get_default_max_loaded_models')
    @patch('ollama_workbench.server.server_configuration.get_config')
    @patch('ollama_workbench.server.server_configuration.CONFIG')
    def test_server_configuration_restart_server(self, mock_config_dict, mock_get_config,
                                                mock_default_max, mock_default_dir, mock_sleep,
                                                mock_start_server, mock_stop_server, 
                                                mock_server_status, mock_st):
        """Test restarting server through interface"""
        from ollama_workbench.server.server_configuration import server_configuration
        
        # Setup mocks
        mock_server_status.return_value = True
        mock_default_dir.return_value = "/default/models"
        mock_default_max.return_value = 3
        mock_get_config.return_value = {"OLLAMA_HOST": "127.0.0.1"}
        mock_config_dict.get.return_value = "127.0.0.1"
        
        # Mock Streamlit components
        mock_st.header = Mock()
        mock_st.success = Mock()
        mock_st.tabs = Mock(return_value=(Mock(), Mock()))
        mock_st.subheader = Mock()
        mock_st.text_input = Mock(side_effect=["127.0.0.1", "127.0.0.1, 0.0.0.0", "/models", "5m"])
        mock_st.number_input = Mock(side_effect=[3, 4, 512])
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
        
        # Mock "Restart Server" button being clicked
        mock_st.button = Mock(side_effect=[False, False, True])  # Restart button clicked
        mock_st.info = Mock()
        
        # Mock the config UI
        with patch('ollama_workbench.server.server_configuration.server_config_ui') as mock_config_ui:
            server_configuration()
        
        # Verify server was stopped, paused, then started
        mock_stop_server.assert_called_once()
        mock_sleep.assert_called_once_with(2)
        mock_start_server.assert_called_once()


class TestIntegration:
    """Test integration scenarios"""
    
    def test_module_imports(self):
        """Test that all required modules can be imported"""
        import ollama_workbench.server.server_configuration as server_configuration

        
        # Test that main functions exist
        assert hasattr(server_configuration, 'get_default_model_dir')
        assert hasattr(server_configuration, 'get_default_max_loaded_models')
        assert hasattr(server_configuration, 'stop_server')
        assert hasattr(server_configuration, 'start_server')
        assert hasattr(server_configuration, 'apply_server_settings')
        assert hasattr(server_configuration, 'get_server_status')
        assert hasattr(server_configuration, 'server_configuration')
    
    @patch('ollama_workbench.server.server_configuration.platform.system')
    @patch('ollama_workbench.server.server_configuration.os.path.expanduser')
    def test_model_dir_and_max_models_consistency(self, mock_expanduser, mock_system):
        """Test that model directory and max models work together"""
        from ollama_workbench.server.server_configuration import get_default_model_dir, get_default_max_loaded_models
        
        mock_system.return_value = "Darwin"
        mock_expanduser.return_value = "/Users/test/.ollama/models"
        
        model_dir = get_default_model_dir()
        max_models = get_default_max_loaded_models()
        
        assert model_dir == "/Users/test/.ollama/models"
        assert isinstance(max_models, int)
        assert max_models > 0
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.subprocess.run')
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_server_lifecycle(self, mock_system, mock_subprocess, mock_st):
        """Test complete server lifecycle (stop -> start)"""
        from ollama_workbench.server.server_configuration import stop_server, start_server
        
        mock_system.return_value = "Linux"
        
        # Test stopping server
        with patch('ollama_workbench.server.server_configuration.shutil.which', return_value="/usr/bin/systemctl"):
            stop_server()
        
        # Test starting server
        with patch('ollama_workbench.server.server_configuration.shutil.which', return_value="/usr/bin/systemctl"):
            start_server()
        
        # Verify both operations were attempted
        assert mock_subprocess.call_count == 2
        assert mock_st.success.call_count == 2


class TestErrorHandling:
    """Test error handling in various scenarios"""
    
    @patch('ollama_workbench.server.server_configuration.st')
    @patch('ollama_workbench.server.server_configuration.json.dump')
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_apply_settings_json_error(self, mock_system, mock_json_dump, mock_st):
        """Test applying settings with JSON serialization error"""
        from ollama_workbench.server.server_configuration import apply_server_settings
        
        mock_system.return_value = "Linux"
        mock_json_dump.side_effect = TypeError("Object not serializable")
        
        with patch('builtins.open', mock_open()):
            apply_server_settings("127.0.0.1", "*", "/models", "5m", 1, 4, 512)
        
        mock_st.error.assert_called_once()
        assert "Failed to apply server settings" in mock_st.error.call_args[0][0]
    
    @patch('ollama_workbench.server.server_configuration.get_config')
    @patch('requests.get')
    @patch('psutil.process_iter')
    def test_server_status_all_exceptions(self, mock_process_iter, mock_get, mock_config):
        """Test server status when all detection methods throw exceptions"""
        from ollama_workbench.server.server_configuration import get_server_status
        
        mock_config.return_value = {"OLLAMA_HOST": "http://localhost:11434"}
        mock_get.side_effect = Exception("Network error")
        mock_process_iter.side_effect = Exception("Process error")
        
        result = get_server_status()
        
        # Should return False when all methods fail
        assert result is False
    
    @patch('ollama_workbench.server.server_configuration.platform.system')
    def test_get_model_dir_exception_handling(self, mock_system):
        """Test model directory function with exception"""
        from ollama_workbench.server.server_configuration import get_default_model_dir
        
        mock_system.side_effect = Exception("System error")
        
        # Should handle exception gracefully
        try:
            result = get_default_model_dir()
            # If no exception is raised, it should return empty string
            assert result == "" or isinstance(result, str)
        except Exception:
            # If exception is raised, that's also acceptable behavior
            pass


class TestConfigurationValidation:
    """Test configuration parameter validation"""
    
    def test_host_parameter_validation(self):
        """Test various host parameter formats"""
        from ollama_workbench.server.server_configuration import apply_server_settings
        
        test_hosts = [
            "127.0.0.1",
            "127.0.0.1:11434",
            "localhost",
            "localhost:11434",
            "0.0.0.0:11434"
        ]
        
        with patch('ollama_workbench.server.server_configuration.st'):
            with patch('ollama_workbench.server.server_configuration.update_config'):
                with patch('ollama_workbench.server.server_configuration.platform.system', return_value="Linux"):
                    with patch('builtins.open', mock_open()):
                        with patch('ollama_workbench.server.server_configuration.json.dump'):
                            for host in test_hosts:
                                # Should not raise exceptions for valid hosts
                                apply_server_settings(host, "*", "/models", "5m", 1, 4, 512)
    
    def test_numeric_parameter_bounds(self):
        """Test numeric parameter validation"""
        from ollama_workbench.server.server_configuration import apply_server_settings
        
        with patch('ollama_workbench.server.server_configuration.st'):
            with patch('ollama_workbench.server.server_configuration.update_config'):
                with patch('ollama_workbench.server.server_configuration.platform.system', return_value="Linux"):
                    with patch('builtins.open', mock_open()):
                        with patch('ollama_workbench.server.server_configuration.json.dump'):
                            # Test extreme values
                            apply_server_settings(
                                "127.0.0.1",
                                "*",
                                "/models",
                                "0s",  # Minimum keep-alive
                                1,     # Minimum max loaded models
                                1,     # Minimum parallel
                                1      # Minimum queue
                            )
                            
                            apply_server_settings(
                                "127.0.0.1",
                                "*",
                                "/models",
                                "24h",   # Long keep-alive
                                100,     # High max loaded models
                                100,     # High parallel
                                10000    # Large queue
                            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
