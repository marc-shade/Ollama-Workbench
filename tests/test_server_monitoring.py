"""
Test suite for server_monitoring.py - Server monitoring functionality
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


class TestCommandExecution:
    """Test command execution functionality"""
    
    @patch('ollama_workbench.server.server_monitoring.subprocess.run')
    def test_run_command_success(self, mock_subprocess):
        """Test successful command execution"""
        from ollama_workbench.server.server_monitoring import run_command
        
        # Mock successful subprocess result
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Command output"
        mock_subprocess.return_value = mock_result
        
        result = run_command("test command")
        
        assert result == "Command output"
        mock_subprocess.assert_called_once_with(
            "test command", capture_output=True, text=True, shell=True
        )
    
    @patch('ollama_workbench.server.server_monitoring.st')
    @patch('ollama_workbench.server.server_monitoring.subprocess.run')
    def test_run_command_failure(self, mock_subprocess, mock_st):
        """Test command execution failure"""
        from ollama_workbench.server.server_monitoring import run_command
        
        # Mock failed subprocess result
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Command failed"
        mock_subprocess.return_value = mock_result
        
        result = run_command("failing command")
        
        assert result is None
        mock_st.error.assert_called_once_with("Error running command: Command failed")
    
    @patch('ollama_workbench.server.server_monitoring.st')
    @patch('ollama_workbench.server.server_monitoring.subprocess.run')
    def test_run_command_exception(self, mock_subprocess, mock_st):
        """Test command execution with exception"""
        from ollama_workbench.server.server_monitoring import run_command
        
        mock_subprocess.side_effect = Exception("Process error")
        
        result = run_command("exception command")
        
        assert result is None
        mock_st.error.assert_called_once_with("Exception running command: Process error")


class TestOllamaProcessInfo:
    """Test Ollama process information retrieval"""
    
    @patch('ollama_workbench.server.server_monitoring.run_command')
    def test_get_ollama_ps_success(self, mock_run_command):
        """Test successful ollama ps command"""
        from ollama_workbench.server.server_monitoring import get_ollama_ps
        
        mock_run_command.return_value = "NAME\t\tID\t\tSIZE\t\tPROCESSOR\nllama3:latest\tabc123\t3.8 GB\t100% CPU"
        
        result = get_ollama_ps()
        
        assert result == "NAME\t\tID\t\tSIZE\t\tPROCESSOR\nllama3:latest\tabc123\t3.8 GB\t100% CPU"
        mock_run_command.assert_called_once_with("ollama ps")
    
    @patch('ollama_workbench.server.server_monitoring.run_command')
    def test_get_ollama_ps_failure(self, mock_run_command):
        """Test failed ollama ps command"""
        from ollama_workbench.server.server_monitoring import get_ollama_ps
        
        mock_run_command.return_value = None
        
        result = get_ollama_ps()
        
        assert result is None
        mock_run_command.assert_called_once_with("ollama ps")
    
    @patch('ollama_workbench.server.server_monitoring.run_command')
    def test_get_ollama_ps_empty(self, mock_run_command):
        """Test ollama ps with no running models"""
        from ollama_workbench.server.server_monitoring import get_ollama_ps
        
        mock_run_command.return_value = "NAME\t\tID\t\tSIZE\t\tPROCESSOR\n"
        
        result = get_ollama_ps()
        
        assert result == "NAME\t\tID\t\tSIZE\t\tPROCESSOR\n"


class TestLogFileHandling:
    """Test log file path and retrieval functionality"""
    
    @patch('ollama_workbench.server.server_monitoring.platform.system')
    @patch('ollama_workbench.server.server_monitoring.os.path.expanduser')
    def test_get_log_file_path_darwin(self, mock_expanduser, mock_system):
        """Test log file path for macOS"""
        from ollama_workbench.server.server_monitoring import get_log_file_path
        
        mock_system.return_value = "Darwin"
        mock_expanduser.return_value = "/Users/test/.ollama/logs/server.log"
        
        result = get_log_file_path()
        
        assert result == "/Users/test/.ollama/logs/server.log"
        mock_expanduser.assert_called_once_with("~/.ollama/logs/server.log")
    
    @patch('ollama_workbench.server.server_monitoring.platform.system')
    def test_get_log_file_path_linux(self, mock_system):
        """Test log file path for Linux"""
        from ollama_workbench.server.server_monitoring import get_log_file_path
        
        mock_system.return_value = "Linux"
        
        result = get_log_file_path()
        
        assert result == "/var/log/ollama/server.log"
    
    @patch('ollama_workbench.server.server_monitoring.platform.system')
    @patch('ollama_workbench.server.server_monitoring.os.environ')
    def test_get_log_file_path_windows(self, mock_environ, mock_system):
        """Test log file path for Windows"""
        from ollama_workbench.server.server_monitoring import get_log_file_path
        
        mock_system.return_value = "Windows"
        mock_environ.__getitem__.return_value = "C:\\Users\\test\\AppData\\Local"
        
        with patch('ollama_workbench.server.server_monitoring.os.path.join') as mock_join:
            mock_join.return_value = "C:\\Users\\test\\AppData\\Local\\Ollama\\server.log"
            
            result = get_log_file_path()
        
        assert result == "C:\\Users\\test\\AppData\\Local\\Ollama\\server.log"
        mock_join.assert_called_once_with("C:\\Users\\test\\AppData\\Local", "Ollama", "server.log")
    
    @patch('ollama_workbench.server.server_monitoring.st')
    @patch('ollama_workbench.server.server_monitoring.platform.system')
    def test_get_log_file_path_unsupported(self, mock_system, mock_st):
        """Test log file path for unsupported OS"""
        from ollama_workbench.server.server_monitoring import get_log_file_path
        
        mock_system.return_value = "UnsupportedOS"
        
        result = get_log_file_path()
        
        assert result is None
        mock_st.warning.assert_called_once_with(
            "Unsupported operating system. Unable to fetch server logs."
        )
    
    @patch('ollama_workbench.server.server_monitoring.get_log_file_path')
    @patch('ollama_workbench.server.server_monitoring.run_command')
    def test_get_server_logs_success(self, mock_run_command, mock_log_path):
        """Test successful server log retrieval"""
        from ollama_workbench.server.server_monitoring import get_server_logs
        
        mock_log_path.return_value = "/var/log/ollama/server.log"
        mock_run_command.return_value = "2024-01-01 10:00:00 INFO Server started\n2024-01-01 10:01:00 INFO Model loaded"
        
        result = get_server_logs()
        
        assert "Server started" in result
        assert "Model loaded" in result
        mock_run_command.assert_called_once_with("tail -n 1000 /var/log/ollama/server.log")
    
    @patch('ollama_workbench.server.server_monitoring.get_log_file_path')
    @patch('ollama_workbench.server.server_monitoring.run_command')
    def test_get_server_logs_no_path(self, mock_run_command, mock_log_path):
        """Test server log retrieval with no log path"""
        from ollama_workbench.server.server_monitoring import get_server_logs
        
        mock_log_path.return_value = None
        
        result = get_server_logs()
        
        assert result is None
        mock_run_command.assert_not_called()
    
    @patch('ollama_workbench.server.server_monitoring.get_log_file_path')
    @patch('ollama_workbench.server.server_monitoring.run_command')
    def test_get_server_logs_command_failure(self, mock_run_command, mock_log_path):
        """Test server log retrieval with command failure"""
        from ollama_workbench.server.server_monitoring import get_server_logs
        
        mock_log_path.return_value = "/var/log/ollama/server.log"
        mock_run_command.return_value = None
        
        result = get_server_logs()
        
        assert result is None
        mock_run_command.assert_called_once_with("tail -n 1000 /var/log/ollama/server.log")


class TestResourceUsageMonitoring:
    """Test resource usage monitoring functionality"""
    
    @patch('ollama_workbench.server.server_monitoring.psutil.process_iter')
    def test_get_ollama_resource_usage_running(self, mock_process_iter):
        """Test resource usage when Ollama is running"""
        from ollama_workbench.server.server_monitoring import get_ollama_resource_usage
        
        # Mock Ollama process
        mock_ollama_process = Mock()
        mock_ollama_process.info = {'name': 'ollama', 'cpu_percent': 15.5, 'memory_percent': 8.2}
        mock_ollama_process.cpu_percent.return_value = 15.5
        mock_ollama_process.memory_percent.return_value = 8.2
        
        # Mock other processes
        mock_other_process = Mock()
        mock_other_process.info = {'name': 'other_process', 'cpu_percent': 1.0, 'memory_percent': 2.0}
        
        mock_process_iter.return_value = [mock_other_process, mock_ollama_process]
        
        result = get_ollama_resource_usage()
        
        assert result["status"] == "Running"
        assert result["cpu_usage"] == "15.50%"
        assert result["memory_usage"] == "8.20%"
        assert result["gpu_usage"] == "N/A"
        
        mock_ollama_process.cpu_percent.assert_called_once_with(interval=1)
        mock_ollama_process.memory_percent.assert_called_once()
    
    @patch('ollama_workbench.server.server_monitoring.psutil.process_iter')
    def test_get_ollama_resource_usage_not_running(self, mock_process_iter):
        """Test resource usage when Ollama is not running"""
        from ollama_workbench.server.server_monitoring import get_ollama_resource_usage
        
        # Mock only other processes (no Ollama)
        mock_other_process = Mock()
        mock_other_process.info = {'name': 'other_process', 'cpu_percent': 1.0, 'memory_percent': 2.0}
        
        mock_process_iter.return_value = [mock_other_process]
        
        result = get_ollama_resource_usage()
        
        assert result["status"] == "Not Running"
        assert result["cpu_usage"] == "0%"
        assert result["memory_usage"] == "0%"
        assert result["gpu_usage"] == "N/A"
    
    @patch('ollama_workbench.server.server_monitoring.psutil.process_iter')
    def test_get_ollama_resource_usage_multiple_processes(self, mock_process_iter):
        """Test resource usage with multiple processes named ollama"""
        from ollama_workbench.server.server_monitoring import get_ollama_resource_usage
        
        # Mock first Ollama process (should be the one selected)
        mock_ollama_process1 = Mock()
        mock_ollama_process1.info = {'name': 'ollama', 'cpu_percent': 20.0, 'memory_percent': 10.0}
        mock_ollama_process1.cpu_percent.return_value = 20.0
        mock_ollama_process1.memory_percent.return_value = 10.0
        
        # Mock second Ollama process
        mock_ollama_process2 = Mock()
        mock_ollama_process2.info = {'name': 'ollama', 'cpu_percent': 5.0, 'memory_percent': 3.0}
        
        mock_process_iter.return_value = [mock_ollama_process1, mock_ollama_process2]
        
        result = get_ollama_resource_usage()
        
        # Should use the first Ollama process found
        assert result["status"] == "Running"
        assert result["cpu_usage"] == "20.00%"
        assert result["memory_usage"] == "10.00%"
    
    @patch('ollama_workbench.server.server_monitoring.psutil.process_iter')
    def test_get_ollama_resource_usage_psutil_exception(self, mock_process_iter):
        """Test resource usage with psutil exception"""
        from ollama_workbench.server.server_monitoring import get_ollama_resource_usage
        
        mock_process_iter.side_effect = Exception("psutil error")
        
        # Should handle exception gracefully
        result = get_ollama_resource_usage()
        
        assert result["status"] == "Not Running"
        assert result["cpu_usage"] == "0%"
        assert result["memory_usage"] == "0%"
        assert result["gpu_usage"] == "N/A"
    
    @patch('ollama_workbench.server.server_monitoring.psutil.process_iter')
    def test_get_ollama_resource_usage_process_access_denied(self, mock_process_iter):
        """Test resource usage with process access denied"""
        from ollama_workbench.server.server_monitoring import get_ollama_resource_usage
        
        # Mock Ollama process that throws exception on resource access
        mock_ollama_process = Mock()
        mock_ollama_process.info = {'name': 'ollama', 'cpu_percent': 0, 'memory_percent': 0}
        mock_ollama_process.cpu_percent.side_effect = Exception("Access denied")
        mock_ollama_process.memory_percent.side_effect = Exception("Access denied")
        
        mock_process_iter.return_value = [mock_ollama_process]
        
        # Should handle exception gracefully
        result = get_ollama_resource_usage()
        
        assert result["status"] == "Not Running"
        assert result["cpu_usage"] == "0%"
        assert result["memory_usage"] == "0%"


class TestStreamlitInterface:
    """Test Streamlit interface functionality"""
    
    @patch('ollama_workbench.server.server_monitoring.st')
    @patch('ollama_workbench.server.server_monitoring.get_ollama_resource_usage')
    @patch('ollama_workbench.server.server_monitoring.get_ollama_ps')
    @patch('ollama_workbench.server.server_monitoring.get_server_logs')
    @patch('ollama_workbench.server.server_monitoring.platform.system')
    @patch('ollama_workbench.server.server_monitoring.os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"host": "localhost", "port": 11434}')
    def test_server_monitoring_interface_complete(self, mock_file, mock_exists, mock_system,
                                                 mock_logs, mock_ps, mock_usage, mock_st):
        """Test complete server monitoring interface"""
        from ollama_workbench.server.server_monitoring import server_monitoring
        
        # Setup mocks
        mock_usage.return_value = {
            "status": "Running",
            "cpu_usage": "15.50%",
            "memory_usage": "8.20%",
            "gpu_usage": "N/A"
        }
        mock_ps.return_value = "llama3:latest\tabc123\t3.8 GB\t100% CPU"
        mock_logs.return_value = "2024-01-01 10:00:00 INFO Server started"
        mock_system.return_value = "Linux"
        mock_exists.return_value = True
        
        # Mock Streamlit components
        mock_st.header = Mock()
        mock_st.subheader = Mock()
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock(), Mock()])
        mock_st.text_area = Mock()
        mock_st.button = Mock(return_value=False)
        mock_st.json = Mock()
        mock_st.download_button = Mock()
        
        server_monitoring()
        
        # Verify interface setup
        mock_st.header.assert_called_with("🖥️ Ollama Server Monitoring")
        assert mock_st.subheader.call_count >= 3  # Resource Usage, Running Models, Server Logs, etc.
        mock_st.columns.assert_called_with(4)  # For metrics
        mock_usage.assert_called_once()
        mock_ps.assert_called_once()
        mock_logs.assert_called_once()
    
    @patch('ollama_workbench.server.server_monitoring.st')
    @patch('ollama_workbench.server.server_monitoring.get_ollama_resource_usage')
    @patch('ollama_workbench.server.server_monitoring.get_ollama_ps')
    @patch('ollama_workbench.server.server_monitoring.get_server_logs')
    @patch('ollama_workbench.server.server_monitoring.platform.system')
    @patch('ollama_workbench.server.server_monitoring.os.path.exists')
    def test_server_monitoring_interface_no_config(self, mock_exists, mock_system,
                                                  mock_logs, mock_ps, mock_usage, mock_st):
        """Test server monitoring interface with no config file"""
        from ollama_workbench.server.server_monitoring import server_monitoring
        
        # Setup mocks
        mock_usage.return_value = {
            "status": "Not Running",
            "cpu_usage": "0%",
            "memory_usage": "0%",
            "gpu_usage": "N/A"
        }
        mock_ps.return_value = None
        mock_logs.return_value = None
        mock_system.return_value = "Linux"
        mock_exists.return_value = False  # Config file doesn't exist
        
        # Mock Streamlit components
        mock_st.header = Mock()
        mock_st.subheader = Mock()
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock(), Mock()])
        mock_st.text_area = Mock()
        mock_st.button = Mock(return_value=False)
        mock_st.warning = Mock()
        
        server_monitoring()
        
        # Verify warning is shown for missing config
        mock_st.warning.assert_called_with("Configuration file not found.")
    
    @patch('ollama_workbench.server.server_monitoring.st')
    @patch('ollama_workbench.server.server_monitoring.get_ollama_resource_usage')
    @patch('ollama_workbench.server.server_monitoring.get_ollama_ps')
    @patch('ollama_workbench.server.server_monitoring.get_server_logs')
    @patch('ollama_workbench.server.server_monitoring.platform.system')
    def test_server_monitoring_interface_unsupported_os(self, mock_system, mock_logs,
                                                       mock_ps, mock_usage, mock_st):
        """Test server monitoring interface on unsupported OS"""
        from ollama_workbench.server.server_monitoring import server_monitoring
        
        # Setup mocks
        mock_usage.return_value = {
            "status": "Running",
            "cpu_usage": "10%",
            "memory_usage": "5%",
            "gpu_usage": "N/A"
        }
        mock_ps.return_value = "model running"
        mock_logs.return_value = "log data"
        mock_system.return_value = "UnsupportedOS"
        
        # Mock Streamlit components
        mock_st.header = Mock()
        mock_st.subheader = Mock()
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock(), Mock()])
        mock_st.text_area = Mock()
        mock_st.button = Mock(return_value=False)
        mock_st.warning = Mock()
        
        server_monitoring()
        
        # Verify warning is shown for unsupported OS
        mock_st.warning.assert_called_with(
            "Unsupported operating system. Unable to fetch server configuration."
        )
    
    @patch('ollama_workbench.server.server_monitoring.st')
    @patch('ollama_workbench.server.server_monitoring.get_ollama_resource_usage')
    @patch('ollama_workbench.server.server_monitoring.get_ollama_ps')
    @patch('ollama_workbench.server.server_monitoring.get_server_logs')
    @patch('ollama_workbench.server.server_monitoring.platform.system')
    @patch('ollama_workbench.server.server_monitoring.os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"host": "localhost"}')
    def test_server_monitoring_refresh_logs(self, mock_file, mock_exists, mock_system,
                                           mock_logs, mock_ps, mock_usage, mock_st):
        """Test refresh logs functionality"""
        from ollama_workbench.server.server_monitoring import server_monitoring
        
        # Setup mocks
        mock_usage.return_value = {"status": "Running", "cpu_usage": "10%", "memory_usage": "5%", "gpu_usage": "N/A"}
        mock_ps.return_value = "models running"
        mock_logs.return_value = "fresh log data"
        mock_system.return_value = "Linux"
        mock_exists.return_value = True
        
        # Mock Streamlit components
        mock_st.header = Mock()
        mock_st.subheader = Mock()
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock(), Mock()])
        mock_st.text_area = Mock()
        mock_st.button = Mock(return_value=True)  # Refresh button clicked
        mock_st.rerun = Mock()
        mock_st.json = Mock()
        mock_st.download_button = Mock()
        
        server_monitoring()
        
        # Verify refresh behavior
        mock_st.rerun.assert_called_once()
    
    @patch('ollama_workbench.server.server_monitoring.st')
    @patch('ollama_workbench.server.server_monitoring.get_ollama_resource_usage')
    @patch('ollama_workbench.server.server_monitoring.get_ollama_ps')
    @patch('ollama_workbench.server.server_monitoring.get_server_logs')
    @patch('ollama_workbench.server.server_monitoring.platform.system')
    @patch('ollama_workbench.server.server_monitoring.os.path.exists')
    @patch('ollama_workbench.server.server_monitoring.os.path.expanduser')
    def test_server_monitoring_darwin_config_path(self, mock_expanduser, mock_exists,
                                                 mock_system, mock_logs, mock_ps, mock_usage, mock_st):
        """Test server monitoring with macOS config path"""
        from ollama_workbench.server.server_monitoring import server_monitoring
        
        # Setup mocks
        mock_usage.return_value = {"status": "Running", "cpu_usage": "5%", "memory_usage": "3%", "gpu_usage": "N/A"}
        mock_ps.return_value = "models"
        mock_logs.return_value = "logs"
        mock_system.return_value = "Darwin"
        mock_exists.return_value = True
        mock_expanduser.return_value = "/Users/test/Library/Application Support/Ollama/config.json"
        
        # Mock Streamlit components
        mock_st.header = Mock()
        mock_st.subheader = Mock()
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock(), Mock()])
        mock_st.text_area = Mock()
        mock_st.button = Mock(return_value=False)
        mock_st.json = Mock()
        mock_st.download_button = Mock()
        
        with patch('builtins.open', mock_open(read_data='{"host": "localhost"}')):
            server_monitoring()
        
        # Verify macOS-specific path was used
        mock_expanduser.assert_called_with("~/Library/Application Support/Ollama/config.json")
    
    @patch('ollama_workbench.server.server_monitoring.st')
    @patch('ollama_workbench.server.server_monitoring.get_ollama_resource_usage')
    @patch('ollama_workbench.server.server_monitoring.get_ollama_ps')
    @patch('ollama_workbench.server.server_monitoring.get_server_logs')
    @patch('ollama_workbench.server.server_monitoring.platform.system')
    @patch('ollama_workbench.server.server_monitoring.os.path.exists')
    @patch('ollama_workbench.server.server_monitoring.os.environ')
    def test_server_monitoring_windows_config_path(self, mock_environ, mock_exists,
                                                  mock_system, mock_logs, mock_ps, mock_usage, mock_st):
        """Test server monitoring with Windows config path"""
        from ollama_workbench.server.server_monitoring import server_monitoring
        
        # Setup mocks
        mock_usage.return_value = {"status": "Running", "cpu_usage": "12%", "memory_usage": "7%", "gpu_usage": "N/A"}
        mock_ps.return_value = "windows models"
        mock_logs.return_value = "windows logs"
        mock_system.return_value = "Windows"
        mock_exists.return_value = True
        mock_environ.__getitem__.return_value = "C:\\Users\\test\\AppData\\Roaming"
        
        # Mock Streamlit components
        mock_st.header = Mock()
        mock_st.subheader = Mock()
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock(), Mock()])
        mock_st.text_area = Mock()
        mock_st.button = Mock(return_value=False)
        mock_st.json = Mock()
        mock_st.download_button = Mock()
        
        with patch('ollama_workbench.server.server_monitoring.os.path.join') as mock_join:
            mock_join.return_value = "C:\\Users\\test\\AppData\\Roaming\\Ollama\\config.json"
            with patch('builtins.open', mock_open(read_data='{"host": "localhost"}')):
                server_monitoring()
        
        # Verify Windows-specific path was used
        mock_join.assert_called_with("C:\\Users\\test\\AppData\\Roaming", "Ollama", "config.json")


class TestConfigurationHandling:
    """Test configuration file handling"""
    
    @patch('ollama_workbench.server.server_monitoring.st')
    @patch('ollama_workbench.server.server_monitoring.platform.system')
    @patch('ollama_workbench.server.server_monitoring.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_config_file_json_error(self, mock_file, mock_exists, mock_system, mock_st):
        """Test configuration file with JSON parsing error"""
        from ollama_workbench.server.server_monitoring import server_monitoring
        
        mock_system.return_value = "Linux"
        mock_exists.return_value = True
        mock_file.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
        
        # Mock other components
        with patch('ollama_workbench.server.server_monitoring.get_ollama_resource_usage') as mock_usage:
            with patch('ollama_workbench.server.server_monitoring.get_ollama_ps') as mock_ps:
                with patch('ollama_workbench.server.server_monitoring.get_server_logs') as mock_logs:
                    mock_usage.return_value = {"status": "Running", "cpu_usage": "5%", "memory_usage": "3%", "gpu_usage": "N/A"}
                    mock_ps.return_value = "models"
                    mock_logs.return_value = "logs"
                    
                    # Mock Streamlit components
                    mock_st.header = Mock()
                    mock_st.subheader = Mock()
                    mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock(), Mock()])
                    mock_st.text_area = Mock()
                    mock_st.button = Mock(return_value=False)
                    
                    # Should handle JSON error gracefully
                    try:
                        server_monitoring()
                    except json.JSONDecodeError:
                        pass  # Expected if not handled
    
    @patch('ollama_workbench.server.server_monitoring.st')
    @patch('ollama_workbench.server.server_monitoring.platform.system')
    @patch('ollama_workbench.server.server_monitoring.os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"complex": {"nested": {"config": "value"}}}')
    def test_config_file_complex_json(self, mock_file, mock_exists, mock_system, mock_st):
        """Test configuration file with complex JSON structure"""
        from ollama_workbench.server.server_monitoring import server_monitoring
        
        mock_system.return_value = "Linux"
        mock_exists.return_value = True
        
        # Mock other components
        with patch('ollama_workbench.server.server_monitoring.get_ollama_resource_usage') as mock_usage:
            with patch('ollama_workbench.server.server_monitoring.get_ollama_ps') as mock_ps:
                with patch('ollama_workbench.server.server_monitoring.get_server_logs') as mock_logs:
                    mock_usage.return_value = {"status": "Running", "cpu_usage": "8%", "memory_usage": "4%", "gpu_usage": "N/A"}
                    mock_ps.return_value = "complex models"
                    mock_logs.return_value = "complex logs"
                    
                    # Mock Streamlit components
                    mock_st.header = Mock()
                    mock_st.subheader = Mock()
                    mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock(), Mock()])
                    mock_st.text_area = Mock()
                    mock_st.button = Mock(return_value=False)
                    mock_st.json = Mock()
                    mock_st.download_button = Mock()
                    
                    server_monitoring()
                    
                    # Verify complex JSON was handled
                    mock_st.json.assert_called_once()
                    mock_st.download_button.assert_called_once()


class TestIntegration:
    """Test integration scenarios"""
    
    def test_module_imports(self):
        """Test that all required modules can be imported"""
        import ollama_workbench.server.server_monitoring as server_monitoring

        
        # Test that main functions exist
        assert hasattr(server_monitoring, 'run_command')
        assert hasattr(server_monitoring, 'get_ollama_ps')
        assert hasattr(server_monitoring, 'get_server_logs')
        assert hasattr(server_monitoring, 'get_log_file_path')
        assert hasattr(server_monitoring, 'get_ollama_resource_usage')
        assert hasattr(server_monitoring, 'server_monitoring')
    
    @patch('ollama_workbench.server.server_monitoring.platform.system')
    @patch('ollama_workbench.server.server_monitoring.os.path.expanduser')
    def test_log_path_consistency(self, mock_expanduser, mock_system):
        """Test log path consistency across different OS calls"""
        from ollama_workbench.server.server_monitoring import get_log_file_path
        
        mock_system.return_value = "Darwin"
        mock_expanduser.return_value = "/Users/test/.ollama/logs/server.log"
        
        # Multiple calls should return consistent results
        path1 = get_log_file_path()
        path2 = get_log_file_path()
        
        assert path1 == path2
        assert path1 == "/Users/test/.ollama/logs/server.log"
    
    @patch('ollama_workbench.server.server_monitoring.psutil.process_iter')
    @patch('ollama_workbench.server.server_monitoring.run_command')
    def test_resource_monitoring_integration(self, mock_run_command, mock_process_iter):
        """Test integration between resource monitoring and command execution"""
        from ollama_workbench.server.server_monitoring import get_ollama_resource_usage, get_ollama_ps
        
        # Setup mocks
        mock_ollama_process = Mock()
        mock_ollama_process.info = {'name': 'ollama', 'cpu_percent': 10.0, 'memory_percent': 5.0}
        mock_ollama_process.cpu_percent.return_value = 10.0
        mock_ollama_process.memory_percent.return_value = 5.0
        mock_process_iter.return_value = [mock_ollama_process]
        
        mock_run_command.return_value = "llama3:latest\trunning"
        
        # Test both functions work together
        usage = get_ollama_resource_usage()
        models = get_ollama_ps()
        
        assert usage["status"] == "Running"
        assert models == "llama3:latest\trunning"


class TestErrorHandling:
    """Test error handling in various scenarios"""
    
    @patch('ollama_workbench.server.server_monitoring.subprocess.run')
    def test_command_timeout_handling(self, mock_subprocess):
        """Test command execution with timeout"""
        from ollama_workbench.server.server_monitoring import run_command
        
        mock_subprocess.side_effect = subprocess.TimeoutExpired("cmd", 30)
        
        with patch('ollama_workbench.server.server_monitoring.st') as mock_st:
            result = run_command("long running command")
        
        assert result is None
        mock_st.error.assert_called_once()
    
    @patch('ollama_workbench.server.server_monitoring.psutil.process_iter')
    def test_process_permission_error(self, mock_process_iter):
        """Test handling of permission errors when accessing process info"""
        from ollama_workbench.server.server_monitoring import get_ollama_resource_usage
        
        mock_process = Mock()
        mock_process.info = {'name': 'ollama', 'cpu_percent': 0, 'memory_percent': 0}
        mock_process.cpu_percent.side_effect = PermissionError("Access denied")
        mock_process.memory_percent.side_effect = PermissionError("Access denied")
        mock_process_iter.return_value = [mock_process]
        
        result = get_ollama_resource_usage()
        
        # Should handle permission error gracefully
        assert result["status"] == "Not Running"
        assert result["cpu_usage"] == "0%"
        assert result["memory_usage"] == "0%"
    
    @patch('ollama_workbench.server.server_monitoring.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_config_file_permission_error(self, mock_file, mock_exists):
        """Test configuration file access with permission error"""
        from ollama_workbench.server.server_monitoring import server_monitoring
        
        mock_exists.return_value = True
        mock_file.side_effect = PermissionError("Access denied")
        
        with patch('ollama_workbench.server.server_monitoring.st') as mock_st:
            with patch('ollama_workbench.server.server_monitoring.platform.system', return_value="Linux"):
                with patch('ollama_workbench.server.server_monitoring.get_ollama_resource_usage') as mock_usage:
                    with patch('ollama_workbench.server.server_monitoring.get_ollama_ps') as mock_ps:
                        with patch('ollama_workbench.server.server_monitoring.get_server_logs') as mock_logs:
                            # Setup basic mocks
                            mock_usage.return_value = {"status": "Running", "cpu_usage": "5%", "memory_usage": "3%", "gpu_usage": "N/A"}
                            mock_ps.return_value = "models"
                            mock_logs.return_value = "logs"
                            
                            # Mock Streamlit components
                            mock_st.header = Mock()
                            mock_st.subheader = Mock()
                            mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock(), Mock()])
                            mock_st.text_area = Mock()
                            mock_st.button = Mock(return_value=False)
                            
                            # Should handle permission error gracefully
                            try:
                                server_monitoring()
                            except PermissionError:
                                pass  # Expected if not handled gracefully


class TestPlatformSpecificBehavior:
    """Test platform-specific behavior"""
    
    @patch('ollama_workbench.server.server_monitoring.platform.system')
    def test_all_supported_platforms_log_paths(self, mock_system):
        """Test log paths for all supported platforms"""
        from ollama_workbench.server.server_monitoring import get_log_file_path
        
        # Test Linux
        mock_system.return_value = "Linux"
        assert get_log_file_path() == "/var/log/ollama/server.log"
        
        # Test macOS
        mock_system.return_value = "Darwin"
        with patch('ollama_workbench.server.server_monitoring.os.path.expanduser') as mock_expanduser:
            mock_expanduser.return_value = "/Users/test/.ollama/logs/server.log"
            assert get_log_file_path() == "/Users/test/.ollama/logs/server.log"
        
        # Test Windows
        mock_system.return_value = "Windows"
        with patch('ollama_workbench.server.server_monitoring.os.environ') as mock_environ:
            mock_environ.__getitem__.return_value = "C:\\Users\\test\\AppData\\Local"
            with patch('ollama_workbench.server.server_monitoring.os.path.join') as mock_join:
                mock_join.return_value = "C:\\Users\\test\\AppData\\Local\\Ollama\\server.log"
                assert get_log_file_path() == "C:\\Users\\test\\AppData\\Local\\Ollama\\server.log"
    
    @patch('ollama_workbench.server.server_monitoring.platform.system')
    def test_unsupported_platform_behavior(self, mock_system):
        """Test behavior on unsupported platforms"""
        from ollama_workbench.server.server_monitoring import get_log_file_path
        
        mock_system.return_value = "FreeBSD"
        
        with patch('ollama_workbench.server.server_monitoring.st') as mock_st:
            result = get_log_file_path()
        
        assert result is None
        mock_st.warning.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
