"""
Comprehensive tests for error_handling.py module.

Tests all error handling functionality including:
- Custom exception classes
- Error categories and levels  
- Error handling decorators
- SafeHTTPClient with error handling
- Utility functions and helpers
"""

import pytest
import json
import logging
import requests
from unittest.mock import Mock, patch, MagicMock, mock_open
from unittest import TestCase
import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from error_handling import (
    ErrorLevel, ErrorCategory, WorkbenchError, APIError, NetworkError,
    AuthenticationError, ModelError, ParameterError, handle_requests_error,
    handle_api_error, display_error, capture_exceptions, safe_json_loads,
    get_error_message, error_response, handle_ollama_api_error,
    handle_openai_api_error, SafeHTTPClient, create_ollama_client,
    error_middleware
)


class TestErrorLevel(TestCase):
    """Test ErrorLevel enum"""
    
    def test_error_level_values(self):
        """Test that all error levels have correct values"""
        self.assertEqual(ErrorLevel.INFO.value, "info")
        self.assertEqual(ErrorLevel.WARNING.value, "warning")
        self.assertEqual(ErrorLevel.ERROR.value, "error")
        self.assertEqual(ErrorLevel.CRITICAL.value, "critical")
    
    def test_error_level_members(self):
        """Test that all expected members exist"""
        levels = [level.value for level in ErrorLevel]
        expected = ["info", "warning", "error", "critical"]
        self.assertEqual(sorted(levels), sorted(expected))


class TestErrorCategory(TestCase):
    """Test ErrorCategory enum"""
    
    def test_error_category_values(self):
        """Test that all error categories have correct values"""
        self.assertEqual(ErrorCategory.API.value, "api")
        self.assertEqual(ErrorCategory.NETWORK.value, "network")
        self.assertEqual(ErrorCategory.AUTHENTICATION.value, "authentication")
        self.assertEqual(ErrorCategory.MODEL.value, "model")
        self.assertEqual(ErrorCategory.PARAMETER.value, "parameter")
        self.assertEqual(ErrorCategory.SYSTEM.value, "system")
        self.assertEqual(ErrorCategory.USER.value, "user")
        self.assertEqual(ErrorCategory.DATA.value, "data")
    
    def test_error_category_members(self):
        """Test that all expected members exist"""
        categories = [cat.value for cat in ErrorCategory]
        expected = ["api", "network", "authentication", "model", "parameter", "system", "user", "data"]
        self.assertEqual(sorted(categories), sorted(expected))


class TestWorkbenchError(TestCase):
    """Test WorkbenchError base class"""
    
    def test_basic_initialization(self):
        """Test basic error initialization"""
        error = WorkbenchError("Test message")
        
        self.assertEqual(error.message, "Test message")
        self.assertIsNone(error.error_code)
        self.assertEqual(error.category, ErrorCategory.SYSTEM)
        self.assertEqual(error.level, ErrorLevel.ERROR)
        self.assertEqual(error.details, {})
        self.assertIsNone(error.original_error)
    
    def test_full_initialization(self):
        """Test error initialization with all parameters"""
        original_error = ValueError("Original")
        details = {"key": "value"}
        
        error = WorkbenchError(
            message="Test message",
            error_code="TEST_001",
            category=ErrorCategory.API,
            level=ErrorLevel.WARNING,
            details=details,
            original_error=original_error
        )
        
        self.assertEqual(error.message, "Test message")
        self.assertEqual(error.error_code, "TEST_001")
        self.assertEqual(error.category, ErrorCategory.API)
        self.assertEqual(error.level, ErrorLevel.WARNING)
        self.assertEqual(error.details, details)
        self.assertEqual(error.original_error, original_error)
    
    def test_to_dict_basic(self):
        """Test converting basic error to dict"""
        error = WorkbenchError("Test message")
        
        result = error.to_dict()
        expected = {
            "message": "Test message",
            "category": "system",
            "level": "error"
        }
        
        self.assertEqual(result, expected)
    
    def test_to_dict_full(self):
        """Test converting full error to dict"""
        original_error = ValueError("Original")
        details = {"key": "value"}
        
        error = WorkbenchError(
            message="Test message",
            error_code="TEST_001",
            category=ErrorCategory.API,
            level=ErrorLevel.WARNING,
            details=details,
            original_error=original_error
        )
        
        result = error.to_dict()
        expected = {
            "message": "Test message",
            "category": "api",
            "level": "warning",
            "error_code": "TEST_001",
            "details": details,
            "original_error": "Original"
        }
        
        self.assertEqual(result, expected)
    
    @patch('error_handling.logger')
    def test_log_info_level(self, mock_logger):
        """Test logging at INFO level"""
        error = WorkbenchError("Test message", level=ErrorLevel.INFO)
        error.log()
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        self.assertIn("Test message", call_args)
    
    @patch('error_handling.logger')
    def test_log_warning_level(self, mock_logger):
        """Test logging at WARNING level"""
        error = WorkbenchError("Test message", level=ErrorLevel.WARNING)
        error.log()
        
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        self.assertIn("Test message", call_args)
    
    @patch('error_handling.logger')
    def test_log_error_level(self, mock_logger):
        """Test logging at ERROR level"""
        error = WorkbenchError("Test message", level=ErrorLevel.ERROR)
        error.log()
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        self.assertIn("Test message", call_args)
    
    @patch('error_handling.logger')
    def test_log_critical_level(self, mock_logger):
        """Test logging at CRITICAL level"""
        error = WorkbenchError("Test message", level=ErrorLevel.CRITICAL)
        error.log()
        
        mock_logger.critical.assert_called_once()
        call_args = mock_logger.critical.call_args[0][0]
        self.assertIn("Test message", call_args)
    
    @patch('error_handling.logger')
    def test_log_with_details_and_original_error(self, mock_logger):
        """Test logging with details and original error"""
        original_error = ValueError("Original")
        details = {"key": "value"}
        
        error = WorkbenchError(
            message="Test message",
            error_code="TEST_001",
            details=details,
            original_error=original_error
        )
        error.log()
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        self.assertIn("TEST_001", call_args)
        self.assertIn("Test message", call_args)
        self.assertIn("Details:", call_args)
        self.assertIn("Original error:", call_args)


class TestAPIError(TestCase):
    """Test APIError specialized exception"""
    
    def test_basic_api_error(self):
        """Test basic API error initialization"""
        error = APIError("API failed", "TestAPI")
        
        self.assertEqual(error.message, "API failed")
        self.assertEqual(error.category, ErrorCategory.API)
        self.assertEqual(error.error_code, "API_ERROR")
        self.assertEqual(error.details["api_name"], "TestAPI")
    
    def test_api_error_with_status_code(self):
        """Test API error with status code"""
        error = APIError("API failed", "TestAPI", status_code=404)
        
        self.assertEqual(error.error_code, "API_404")
        self.assertEqual(error.details["status_code"], 404)
    
    def test_api_error_full_details(self):
        """Test API error with all details"""
        error = APIError(
            message="API failed",
            api_name="TestAPI",
            status_code=500,
            response_body="Internal Server Error",
            endpoint="/api/test"
        )
        
        expected_details = {
            "api_name": "TestAPI",
            "status_code": 500,
            "response_body": "Internal Server Error",
            "endpoint": "/api/test"
        }
        
        self.assertEqual(error.details, expected_details)
        self.assertEqual(error.error_code, "API_500")


class TestNetworkError(TestCase):
    """Test NetworkError specialized exception"""
    
    def test_basic_network_error(self):
        """Test basic network error initialization"""
        error = NetworkError("Connection failed")
        
        self.assertEqual(error.message, "Connection failed")
        self.assertEqual(error.category, ErrorCategory.NETWORK)
        self.assertEqual(error.error_code, "NETWORK_ERROR")
    
    def test_network_error_with_url(self):
        """Test network error with URL"""
        error = NetworkError("Connection failed", url="http://example.com")
        
        self.assertEqual(error.details["url"], "http://example.com")
    
    def test_network_error_with_timeout(self):
        """Test network error with timeout flag"""
        error = NetworkError("Connection failed", timeout=True)
        
        self.assertEqual(error.details["timeout"], True)


class TestAuthenticationError(TestCase):
    """Test AuthenticationError specialized exception"""
    
    def test_basic_auth_error(self):
        """Test basic authentication error"""
        error = AuthenticationError("Auth failed")
        
        self.assertEqual(error.message, "Auth failed")
        self.assertEqual(error.category, ErrorCategory.AUTHENTICATION)
        self.assertEqual(error.error_code, "AUTH_ERROR")
    
    def test_auth_error_with_service(self):
        """Test authentication error with service"""
        error = AuthenticationError("Auth failed", service="OpenAI")
        
        self.assertEqual(error.details["service"], "OpenAI")


class TestModelError(TestCase):
    """Test ModelError specialized exception"""
    
    def test_basic_model_error(self):
        """Test basic model error"""
        error = ModelError("Model failed")
        
        self.assertEqual(error.message, "Model failed")
        self.assertEqual(error.category, ErrorCategory.MODEL)
        self.assertEqual(error.error_code, "MODEL_ERROR")
    
    def test_model_error_with_details(self):
        """Test model error with model and provider"""
        error = ModelError("Model failed", model="llama3", provider="ollama")
        
        expected_details = {
            "model": "llama3",
            "provider": "ollama"
        }
        
        self.assertEqual(error.details, expected_details)


class TestParameterError(TestCase):
    """Test ParameterError specialized exception"""
    
    def test_basic_parameter_error(self):
        """Test basic parameter error"""
        error = ParameterError("Invalid parameter")
        
        self.assertEqual(error.message, "Invalid parameter")
        self.assertEqual(error.category, ErrorCategory.PARAMETER)
        self.assertEqual(error.error_code, "PARAMETER_ERROR")
    
    def test_parameter_error_with_details(self):
        """Test parameter error with parameter details"""
        error = ParameterError("Invalid parameter", parameter="temperature", value=2.0)
        
        expected_details = {
            "parameter": "temperature",
            "value": "2.0"
        }
        
        self.assertEqual(error.details, expected_details)


class TestHandleRequestsError(TestCase):
    """Test handle_requests_error function"""
    
    def test_handle_http_error(self):
        """Test handling requests.HTTPError"""
        # Create a mock response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        
        # Create HTTPError
        http_error = requests.HTTPError()
        http_error.response = mock_response
        
        result = handle_requests_error(http_error, "TestAPI", "/test")
        
        self.assertIsInstance(result, APIError)
        self.assertEqual(result.details["api_name"], "TestAPI")
        self.assertEqual(result.details["status_code"], 404)
        self.assertEqual(result.details["response_body"], "Not Found")
        self.assertEqual(result.details["endpoint"], "/test")
    
    def test_handle_connection_error(self):
        """Test handling requests.ConnectionError"""
        conn_error = requests.ConnectionError("Connection failed")
        
        result = handle_requests_error(conn_error, "TestAPI", "/test")
        
        self.assertIsInstance(result, NetworkError)
        self.assertEqual(result.details["url"], "/test")
    
    def test_handle_timeout_error(self):
        """Test handling requests.Timeout"""
        timeout_error = requests.Timeout("Request timed out")
        
        result = handle_requests_error(timeout_error, "TestAPI", "/test")
        
        self.assertIsInstance(result, NetworkError)
        self.assertEqual(result.details["timeout"], True)
    
    def test_handle_other_requests_error(self):
        """Test handling other requests errors"""
        other_error = requests.RequestException("Unknown error")
        
        result = handle_requests_error(other_error, "TestAPI", "/test")
        
        self.assertIsInstance(result, APIError)
        self.assertEqual(result.details["api_name"], "TestAPI")


class TestHandleApiErrorDecorator(TestCase):
    """Test handle_api_error decorator"""
    
    @patch('error_handling.display_error')
    @patch('error_handling.logger')
    def test_decorator_catches_requests_exception(self, mock_logger, mock_display):
        """Test decorator catches requests exceptions"""
        
        @handle_api_error
        def test_func():
            raise requests.ConnectionError("Connection failed")
        
        result = test_func()
        
        self.assertIsNone(result)
        mock_display.assert_called_once()
        # Check that an error was logged
        self.assertTrue(mock_logger.method_calls)
    
    @patch('error_handling.display_error')
    @patch('error_handling.logger')
    def test_decorator_catches_general_exception(self, mock_logger, mock_display):
        """Test decorator catches general exceptions"""
        
        @handle_api_error
        def test_func():
            raise ValueError("Test error")
        
        result = test_func()
        
        self.assertIsNone(result)
        mock_display.assert_called_once()
    
    def test_decorator_returns_normal_result(self):
        """Test decorator returns normal result on success"""
        
        @handle_api_error
        def test_func():
            return "success"
        
        result = test_func()
        
        self.assertEqual(result, "success")


class TestCaptureExceptionsDecorator(TestCase):
    """Test capture_exceptions decorator"""
    
    @patch('error_handling.display_error')
    @patch('error_handling.logger')
    def test_decorator_catches_workbench_error(self, mock_logger, mock_display):
        """Test decorator catches WorkbenchError"""
        
        @capture_exceptions
        def test_func():
            raise WorkbenchError("Test error")
        
        result = test_func()
        
        self.assertIsNone(result)
        mock_display.assert_called_once()
    
    @patch('error_handling.display_error')
    @patch('error_handling.logger')
    def test_decorator_catches_general_exception(self, mock_logger, mock_display):
        """Test decorator catches general exceptions"""
        
        @capture_exceptions
        def test_func():
            raise ValueError("Test error")
        
        result = test_func()
        
        self.assertIsNone(result)
        mock_display.assert_called_once()
    
    def test_decorator_returns_normal_result(self):
        """Test decorator returns normal result on success"""
        
        @capture_exceptions
        def test_func():
            return "success"
        
        result = test_func()
        
        self.assertEqual(result, "success")


class TestDisplayError(TestCase):
    """Test display_error function"""
    
    @patch('streamlit.info')
    def test_display_info_error(self, mock_info):
        """Test displaying INFO level error"""
        error = WorkbenchError("Test message", level=ErrorLevel.INFO)
        
        display_error(error)
        
        mock_info.assert_called_once_with("Test message")
    
    @patch('streamlit.warning')
    def test_display_warning_error(self, mock_warning):
        """Test displaying WARNING level error"""
        error = WorkbenchError("Test message", level=ErrorLevel.WARNING)
        
        display_error(error)
        
        mock_warning.assert_called_once_with("Test message")
    
    @patch('streamlit.error')
    def test_display_error_level(self, mock_error):
        """Test displaying ERROR level error"""
        error = WorkbenchError("Test message", level=ErrorLevel.ERROR)
        
        display_error(error)
        
        mock_error.assert_called_once_with("Test message")
    
    @patch('streamlit.error')
    def test_display_critical_error(self, mock_error):
        """Test displaying CRITICAL level error"""
        error = WorkbenchError("Test message", level=ErrorLevel.CRITICAL)
        
        display_error(error)
        
        mock_error.assert_called_once_with("CRITICAL ERROR: Test message")
    
    @patch('streamlit.expander')
    @patch('streamlit.error')
    def test_display_error_with_details(self, mock_error, mock_expander):
        """Test displaying error with details"""
        mock_context = Mock()
        mock_expander.return_value.__enter__ = Mock(return_value=mock_context)
        mock_expander.return_value.__exit__ = Mock(return_value=None)
        
        error = WorkbenchError(
            "Test message",
            error_code="TEST_001",
            details={"key": "value"},
            original_error=ValueError("Original")
        )
        
        with patch('streamlit.write'), patch('streamlit.json'), patch('streamlit.code'):
            display_error(error)
        
        mock_expander.assert_called_once_with("Error Details")


class TestUtilityFunctions(TestCase):
    """Test utility functions"""
    
    def test_safe_json_loads_valid_json(self):
        """Test safe_json_loads with valid JSON"""
        json_str = '{"key": "value"}'
        result = safe_json_loads(json_str)
        
        self.assertEqual(result, {"key": "value"})
    
    def test_safe_json_loads_invalid_json(self):
        """Test safe_json_loads with invalid JSON"""
        json_str = '{"key": invalid}'
        result = safe_json_loads(json_str, default={"default": True})
        
        self.assertEqual(result, {"default": True})
    
    def test_safe_json_loads_invalid_json_no_default(self):
        """Test safe_json_loads with invalid JSON and no default"""
        json_str = '{"key": invalid}'
        result = safe_json_loads(json_str)
        
        self.assertIsNone(result)
    
    def test_get_error_message_with_message_attr(self):
        """Test get_error_message with exception that has message attribute"""
        class CustomError(Exception):
            def __init__(self, message):
                self.message = message
        
        error = CustomError("Custom message")
        result = get_error_message(error)
        
        self.assertEqual(result, "Custom message")
    
    def test_get_error_message_without_message_attr(self):
        """Test get_error_message with standard exception"""
        error = ValueError("Standard message")
        result = get_error_message(error)
        
        self.assertEqual(result, "Standard message")
    
    def test_error_response_basic(self):
        """Test error_response with basic parameters"""
        result = error_response("Test error")
        
        expected = {
            "error": {
                "message": "Test error",
                "status_code": 400
            }
        }
        
        self.assertEqual(result, expected)
    
    def test_error_response_with_details(self):
        """Test error_response with details"""
        details = {"field": "invalid"}
        result = error_response("Test error", status_code=422, details=details)
        
        expected = {
            "error": {
                "message": "Test error",
                "status_code": 422,
                "details": details
            }
        }
        
        self.assertEqual(result, expected)


class TestApiSpecificHandlers(TestCase):
    """Test API-specific error handlers"""
    
    def test_handle_ollama_api_error(self):
        """Test handle_ollama_api_error"""
        http_error = requests.HTTPError()
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Server Error"
        http_error.response = mock_response
        
        result = handle_ollama_api_error(http_error, "/api/test")
        
        self.assertIsInstance(result, APIError)
        self.assertEqual(result.details["api_name"], "Ollama")
    
    def test_handle_openai_api_error(self):
        """Test handle_openai_api_error"""
        http_error = requests.HTTPError()
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        http_error.response = mock_response
        
        result = handle_openai_api_error(http_error, "/v1/chat")
        
        self.assertIsInstance(result, APIError)
        self.assertEqual(result.details["api_name"], "OpenAI")


class TestSafeHTTPClient(TestCase):
    """Test SafeHTTPClient class"""
    
    def setUp(self):
        """Set up test client"""
        self.client = SafeHTTPClient(base_url="https://api.example.com")
    
    def test_init_with_base_url(self):
        """Test client initialization with base URL"""
        client = SafeHTTPClient(base_url="https://api.example.com")
        
        self.assertEqual(client.base_url, "https://api.example.com")
    
    def test_init_with_headers(self):
        """Test client initialization with headers"""
        headers = {"Authorization": "Bearer token"}
        client = SafeHTTPClient(headers=headers)
        
        self.assertEqual(client.headers, headers)
    
    def test_build_url_with_base_url(self):
        """Test URL building with base URL"""
        url = self.client._build_url("/api/test")
        
        self.assertEqual(url, "https://api.example.com/api/test")
    
    def test_build_url_with_full_url(self):
        """Test URL building with full URL"""
        url = self.client._build_url("https://other.example.com/api/test")
        
        self.assertEqual(url, "https://other.example.com/api/test")
    
    def test_build_url_no_base_url(self):
        """Test URL building without base URL"""
        client = SafeHTTPClient()
        url = client._build_url("/api/test")
        
        self.assertEqual(url, "/api/test")
    
    @patch('requests.Session.get')
    def test_get_success(self, mock_get):
        """Test successful GET request"""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.client.get("/test", api_name="TestAPI")
        
        self.assertEqual(result, {"success": True})
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_get_with_params(self, mock_get):
        """Test GET request with parameters"""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        params = {"page": 1, "limit": 10}
        result = self.client.get("/test", params=params, api_name="TestAPI")
        
        mock_get.assert_called_with("https://api.example.com/test", params=params)
    
    @patch('requests.Session.get')
    def test_get_http_error(self, mock_get):
        """Test GET request with HTTP error"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        
        http_error = requests.HTTPError()
        http_error.response = mock_response
        mock_response.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_response
        
        with self.assertRaises(APIError) as context:
            self.client.get("/test", api_name="TestAPI")
        
        self.assertEqual(context.exception.details["status_code"], 404)
    
    @patch('requests.Session.get')
    def test_get_connection_error(self, mock_get):
        """Test GET request with connection error"""
        mock_get.side_effect = requests.ConnectionError("Connection failed")
        
        with self.assertRaises(NetworkError):
            self.client.get("/test", api_name="TestAPI")
    
    @patch('requests.Session.get')
    def test_get_json_decode_error(self, mock_get):
        """Test GET request with JSON decode error"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Invalid JSON"
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = json.JSONDecodeError("Invalid", "doc", 0)
        mock_get.return_value = mock_response
        
        with self.assertRaises(APIError) as context:
            self.client.get("/test", api_name="TestAPI")
        
        self.assertIn("Invalid JSON response", context.exception.message)
    
    @patch('requests.Session.post')
    def test_post_success(self, mock_post):
        """Test successful POST request"""
        mock_response = Mock()
        mock_response.json.return_value = {"created": True}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        json_data = {"name": "test"}
        result = self.client.post("/test", json_data=json_data, api_name="TestAPI")
        
        self.assertEqual(result, {"created": True})
        mock_post.assert_called_with("https://api.example.com/test", json=json_data, data=None)
    
    @patch('requests.Session.put')
    def test_put_success(self, mock_put):
        """Test successful PUT request"""
        mock_response = Mock()
        mock_response.json.return_value = {"updated": True}
        mock_response.raise_for_status.return_value = None
        mock_put.return_value = mock_response
        
        json_data = {"name": "updated"}
        result = self.client.put("/test", json_data=json_data, api_name="TestAPI")
        
        self.assertEqual(result, {"updated": True})
        mock_put.assert_called_with("https://api.example.com/test", json=json_data, data=None)
    
    @patch('requests.Session.delete')
    def test_delete_success(self, mock_delete):
        """Test successful DELETE request"""
        mock_response = Mock()
        mock_response.json.return_value = {"deleted": True}
        mock_response.raise_for_status.return_value = None
        mock_delete.return_value = mock_response
        
        result = self.client.delete("/test", api_name="TestAPI")
        
        self.assertEqual(result, {"deleted": True})
        mock_delete.assert_called_with("https://api.example.com/test")


class TestCreateOllamaClient(TestCase):
    """Test create_ollama_client function"""
    
    @patch('config.get_config')
    def test_create_ollama_client_default_host(self, mock_get_config):
        """Test creating Ollama client with default host"""
        mock_get_config.return_value = {"OLLAMA_HOST": "http://localhost:11434"}
        
        client = create_ollama_client()
        
        self.assertIsInstance(client, SafeHTTPClient)
        self.assertEqual(client.base_url, "http://localhost:11434")
    
    @patch('config.get_config')
    def test_create_ollama_client_custom_host(self, mock_get_config):
        """Test creating Ollama client with custom host"""
        mock_get_config.return_value = {}
        
        client = create_ollama_client(host="http://custom:11434")
        
        self.assertEqual(client.base_url, "http://custom:11434")
    
    @patch('config.get_config')
    def test_create_ollama_client_host_without_http(self, mock_get_config):
        """Test creating Ollama client with host without http prefix"""
        mock_get_config.return_value = {}
        
        client = create_ollama_client(host="localhost:11434")
        
        self.assertEqual(client.base_url, "http://localhost:11434")


class TestErrorMiddleware(TestCase):
    """Test error_middleware function"""
    
    @patch('error_handling.logger')
    def test_middleware_successful_callback(self, mock_logger):
        """Test middleware with successful callback"""
        def test_callback(app_state, action):
            return {"success": True}
        
        wrapped = error_middleware(test_callback)
        result = wrapped({"initial": True}, "test_action")
        
        self.assertEqual(result, {"success": True})
        mock_logger.error.assert_not_called()
    
    @patch('error_handling.logger')
    def test_middleware_error_callback(self, mock_logger):
        """Test middleware with error in callback"""
        def test_callback(app_state, action):
            raise ValueError("Test error")
        
        wrapped = error_middleware(test_callback)
        result = wrapped({"initial": True}, "test_action")
        
        # Should return app state with error
        self.assertIn("error", result)
        self.assertIn("initial", result)
        self.assertEqual(result["initial"], True)
        self.assertEqual(result["error"]["message"], "Test error")
        
        # Should log the error
        mock_logger.error.assert_called()


class TestIntegrationScenarios(TestCase):
    """Test integration scenarios and edge cases"""
    
    def test_error_chain_api_to_network(self):
        """Test error chain from API to network error"""
        # Create a connection error
        conn_error = requests.ConnectionError("DNS resolution failed")
        
        # Convert to network error
        network_error = handle_requests_error(conn_error, "TestAPI", "http://invalid.example.com")
        
        # Verify the chain
        self.assertIsInstance(network_error, NetworkError)
        self.assertEqual(network_error.details["url"], "http://invalid.example.com")
        self.assertEqual(network_error.original_error, conn_error)
    
    def test_error_serialization_roundtrip(self):
        """Test error can be serialized and information preserved"""
        original_error = ValueError("Original issue")
        
        error = APIError(
            message="API request failed",
            api_name="TestAPI",
            status_code=503,
            response_body="Service Unavailable",
            endpoint="/api/test",
            original_error=original_error
        )
        
        # Convert to dict (simulate serialization)
        error_dict = error.to_dict()
        
        # Verify all information is preserved
        self.assertEqual(error_dict["message"], "API request failed")
        self.assertEqual(error_dict["category"], "api")
        self.assertEqual(error_dict["error_code"], "API_503")
        self.assertEqual(error_dict["details"]["api_name"], "TestAPI")
        self.assertEqual(error_dict["details"]["status_code"], 503)
        self.assertEqual(error_dict["original_error"], "Original issue")
    
    @patch('error_handling.logger')
    def test_multiple_decorator_layers(self, mock_logger):
        """Test multiple error handling decorators"""
        
        @capture_exceptions
        @handle_api_error
        def test_func():
            raise requests.Timeout("Request timed out")
        
        with patch('error_handling.display_error'):
            result = test_func()
        
        self.assertIsNone(result)
        # Should be logged by one of the decorators
        self.assertTrue(mock_logger.method_calls)
    
    def test_error_hierarchy_inheritance(self):
        """Test that all custom errors inherit properly"""
        errors = [
            WorkbenchError("base"),
            APIError("api", "TestAPI"),
            NetworkError("network"),
            AuthenticationError("auth"),
            ModelError("model"),
            ParameterError("param")
        ]
        
        for error in errors:
            # All should be instances of WorkbenchError
            self.assertIsInstance(error, WorkbenchError)
            # All should be instances of Exception
            self.assertIsInstance(error, Exception)
            # All should have required methods
            self.assertTrue(hasattr(error, 'to_dict'))
            self.assertTrue(hasattr(error, 'log'))


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
