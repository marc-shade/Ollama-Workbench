import streamlit as st
import logging
import traceback
import sys
import json
import requests
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)

class ErrorLevel(Enum):
    """Error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Categories of errors"""
    API = "api"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    MODEL = "model"
    PARAMETER = "parameter"
    SYSTEM = "system"
    USER = "user"
    DATA = "data"

class WorkbenchError(Exception):
    """
    Base class for Ollama Workbench errors.
    """
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        level: ErrorLevel = ErrorLevel.ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = error_code
        self.category = category
        self.level = level
        self.details = details or {}
        self.original_error = original_error
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation"""
        result = {
            "message": self.message,
            "category": self.category.value,
            "level": self.level.value
        }
        
        if self.error_code:
            result["error_code"] = self.error_code
            
        if self.details:
            result["details"] = self.details
            
        if self.original_error:
            result["original_error"] = str(self.original_error)
            
        return result
    
    def log(self):
        """Log the error with appropriate severity"""
        error_dict = self.to_dict()
        log_message = f"{self.error_code or 'ERROR'}: {self.message}"
        
        if self.details:
            log_message += f" - Details: {json.dumps(self.details)}"
            
        if self.original_error:
            log_message += f" - Original error: {str(self.original_error)}"
            
        if self.level == ErrorLevel.INFO:
            logger.info(log_message)
        elif self.level == ErrorLevel.WARNING:
            logger.warning(log_message)
        elif self.level == ErrorLevel.ERROR:
            logger.error(log_message)
        elif self.level == ErrorLevel.CRITICAL:
            logger.critical(log_message)

class APIError(WorkbenchError):
    """Error when interacting with external APIs"""
    def __init__(
        self,
        message: str,
        api_name: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        endpoint: Optional[str] = None,
        **kwargs
    ):
        details = {
            "api_name": api_name
        }
        
        if status_code:
            details["status_code"] = status_code
            
        if response_body:
            details["response_body"] = response_body
            
        if endpoint:
            details["endpoint"] = endpoint
            
        super().__init__(
            message=message,
            error_code=f"API_{status_code}" if status_code else "API_ERROR",
            category=ErrorCategory.API,
            details=details,
            **kwargs
        )

class NetworkError(WorkbenchError):
    """Error with network connectivity"""
    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        timeout: Optional[bool] = None,
        **kwargs
    ):
        details = {}
        
        if url:
            details["url"] = url
            
        if timeout:
            details["timeout"] = timeout
            
        super().__init__(
            message=message,
            error_code="NETWORK_ERROR",
            category=ErrorCategory.NETWORK,
            details=details,
            **kwargs
        )

class AuthenticationError(WorkbenchError):
    """Error with authentication or API keys"""
    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        **kwargs
    ):
        details = {}
        
        if service:
            details["service"] = service
            
        super().__init__(
            message=message,
            error_code="AUTH_ERROR",
            category=ErrorCategory.AUTHENTICATION,
            details=details,
            **kwargs
        )

class ModelError(WorkbenchError):
    """Error related to AI models"""
    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ):
        details = {}
        
        if model:
            details["model"] = model
            
        if provider:
            details["provider"] = provider
            
        super().__init__(
            message=message,
            error_code="MODEL_ERROR",
            category=ErrorCategory.MODEL,
            details=details,
            **kwargs
        )

class ParameterError(WorkbenchError):
    """Error with function parameters or API parameters"""
    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        details = {}
        
        if parameter:
            details["parameter"] = parameter
            
        if value is not None:
            details["value"] = str(value)
            
        super().__init__(
            message=message,
            error_code="PARAMETER_ERROR",
            category=ErrorCategory.PARAMETER,
            details=details,
            **kwargs
        )

def handle_requests_error(e: Exception, api_name: str, endpoint: Optional[str] = None) -> APIError:
    """
    Convert requests exceptions to APIError
    
    Args:
        e: The original exception
        api_name: Name of the API
        endpoint: Optional endpoint that was called
        
    Returns:
        APIError: Converted error
    """
    # Handle requests.HTTPError
    if isinstance(e, requests.HTTPError):
        try:
            status_code = e.response.status_code
            response_body = e.response.text
        except:
            status_code = None
            response_body = None
            
        return APIError(
            message=f"HTTP error occurred with {api_name}: {str(e)}",
            api_name=api_name,
            status_code=status_code,
            response_body=response_body,
            endpoint=endpoint,
            original_error=e
        )
    
    # Handle requests.ConnectionError
    elif isinstance(e, requests.ConnectionError):
        return NetworkError(
            message=f"Connection error with {api_name}: {str(e)}",
            url=endpoint,
            original_error=e
        )
    
    # Handle requests.Timeout
    elif isinstance(e, requests.Timeout):
        return NetworkError(
            message=f"Timeout occurred with {api_name}: {str(e)}",
            url=endpoint,
            timeout=True,
            original_error=e
        )
    
    # Handle other requests errors
    else:
        return APIError(
            message=f"Error with {api_name}: {str(e)}",
            api_name=api_name,
            endpoint=endpoint,
            original_error=e
        )

def handle_api_error(func):
    """
    Decorator to handle API errors in a standardized way.
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.RequestException as e:
            # Get API name from func name or kwargs
            api_name = kwargs.get("api_name", func.__name__)
            endpoint = kwargs.get("endpoint", None)
            
            # Convert to APIError
            error = handle_requests_error(e, api_name, endpoint)
            
            # Log the error
            error.log()
            
            # Display to user
            display_error(error)
            
            # Return None or empty result based on function return type
            return None
        except Exception as e:
            # General error handling
            error = WorkbenchError(
                message=f"Error in {func.__name__}: {str(e)}",
                original_error=e
            )
            
            # Log the error
            error.log()
            
            # Display to user
            display_error(error)
            
            # Return None or empty result
            return None
    
    return wrapper

def display_error(error: WorkbenchError):
    """
    Display an error to the user using Streamlit.
    
    Args:
        error: The error to display
    """
    if error.level == ErrorLevel.INFO:
        st.info(error.message)
    elif error.level == ErrorLevel.WARNING:
        st.warning(error.message)
    elif error.level == ErrorLevel.ERROR:
        st.error(error.message)
    elif error.level == ErrorLevel.CRITICAL:
        st.error(f"CRITICAL ERROR: {error.message}")
        
    # Optionally show details in an expander
    if error.details or error.original_error:
        with st.expander("Error Details"):
            if error.error_code:
                st.write(f"**Error Code:** {error.error_code}")
                
            if error.category:
                st.write(f"**Category:** {error.category.value}")
                
            if error.details:
                st.write("**Details:**")
                st.json(error.details)
                
            if error.original_error:
                st.write("**Original Error:**")
                st.code(str(error.original_error))

def capture_exceptions(func):
    """
    Decorator to capture all exceptions and display them to the user.
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except WorkbenchError as e:
            # Already a workbench error, just log and display
            e.log()
            display_error(e)
            return None
        except Exception as e:
            # Convert to WorkbenchError
            error = WorkbenchError(
                message=f"Unexpected error: {str(e)}",
                original_error=e
            )
            
            # Log the error
            error.log()
            logger.error(traceback.format_exc())
            
            # Display to user
            display_error(error)
            
            # Return None
            return None
    
    return wrapper

def safe_json_loads(json_str: str, default=None) -> Any:
    """
    Safely parse JSON string, returning default value on error.
    
    Args:
        json_str: JSON string to parse
        default: Default value to return on error
        
    Returns:
        Any: Parsed JSON object or default value
    """
    try:
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"JSON parse error: {str(e)}")
        return default

def get_error_message(e: Exception) -> str:
    """
    Extract a human-readable error message from an exception.
    
    Args:
        e: The exception
        
    Returns:
        str: Human-readable error message
    """
    if hasattr(e, 'message'):
        return str(e.message)
    else:
        return str(e)

# Error response helpers
def error_response(message: str, status_code: int = 400, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        message: Error message
        status_code: HTTP status code
        details: Optional error details
        
    Returns:
        Dict[str, Any]: Error response dictionary
    """
    response = {
        "error": {
            "message": message,
            "status_code": status_code
        }
    }
    
    if details:
        response["error"]["details"] = details
        
    return response

# Common API error handling
def handle_ollama_api_error(e: Exception, endpoint: Optional[str] = None) -> APIError:
    """
    Handle Ollama API errors specifically.
    
    Args:
        e: The exception
        endpoint: Optional API endpoint
        
    Returns:
        APIError: Converted error
    """
    return handle_requests_error(e, api_name="Ollama", endpoint=endpoint)

def handle_openai_api_error(e: Exception, endpoint: Optional[str] = None) -> APIError:
    """
    Handle OpenAI API errors specifically.
    
    Args:
        e: The exception
        endpoint: Optional API endpoint
        
    Returns:
        APIError: Converted error
    """
    return handle_requests_error(e, api_name="OpenAI", endpoint=endpoint)

# HTTP client with error handling
class SafeHTTPClient:
    """
    HTTP client with built-in error handling.
    """
    
    def __init__(self, base_url: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
        """
        Initialize the client.
        
        Args:
            base_url: Optional base URL for all requests
            headers: Optional default headers
        """
        self.base_url = base_url
        self.headers = headers or {}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint"""
        if self.base_url and not endpoint.startswith(('http://', 'https://')):
            return f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        return endpoint
    
    def _handle_response(self, response: requests.Response, endpoint: str, api_name: str) -> Dict[str, Any]:
        """Handle API response"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            raise handle_requests_error(e, api_name=api_name, endpoint=endpoint)
        except json.JSONDecodeError as e:
            raise APIError(
                message="Invalid JSON response",
                api_name=api_name,
                status_code=response.status_code,
                response_body=response.text,
                endpoint=endpoint,
                original_error=e
            )
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, api_name: str = "API") -> Dict[str, Any]:
        """
        Make a GET request.
        
        Args:
            endpoint: API endpoint
            params: Optional query parameters
            api_name: Name of the API for error messages
            
        Returns:
            Dict[str, Any]: Response JSON
        """
        url = self._build_url(endpoint)
        try:
            response = self.session.get(url, params=params)
            return self._handle_response(response, endpoint, api_name)
        except requests.RequestException as e:
            raise handle_requests_error(e, api_name=api_name, endpoint=endpoint)
    
    def post(self, endpoint: str, json_data: Optional[Dict[str, Any]] = None, 
             data: Optional[Any] = None, api_name: str = "API") -> Dict[str, Any]:
        """
        Make a POST request.
        
        Args:
            endpoint: API endpoint
            json_data: Optional JSON data
            data: Optional form data
            api_name: Name of the API for error messages
            
        Returns:
            Dict[str, Any]: Response JSON
        """
        url = self._build_url(endpoint)
        try:
            response = self.session.post(url, json=json_data, data=data)
            return self._handle_response(response, endpoint, api_name)
        except requests.RequestException as e:
            raise handle_requests_error(e, api_name=api_name, endpoint=endpoint)
    
    def put(self, endpoint: str, json_data: Optional[Dict[str, Any]] = None, 
            data: Optional[Any] = None, api_name: str = "API") -> Dict[str, Any]:
        """
        Make a PUT request.
        
        Args:
            endpoint: API endpoint
            json_data: Optional JSON data
            data: Optional form data
            api_name: Name of the API for error messages
            
        Returns:
            Dict[str, Any]: Response JSON
        """
        url = self._build_url(endpoint)
        try:
            response = self.session.put(url, json=json_data, data=data)
            return self._handle_response(response, endpoint, api_name)
        except requests.RequestException as e:
            raise handle_requests_error(e, api_name=api_name, endpoint=endpoint)
    
    def delete(self, endpoint: str, api_name: str = "API") -> Dict[str, Any]:
        """
        Make a DELETE request.
        
        Args:
            endpoint: API endpoint
            api_name: Name of the API for error messages
            
        Returns:
            Dict[str, Any]: Response JSON
        """
        url = self._build_url(endpoint)
        try:
            response = self.session.delete(url)
            return self._handle_response(response, endpoint, api_name)
        except requests.RequestException as e:
            raise handle_requests_error(e, api_name=api_name, endpoint=endpoint)

# Create Ollama client with error handling
def create_ollama_client(host: Optional[str] = None) -> SafeHTTPClient:
    """
    Create a SafeHTTPClient for Ollama API.
    
    Args:
        host: Optional Ollama host
        
    Returns:
        SafeHTTPClient: Client for Ollama API
    """
    from config import get_config
    
    config = get_config()
    host = host or config.get("OLLAMA_HOST", "http://localhost:11434")
    
    # Ensure host has http:// prefix
    if not host.startswith(('http://', 'https://')):
        host = f"http://{host}"
    
    return SafeHTTPClient(base_url=host, headers={"Content-Type": "application/json"})

# Error handling middleware for Streamlit
def error_middleware(callback):
    """
    Error handling middleware for Streamlit callbacks.
    
    Args:
        callback: The callback function
        
    Returns:
        Function: Wrapped callback
    """
    @wraps(callback)
    def wrapper(app_state, action):
        try:
            return callback(app_state, action)
        except Exception as e:
            # Log the error
            logger.error(f"Error in callback: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Update app state with error
            return {
                **app_state,
                "error": {
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
            }
    
    return wrapper