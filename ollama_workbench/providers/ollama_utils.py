# ollama_utils.py

import requests
import json
import io
import logging
import time
from typing import Optional, Dict, Any, Tuple

import streamlit as st
import numpy as np
import ollama
import psutil
import platform
import subprocess
import os

logger = logging.getLogger(__name__)


def _compat_get(obj, key, default=None):
    """Get a value from an ollama response object or dict, compatible with both
    old dict-style responses and new v0.4.8+ object-style responses."""
    # Try attribute access first (v0.4.8+ objects)
    val = getattr(obj, key, None)
    if val is not None:
        return val
    # Try dict-style access (older versions)
    if isinstance(obj, dict):
        return obj.get(key, default)
    # Try __getitem__ (some response objects support it)
    try:
        return obj[key]
    except (KeyError, TypeError, IndexError):
        pass
    return default


def _normalize_response(obj):
    """Convert an ollama response object to a dict for backwards compatibility.
    If already a dict, return as-is. Handles nested response objects recursively."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    # Convert object with __dict__ or known attributes to dict
    result = {}
    if hasattr(obj, '__dict__'):
        for key, val in obj.__dict__.items():
            if not key.startswith('_'):
                # Recursively normalize nested objects
                if hasattr(val, '__dict__') and not isinstance(val, (str, int, float, bool, list, dict, type(None))):
                    result[key] = _normalize_response(val)
                elif isinstance(val, list):
                    result[key] = [_normalize_response(item) if hasattr(item, '__dict__') and not isinstance(item, (str, int, float, bool, dict, type(None))) else item for item in val]
                else:
                    result[key] = val
    # Fallback: try common response attributes
    if not result:
        for attr in ('models', 'model', 'name', 'modified_at', 'digest', 'size', 'details',
                     'modelfile', 'template', 'license', 'modelinfo', 'parameters',
                     'message', 'response', 'context', 'embedding', 'embeddings',
                     'eval_count', 'eval_duration', 'total_duration', 'load_duration',
                     'prompt_eval_count', 'prompt_eval_duration', 'done'):
            val = getattr(obj, attr, None)
            if val is not None:
                if hasattr(val, '__dict__') and not isinstance(val, (str, int, float, bool, list, dict, type(None))):
                    result[attr] = _normalize_response(val)
                elif isinstance(val, list):
                    result[attr] = [_normalize_response(item) if hasattr(item, '__dict__') and not isinstance(item, (str, int, float, bool, dict, type(None))) else item for item in val]
                else:
                    result[attr] = val
    return result

# Performance monitoring constants
PERFORMANCE_THRESHOLDS = {
    "slow_response_time": 5.0,  # seconds
    "very_slow_response_time": 10.0,  # seconds
    "low_tokens_per_second": 10.0,  # tokens/sec
    "very_low_tokens_per_second": 5.0  # tokens/sec
}

# Import observability integration
try:
    from observability import (
        trace_llm_call, 
        add_trace_metadata, 
        log_trace_error, 
        start_trace_span,
        log_performance_metrics,
        is_opik_enabled
    )
    OBSERVABILITY_AVAILABLE = True
    logger.info("Observability integration loaded successfully")
except ImportError as e:
    logger.warning(f"Observability not available: {e}")
    OBSERVABILITY_AVAILABLE = False
    
    # Create fallback functions
    def trace_llm_call(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def add_trace_metadata(metadata):
        pass
    
    def log_trace_error(error, context=None):
        pass
    
    def start_trace_span(name, metadata=None):
        class MockSpan:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
            def set_output(self, data):
                pass
        return MockSpan()
    
    def log_performance_metrics(*args, **kwargs):
        pass
    
    def is_opik_enabled():
        return False
    
    def create_observability_context(*args, **kwargs):
        return {"observability_enabled": False}
    
    def monitor_model_health(*args, **kwargs):
        return {"status": "unknown", "error": "Observability not available"}
    
    def get_operation_metrics(*args, **kwargs):
        return {}

# Import configuration
from ollama_workbench.core.config import get_config

# Legacy files - will be migrated to central config
API_KEYS_FILE = "api_keys.json"
MODEL_SETTINGS_FILE = "model_settings.json"

_api_keys_cache = None
_api_keys_cache_time = 0
_API_KEYS_CACHE_TTL = 10  # seconds

def load_api_keys():
    """Loads API keys from the JSON file. Cached for 10 seconds."""
    global _api_keys_cache, _api_keys_cache_time
    now = time.monotonic()
    if _api_keys_cache is not None and (now - _api_keys_cache_time) < _API_KEYS_CACHE_TTL:
        return _api_keys_cache
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, "r") as f:
            _api_keys_cache = json.load(f)
    else:
        _api_keys_cache = {}
    _api_keys_cache_time = now
    return _api_keys_cache

def save_api_keys(api_keys):
    """Saves API keys to the JSON file."""
    global _api_keys_cache, _api_keys_cache_time
    with open(API_KEYS_FILE, "w") as f:
        json.dump(api_keys, f, indent=4)
    os.chmod(API_KEYS_FILE, 0o600)
    # Invalidate cache so next load_api_keys() re-reads from disk
    _api_keys_cache = None
    _api_keys_cache_time = 0

def load_model_settings():
    """Loads model settings from the JSON file."""
    if os.path.exists(MODEL_SETTINGS_FILE):
        with open(MODEL_SETTINGS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_model_settings(settings):
    """Saves model settings to the JSON file."""
    with open(MODEL_SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)

# Cross-provider imports (MUST be after load_api_keys/save_api_keys to avoid circular import)
from .openai_utils import (
    call_openai_api,
    OPENAI_MODELS,
)
try:
    from .groq_utils import GROQ_MODELS
except ImportError:
    GROQ_MODELS = []
try:
    from .mistral_utils import MISTRAL_MODELS
except ImportError:
    MISTRAL_MODELS = []

# Lazy accessors for dynamic model lists (avoids circular import at module load time)
def get_openai_models():
    from .openai_utils import get_openai_models as _fn
    return _fn()

def get_groq_models():
    from .groq_utils import get_groq_models as _fn
    return _fn()

def get_mistral_models():
    from .mistral_utils import get_mistral_models as _fn
    return _fn()

# Get Ollama URL from config
def get_ollama_url():
    """
    Get the Ollama API URL from the configuration.
    
    Returns:
        str: The Ollama API URL
    """
    config = get_config()
    host = config.get("OLLAMA_HOST", "http://localhost:11434")
    
    # Ensure the host has http:// prefix
    if not host.startswith("http://") and not host.startswith("https://"):
        host = f"http://{host}"
        
    # Add port if not specified
    if ":" not in host.split("//")[1]:
        host = f"{host}:11434"
        
    return f"{host}/api"

def get_ollama_client():
    """
    Get an Ollama client instance configured with the host from config.
    
    Returns:
        ollama.Client or None: A configured client instance or None if client not available
    """
    try:
        config = get_config()
        host = config.get("OLLAMA_HOST", "http://localhost:11434")
        
        # Ensure the host has http:// prefix
        if not host.startswith("http://") and not host.startswith("https://"):
            host = f"http://{host}"
        
        # Handle different versions of the ollama package
        try:
            # First check if Client exists in the ollama module
            if hasattr(ollama, 'Client'):
                # Newer versions of ollama package (>= 0.1.0) use Client class
                client = ollama.Client(host=host)
                
                # Test a simple API call to verify the client works
                try:
                    _ = client.list()  # Try a simple API call
                    return client      # Only return client if it works
                except Exception as test_error:
                    logger.warning(f"Client created but list failed: {test_error}, falling back")
                    # If the test fails, fall through to the module-level functions
            
            # Fallback to older module-level functions approach
            logger.info("Using older version of ollama package without Client class or client failed")
            
            # Set base_url for module-level functions
            if hasattr(ollama, 'base_url'):
                ollama.base_url = host
                
                # Test a simple API call with module-level function
                try:
                    if hasattr(ollama, 'list'):
                        _ = ollama.list()  # Test if module-level function works
                        return None        # Return None to indicate module-level functions should be used
                except Exception as module_error:
                    logger.warning(f"Module-level functions failed: {module_error}")
                    # If both client and module-level functions fail, return None
            
            return None
            
        except (AttributeError, TypeError) as client_error:
            # Older versions (<= 0.4.8) don't have a Client class
            logger.warning(f"Client class not available: {client_error}")
            
            # Set base_url for module-level functions
            if hasattr(ollama, 'base_url'):
                ollama.base_url = host
                
                # Test with module-level function
                try:
                    if hasattr(ollama, 'list'):
                        _ = ollama.list()  # Test if it works
                        return None        # Return None to indicate module-level functions should be used
                except Exception as module_error:
                    logger.warning(f"Module-level functions failed: {module_error}")
            
            return None
    except Exception as e:
        logger.error(f"Unexpected error getting Ollama client: {e}")
        return None

# Dynamic URL based on config
OLLAMA_URL = get_ollama_url()

@st.cache_data(ttl=0)
def get_ollama_resource_usage():
    """Gets Ollama server resource usage."""
    try:
        # Check if Ollama process is running
        for process in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']):
            if process.info['name'] == 'ollama':
                cpu_usage = process.info['cpu_percent']
                memory_usage = process.info['memory_percent']

                # Get GPU usage if available (placeholder for now)
                gpu_usage = "N/A"

                # Check server responsiveness
                response = requests.get("http://localhost:11434/api/tags")
                server_status = "Running" if response.status_code == 200 else "Not Responding"

                return {
                    "status": server_status,
                    "cpu_usage": f"{cpu_usage:.2f}%",
                    "memory_usage": f"{memory_usage:.2f}%",
                    "gpu_usage": gpu_usage
                }

        return {"status": "Not Running", "cpu_usage": "N/A", "memory_usage": "N/A", "gpu_usage": "N/A"}
    except Exception as e:
        return {"status": f"Error: {str(e)}", "cpu_usage": "N/A", "memory_usage": "N/A", "gpu_usage": "N/A"}

@st.cache_data(ttl=5)  # Short TTL to ensure fresh results but avoid hammering the API
def get_available_models():
    """
    Get a list of available Ollama models.
    
    Returns:
        list: A list of model names (strings)
    """
    models = []
    errors = []
    
    # Try multiple methods in sequence for more robust detection
    try:
        # METHOD 1: Try using the client approach first
        client = get_ollama_client()
        
        if client:
            # Newer version with Client class
            try:
                models_response = client.list()
                # v0.4.8+: ListResponse object with .models attribute
                # Older versions: dict with "models" key
                model_list = getattr(models_response, 'models', None)
                if model_list is None and isinstance(models_response, dict):
                    model_list = models_response.get('models', [])

                if model_list:
                    for model in model_list:
                        # v0.4.8+: Model object with .model attribute (NO .name)
                        # Older versions: dict with "name" key
                        name = getattr(model, 'model', None) or (model.get('name') if isinstance(model, dict) else None)
                        if name and 'embed' not in name:
                            models.append(name)

                    if models:
                        logger.info("Found %d models using client.list()", len(models))
                        return models
                else:
                    errors.append("Missing 'models' key/attribute in client response")
            except Exception as client_error:
                errors.append(f"Client list error: {str(client_error)}")
                # Continue to next method
        
        # METHOD 2: Older version fallback - module level functions
        if not models:
            try:
                if hasattr(ollama, 'list'):
                    try:
                        models_response = ollama.list()
                        # v0.4.8+: ListResponse object with .models attribute
                        # Older versions: dict with "models" key
                        model_list = getattr(models_response, 'models', None)
                        if model_list is None and isinstance(models_response, dict):
                            model_list = models_response.get('models', [])

                        if model_list:
                            for model in model_list:
                                # v0.4.8+: Model object with .model attribute (NO .name)
                                # Older versions: dict with "name" key
                                name = getattr(model, 'model', None) or (model.get('name') if isinstance(model, dict) else None)
                                if name and 'embed' not in name:
                                    models.append(name)

                            if models:
                                logger.info("Found %d models using ollama.list()", len(models))
                                return models
                        else:
                            errors.append("Missing 'models' key/attribute in module response")
                    except Exception as list_error:
                        errors.append(f"Module list error: {str(list_error)}")
                        # Continue to next method
            except Exception as module_error:
                errors.append(f"Module approach error: {str(module_error)}")
                # Continue to next method
        
        # METHOD 3: Direct API call fallback
        if not models:
            try:
                # First try the configured URL
                url = f"{get_ollama_url()}/tags"
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    result = response.json()
                    if "models" in result:
                        for model in result["models"]:
                            if "name" in model and "embed" not in model["name"]:
                                models.append(model["name"])
                        
                        if models:
                            logger.info("Found %d models using API at %s", len(models), url)
                            return models
                    else:
                        errors.append("Missing 'models' key in API response")
                else:
                    errors.append(f"API error: {response.status_code}")
                    
                # If the configured URL failed, try the default URL as a fallback
                if not models:
                    default_url = "http://localhost:11434/api/tags"
                    if url != default_url:  # Only try if it's different
                        try:
                            response = requests.get(default_url, timeout=2)
                            if response.status_code == 200:
                                result = response.json()
                                if "models" in result:
                                    for model in result["models"]:
                                        if "name" in model and "embed" not in model["name"]:
                                            models.append(model["name"])
                                    
                                    if models:
                                        logger.info("Found %d models using default API", len(models))
                                        return models
                        except Exception as default_error:
                            errors.append(f"Default URL error: {str(default_error)}")
            except Exception as api_error:
                errors.append(f"API approach error: {str(api_error)}")
        
        # METHOD 4: Last resort - use CLI
        if not models:
            try:
                import subprocess
                result = subprocess.run(
                    ["ollama", "list"], 
                    capture_output=True, 
                    text=True,
                    timeout=3
                )
                
                if result.returncode == 0 and result.stdout:
                    # Parse the CLI output - format is typically NAME TAG SIZE ...
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:  # Skip header row
                        for line in lines[1:]:
                            parts = line.split()
                            if len(parts) >= 2:
                                model_name = parts[0]
                                model_tag = parts[1]
                                full_name = f"{model_name}:{model_tag}" if model_tag != "latest" else model_name
                                if "embed" not in full_name and full_name not in models:
                                    models.append(full_name)
                        
                        if models:
                            logger.info("Found %d models using CLI", len(models))
                            return models
            except Exception as cli_error:
                errors.append(f"CLI approach error: {str(cli_error)}")
                
        # If we got here and still have no models, log all errors
        if not models:
            logger.warning(f"Failed to find models using any method. Errors: {', '.join(errors)}")
            
        return models
            
    except Exception as e:
        logger.error(f"Unexpected error fetching models: {str(e)}")
        return []

def call_ollama_endpoint(model, prompt=None, image=None, temperature=0.5, max_tokens=150, presence_penalty=0.0, frequency_penalty=0.0, context=None, tools=None, episodic_memory=None, format=None, capture_metrics=False, stream=False):
    """
    Call the Ollama API to generate text or process images with comprehensive observability
    
    Args:
        model: The model to use
        prompt: The text prompt
        image: Optional image data (base64 encoded)
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        presence_penalty: Penalty for token presence (0.0 to 1.0)
        frequency_penalty: Penalty for token frequency (0.0 to 1.0)
        context: Optional context for the model
        tools: Optional tools for function calling
        episodic_memory: Optional episodic memory for the model
        format: Optional format for the response (e.g., json)
        capture_metrics: Whether to capture detailed performance metrics using CLI --verbose
        
    Returns:
        Tuple of (response text, context, evaluation count, evaluation duration, detailed metrics)
        The detailed metrics is a dict with performance data when capture_metrics is True
    """
    # Enhanced observability wrapper
    return _call_ollama_with_observability(
        model=model,
        prompt=prompt,
        image=image,
        temperature=temperature,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        context=context,
        tools=tools,
        episodic_memory=episodic_memory,
        format=format,
        capture_metrics=capture_metrics,
        stream=stream
    )


@trace_llm_call(name="ollama_generation")
def _call_ollama_with_observability(model, prompt=None, image=None, temperature=0.5, max_tokens=150, presence_penalty=0.0, frequency_penalty=0.0, context=None, tools=None, episodic_memory=None, format=None, capture_metrics=False, stream=False):
    """
    Internal function that handles the actual Ollama call with comprehensive observability tracing
    """
    start_time = time.time()
    operation_id = f"ollama_{int(start_time * 1000)}"
    
    # Initialize metadata before observability check so it always exists in scope
    metadata = {
        "operation_type": "vision" if image else "text_generation",
    }

    # Enhanced metadata for comprehensive observability
    if OBSERVABILITY_AVAILABLE:
        metadata = {
            "operation_id": operation_id,
            "model": model,
            "provider": "ollama",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "has_image": image is not None,
            "has_tools": tools is not None and len(tools) > 0,
            "has_context": context is not None and len(context) > 0,
            "prompt_length": len(prompt) if prompt else 0,
            "capture_metrics": capture_metrics,
            "operation_type": "vision" if image else "text_generation",
            "timestamp": time.time(),
            "session_id": getattr(st.session_state, 'session_id', 'unknown'),
            "user_id": getattr(st.session_state, 'user_id', 'anonymous'),
            "context_size": len(context) if context else 0,
            "tool_count": len(tools) if tools else 0,
            "format_requested": format is not None,
            "episodic_memory_enabled": episodic_memory is not None
        }
        add_trace_metadata(metadata)
        
        # Log structured operation start
        logger.info(f"Starting Ollama operation", extra={
            "operation_id": operation_id,
            "model": model,
            "operation_type": metadata["operation_type"],
            "prompt_length": metadata["prompt_length"]
        })
    
    try:
        # Decide whether to use CLI with --verbose based on capture_metrics flag
        if capture_metrics:
            # Use CLI approach with --verbose to get detailed metrics
            return call_ollama_cli_verbose(
                model=model, 
                prompt=prompt, 
                temperature=temperature, 
                max_tokens=max_tokens, 
                tools=tools
            )
        
        # If not capturing metrics, use the original implementation
        response, context_info, eval_count, eval_duration = _call_ollama_endpoint_impl(
            model=model,
            prompt=prompt,
            image=image,
            temperature=temperature,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            context=context,
            tools=tools,
            episodic_memory=episodic_memory,
            format=format,
            stream=stream
        )
        
        # Calculate comprehensive performance metrics
        elapsed_time = time.time() - start_time
        tokens_per_second = eval_count / elapsed_time if eval_count and elapsed_time > 0 else 0
        
        # Enhanced performance monitoring with thresholds
        performance_data = {
            "response_time": elapsed_time,
            "token_count": eval_count or 0,
            "tokens_per_second": tokens_per_second,
            "model_name": model,
            "operation_type": metadata["operation_type"],
            "operation_id": operation_id,
            "is_slow_response": elapsed_time > PERFORMANCE_THRESHOLDS["slow_response_time"],
            "is_very_slow_response": elapsed_time > PERFORMANCE_THRESHOLDS["very_slow_response_time"],
            "is_low_throughput": tokens_per_second < PERFORMANCE_THRESHOLDS["low_tokens_per_second"],
            "is_very_low_throughput": tokens_per_second < PERFORMANCE_THRESHOLDS["very_low_tokens_per_second"]
        }
        
        # Log performance metrics to observability system
        if OBSERVABILITY_AVAILABLE:
            log_performance_metrics(**performance_data)
            
        # Log performance alerts for slow operations
        if performance_data["is_very_slow_response"]:
            logger.warning(f"Very slow response detected", extra={
                "operation_id": operation_id,
                "model": model,
                "response_time": elapsed_time,
                "threshold": PERFORMANCE_THRESHOLDS["very_slow_response_time"]
            })
        elif performance_data["is_slow_response"]:
            logger.info(f"Slow response detected", extra={
                "operation_id": operation_id,
                "model": model,
                "response_time": elapsed_time,
                "threshold": PERFORMANCE_THRESHOLDS["slow_response_time"]
            })
            
        if performance_data["is_very_low_throughput"] and eval_count:
            logger.warning(f"Very low throughput detected", extra={
                "operation_id": operation_id,
                "model": model,
                "tokens_per_second": tokens_per_second,
                "threshold": PERFORMANCE_THRESHOLDS["very_low_tokens_per_second"]
            })
        
        # Log model usage for analytics if available
        try:
            from ollama_workbench.models.model_management import log_model_usage, log_model_performance
            
            log_model_usage(
                model_name=model,
                tokens_generated=eval_count or 0,
                response_time=elapsed_time,
                operation_type=metadata.get("operation_type", "generate")
            )
            
            # Log performance metrics if we have the data
            if eval_count and eval_duration:
                tokens_per_second = eval_count / (eval_duration / (10**9))
                log_model_performance(
                    model_name=model,
                    prompt_text=prompt[:100] if prompt and len(prompt) > 100 else prompt or "Empty prompt",
                    tokens_per_second=tokens_per_second,
                    latency=elapsed_time,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
        except (ImportError, Exception) as log_error:
            logger.debug(f"Could not log model usage: {str(log_error)}")
        
        return response, context_info, eval_count, eval_duration, {}
        
    except Exception as e:
        # Enhanced error tracking with comprehensive context
        elapsed_time = time.time() - start_time
        error_context = {
            "operation_id": operation_id,
            "model": model,
            "prompt_length": len(prompt) if prompt else 0,
            "has_image": image is not None,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "elapsed_time": elapsed_time,
            "operation_type": "vision" if image else "text_generation",
            "provider": "ollama",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "capture_metrics": capture_metrics,
            "timestamp": time.time()
        }
        
        if OBSERVABILITY_AVAILABLE:
            log_trace_error(e, error_context)
        
        # Structured error logging
        logger.error(f"Ollama operation failed", extra={
            "operation_id": operation_id,
            "model": model,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "elapsed_time": elapsed_time,
            "operation_type": "vision" if image else "text_generation"
        })
        
        # Log error patterns for analysis
        try:
            from ollama_workbench.models.model_management import log_model_error
            log_model_error(
                model_name=model,
                error_type=type(e).__name__,
                error_message=str(e),
                operation_type="vision" if image else "text_generation",
                context=error_context
            )
        except (ImportError, Exception) as log_error:
            logger.debug(f"Could not log model error: {str(log_error)}")
            
        raise
            
def call_ollama_cli_verbose(model, prompt, temperature=0.5, max_tokens=150, tools=None):
    """
    Call Ollama using the CLI with --verbose to capture detailed performance metrics
    """
    import subprocess
    import json
    import tempfile
    import os
    import time
    
    start_time = time.time()
    
    # Create a temporary file to store the prompt
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as prompt_file:
        prompt_file.write(prompt)
        prompt_path = prompt_file.name
    
    # Create a temporary file for the tools if provided
    tools_path = None
    if tools:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tools_file:
            json.dump(tools, tools_file)
            tools_path = tools_file.name
    
    try:
        # Build the command
        command = ["ollama", "run", "--verbose", model]
        
        # Add parameters
        if temperature != 0.5:
            command.extend(["--temperature", str(temperature)])
        if max_tokens != 150:
            command.extend(["--num-predict", str(max_tokens)])
        if tools_path:
            command.extend(["--tools", tools_path])
            
        # Run command with stdin from prompt file (no shell=True needed)
        with open(prompt_path, 'r') as stdin_file:
            result = subprocess.run(command, stdin=stdin_file, capture_output=True, text=True)
        
        # Extract the response and metrics
        response_text = ""
        metrics = {}
        
        if result.returncode == 0:
            # The output is on stdout
            output_lines = result.stdout.strip().split('\n')
            
            # The response is all lines except the last few which contain metrics
            response_text = '\n'.join(output_lines[:-5] if len(output_lines) > 5 else output_lines)
            
            # Parse metrics from the last few lines
            metric_lines = output_lines[-5:] if len(output_lines) > 5 else []
            for line in metric_lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    metrics[key] = value
        else:
            # Error occurred
            response_text = f"CLI error: {result.stderr}"
            logger.error(f"CLI error: {result.stderr}")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        metrics["total_wall_time"] = f"{elapsed_time:.2f}s"
        
        # Log model usage for analytics if available
        try:
            from ollama_workbench.models.model_management import log_model_usage, log_model_performance
            
            # Extract token count and other metrics
            eval_count = 0
            if "eval count" in metrics:
                try:
                    eval_count = int(metrics["eval count"].split()[0])
                except (ValueError, KeyError, IndexError):
                    pass
            
            # Log usage
            log_model_usage(
                model_name=model,
                tokens_generated=eval_count,
                response_time=elapsed_time,
                operation_type="generate_cli_verbose"
            )
            
            # Extract and log performance metrics
            if "eval rate" in metrics:
                try:
                    tokens_per_second = float(metrics["eval rate"].split()[0])
                    log_model_performance(
                        model_name=model,
                        prompt_text=prompt[:100] if prompt and len(prompt) > 100 else prompt or "Empty prompt",
                        tokens_per_second=tokens_per_second,
                        latency=elapsed_time,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                except (ValueError, KeyError, IndexError):
                    pass
        except (ImportError, Exception) as log_error:
            logger.debug(f"Could not log model usage: {str(log_error)}")
        
        return response_text, None, metrics.get("eval count", None), metrics.get("eval duration", None), metrics
    
    finally:
        # Clean up temporary files
        try:
            os.unlink(prompt_path)
            if tools_path:
                os.unlink(tools_path)
        except OSError:
            pass

def _call_ollama_endpoint_impl(model, prompt=None, image=None, temperature=0.5, max_tokens=150, presence_penalty=0.0, frequency_penalty=0.0, context=None, tools=None, episodic_memory=None, format=None, stream=False):
    import time
    start_time = time.time()
    
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "context": context if context is not None else [],
        "tools": tools if tools is not None else []
    }
    if prompt:
        payload["prompt"] = prompt
    if format:
        payload["format"] = format
    
    # Get the client, which might be None for older versions
    client = get_ollama_client()
    
    # Handle image processing
    if image:
        try:
            if client:
                # Newer version with Client class
                if prompt:
                    # For the chat API with images
                    messages = [
                        {
                            'role': 'user',
                            'content': prompt,
                            'images': [image]
                        }
                    ]
                    response = client.chat(model=model, messages=messages, options={
                        "temperature": temperature,
                        "num_predict": max_tokens
                    })
                    # v0.4.8+: ChatResponse object — normalize for dict access
                    response = _normalize_response(response)

                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time

                    # Log model usage for analytics if the module is available
                    try:
                        from ollama_workbench.models.model_management import log_model_usage, log_model_performance
                        # Log usage statistics
                        eval_count = response.get('eval_count', 0)
                        eval_duration = response.get('eval_duration', 0)

                        log_model_usage(
                            model_name=model,
                            tokens_generated=eval_count,
                            response_time=elapsed_time,
                            operation_type="vision"
                        )

                        # Log performance metrics if we have the data
                        if eval_count and eval_duration:
                            tokens_per_second = eval_count / (eval_duration / (10**9))
                            log_model_performance(
                                model_name=model,
                                prompt_text="Vision prompt" if len(prompt) > 100 else prompt[:100],
                                tokens_per_second=tokens_per_second,
                                latency=elapsed_time,
                                temperature=temperature,
                                max_tokens=max_tokens
                            )
                    except (ImportError, Exception) as log_error:
                        logger.debug(f"Could not log model usage: {str(log_error)}")

                    msg = response.get('message', {})
                    content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
                    return content, response.get('context', None), response.get('eval_count', None), response.get('eval_duration', None)
                else:
                    # If no prompt is provided, use default "Describe this image"
                    messages = [
                        {
                            'role': 'user',
                            'content': 'Describe this image:',
                            'images': [image]
                        }
                    ]
                    response = client.chat(model=model, messages=messages, options={
                        "temperature": temperature,
                        "num_predict": max_tokens
                    })
                    # v0.4.8+: ChatResponse object — normalize for dict access
                    response = _normalize_response(response)

                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time

                    # Log model usage for analytics if the module is available
                    try:
                        from ollama_workbench.models.model_management import log_model_usage, log_model_performance
                        # Log usage statistics
                        eval_count = response.get('eval_count', 0)
                        eval_duration = response.get('eval_duration', 0)

                        log_model_usage(
                            model_name=model,
                            tokens_generated=eval_count,
                            response_time=elapsed_time,
                            operation_type="vision"
                        )
                        
                        # Log performance metrics if we have the data
                        if eval_count and eval_duration:
                            tokens_per_second = eval_count / (eval_duration / (10**9))
                            log_model_performance(
                                model_name=model,
                                prompt_text="Describe this image (default vision prompt)",
                                tokens_per_second=tokens_per_second,
                                latency=elapsed_time,
                                temperature=temperature,
                                max_tokens=max_tokens
                            )
                    except (ImportError, Exception) as log_error:
                        logger.debug(f"Could not log model usage: {str(log_error)}")
                    
                    msg = response.get('message', {})
                    content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
                    return content, response.get('context', None), response.get('eval_count', None), response.get('eval_duration', None)
            else:
                # Older version fallback - use direct API calls
                st.warning("Using older Ollama package version. Images might not be supported correctly.")
                # For older versions, we'll use direct HTTP requests
                image_base64 = image if isinstance(image, str) else ""
                url = f"{get_ollama_url()}/generate"
                payload["image"] = image_base64
                payload["prompt"] = prompt if prompt else "Describe this image:"
                response = requests.post(url, json=payload)
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Log model usage for analytics if the module is available
                    try:
                        from ollama_workbench.models.model_management import log_model_usage, log_model_performance
                        # Log usage statistics
                        eval_count = result.get('eval_count', 0)
                        eval_duration = result.get('eval_duration', 0)
                        
                        log_model_usage(
                            model_name=model,
                            tokens_generated=eval_count,
                            response_time=elapsed_time,
                            operation_type="vision"
                        )
                        
                        # Log performance metrics if we have the data
                        if eval_count and eval_duration:
                            tokens_per_second = eval_count / (eval_duration / (10**9))
                            log_model_performance(
                                model_name=model,
                                prompt_text="Vision prompt (direct API)" if prompt else "Describe this image (default vision prompt)",
                                tokens_per_second=tokens_per_second,
                                latency=elapsed_time,
                                temperature=temperature,
                                max_tokens=max_tokens
                            )
                    except (ImportError, Exception) as log_error:
                        logger.debug(f"Could not log model usage: {str(log_error)}")
                    
                    return result.get("response", ""), result.get("context", None), result.get("eval_count", None), result.get("eval_duration", None)
                else:
                    return f"API error with multimodal processing: {response.text}", None, None, None
        except Exception as e:
            return f"An error occurred with multimodal processing: {str(e)}", None, None, None
    
    # Handle text generation
    try:
        if client:
            # Newer version with Client class
            response = client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty
                },
                stream=True
            )
            
            # If streaming is requested, return the generator
            if stream:
                return response, None, None, None
            
            # Otherwise, consume the stream and return the full response
            response_parts = []
            eval_count = None
            eval_duration = None
            context_info = None
            
            for chunk in response:
                # v0.4.8+: streaming chunks may be GenerateResponse objects
                response_parts.append(_compat_get(chunk, "response", ""))
                if _compat_get(chunk, "done", False):
                    eval_count = _compat_get(chunk, "eval_count", None)
                    eval_duration = _compat_get(chunk, "eval_duration", None)
                    context_info = _compat_get(chunk, "context", None)

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            # Log model usage for analytics if the module is available
            try:
                from ollama_workbench.models.model_management import log_model_usage, log_model_performance
                # Log usage statistics
                log_model_usage(
                    model_name=model,
                    tokens_generated=eval_count or 0,
                    response_time=elapsed_time,
                    operation_type="generate"
                )
                
                # Log performance metrics if we have the data
                if eval_count and eval_duration:
                    tokens_per_second = eval_count / (eval_duration / (10**9))
                    log_model_performance(
                        model_name=model,
                        prompt_text=prompt[:100] if prompt and len(prompt) > 100 else prompt or "Empty prompt",
                        tokens_per_second=tokens_per_second,
                        latency=elapsed_time,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
            except (ImportError, Exception) as log_error:
                logger.debug(f"Could not log model usage: {str(log_error)}")
            
            return "".join(response_parts), context_info, eval_count, eval_duration
        else:
            # Older version fallback - use module level functions
            try:
                # First try module.generate method
                if hasattr(ollama, 'generate'):
                    response = ollama.generate(
                        model=model,
                        prompt=prompt,
                        options={
                            "temperature": temperature,
                            "num_predict": max_tokens,
                            "presence_penalty": presence_penalty,
                            "frequency_penalty": frequency_penalty
                        },
                        stream=True
                    )
                    
                    # If streaming is requested, return the generator
                    if stream:
                        return response, None, None, None
                    
                    # Otherwise, consume the stream and return the full response
                    response_parts = []
                    eval_count = None
                    eval_duration = None
                    context_info = None
                    
                    for chunk in response:
                        # v0.4.8+: streaming chunks may be GenerateResponse objects
                        response_parts.append(_compat_get(chunk, "response", ""))
                        if _compat_get(chunk, "done", False):
                            eval_count = _compat_get(chunk, "eval_count", None)
                            eval_duration = _compat_get(chunk, "eval_duration", None)
                            context_info = _compat_get(chunk, "context", None)

                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time

                    # Log model usage for analytics if the module is available
                    try:
                        from ollama_workbench.models.model_management import log_model_usage, log_model_performance
                        # Log usage statistics
                        log_model_usage(
                            model_name=model,
                            tokens_generated=eval_count or 0,
                            response_time=elapsed_time,
                            operation_type="generate"
                        )

                        # Log performance metrics if we have the data
                        if eval_count and eval_duration:
                            tokens_per_second = eval_count / (eval_duration / (10**9))
                            log_model_performance(
                                model_name=model,
                                prompt_text=prompt[:100] if prompt and len(prompt) > 100 else prompt or "Empty prompt",
                                tokens_per_second=tokens_per_second,
                                latency=elapsed_time,
                                temperature=temperature,
                                max_tokens=max_tokens
                            )
                    except (ImportError, Exception) as log_error:
                        logger.debug(f"Could not log model usage: {str(log_error)}")
                    
                    return "".join(response_parts), context_info, eval_count, eval_duration
                else:
                    # Last resort: fall back to direct API call
                    url = f"{get_ollama_url()}/generate"
                    response = requests.post(url, json=payload)
                    
                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Log model usage for analytics if the module is available
                        try:
                            from ollama_workbench.models.model_management import log_model_usage, log_model_performance
                            # Log usage statistics
                            eval_count = result.get('eval_count', 0)
                            eval_duration = result.get('eval_duration', 0)
                            
                            log_model_usage(
                                model_name=model,
                                tokens_generated=eval_count,
                                response_time=elapsed_time,
                                operation_type="generate"
                            )
                            
                            # Log performance metrics if we have the data
                            if eval_count and eval_duration:
                                tokens_per_second = eval_count / (eval_duration / (10**9))
                                log_model_performance(
                                    model_name=model,
                                    prompt_text=prompt[:100] if prompt and len(prompt) > 100 else prompt or "Empty prompt",
                                    tokens_per_second=tokens_per_second,
                                    latency=elapsed_time,
                                    temperature=temperature,
                                    max_tokens=max_tokens
                                )
                        except (ImportError, Exception) as log_error:
                            logger.debug(f"Could not log model usage: {str(log_error)}")
                        
                        return result.get("response", ""), result.get("context", None), result.get("eval_count", None), result.get("eval_duration", None)
                    else:
                        return f"API error: {response.text}", None, None, None
            except Exception as inner_e:
                # If module level functions fail, fall back to direct API call
                logger.warning(f"Module level fallback failed: {str(inner_e)}, using direct API call")
                url = f"{get_ollama_url()}/generate"
                response = requests.post(url, json=payload)
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Log model usage for analytics if the module is available
                    try:
                        from ollama_workbench.models.model_management import log_model_usage, log_model_performance
                        # Log usage statistics
                        eval_count = result.get('eval_count', 0)
                        eval_duration = result.get('eval_duration', 0)
                        
                        log_model_usage(
                            model_name=model,
                            tokens_generated=eval_count,
                            response_time=elapsed_time,
                            operation_type="generate"
                        )
                        
                        # Log performance metrics if we have the data
                        if eval_count and eval_duration:
                            tokens_per_second = eval_count / (eval_duration / (10**9))
                            log_model_performance(
                                model_name=model,
                                prompt_text=prompt[:100] if prompt and len(prompt) > 100 else prompt or "Empty prompt",
                                tokens_per_second=tokens_per_second,
                                latency=elapsed_time,
                                temperature=temperature,
                                max_tokens=max_tokens
                            )
                    except (ImportError, Exception) as log_error:
                        logger.debug(f"Could not log model usage: {str(log_error)}")
                    
                    return result.get("response", ""), result.get("context", None), result.get("eval_count", None), result.get("eval_duration", None)
                else:
                    return f"API error: {response.text}", None, None, None
    except Exception as e:
        logger.error(f"Error in call_ollama_endpoint: {str(e)}")
        return f"An error occurred: {str(e)}", None, None, None

def check_json_handling(model, temperature, max_tokens, presence_penalty, frequency_penalty):
    prompt = "Return the following data in JSON format: name: John, age: 30, city: New York"
    result, _, _, _, _ = call_ollama_endpoint(model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)
    try:
        json.loads(result)
        return True
    except json.JSONDecodeError:
        return False

def get_token_embeddings(model: str, text: str, api_keys: dict) -> np.ndarray:
    """Gets embeddings for each token in the text and returns a 2D array."""
    try:
        if model in get_openai_models():
            response = call_openai_api(
                "text-embedding-ada-002",  # OpenAI's embedding model
                prompt=[{"role": "user", "content": text}],
                openai_api_key=api_keys.get("openai_api_key"),
                use_chat=False
            )
            embeddings = np.array(response['data'][0]['embedding'])
        elif model in get_groq_models():
            from .groq_utils import get_local_embeddings
            embeddings = get_local_embeddings(text)
        else:
            client = get_ollama_client()
            
            if client:
                # Newer version with Client class
                response = client.embeddings(model=model, prompt=text)
                # v0.4.8+: EmbedResponse object — use _compat_get
                embedding_data = _compat_get(response, 'embedding')
                embeddings = np.array(embedding_data)
            else:
                # Older version fallback - use module level functions
                try:
                    if hasattr(ollama, 'embeddings'):
                        response = ollama.embeddings(model=model, prompt=text)
                        embedding_data = _compat_get(response, 'embedding')
                        embeddings = np.array(embedding_data)
                    else:
                        # Last resort: fall back to direct API call
                        url = f"{get_ollama_url()}/embeddings"
                        response = requests.post(url, json={"model": model, "prompt": text})
                        if response.status_code == 200:
                            result = response.json()
                            embeddings = np.array(result['embedding'])
                        else:
                            logger.error(f"API error: {response.text}")
                            return np.zeros((1, 1536))
                except Exception as inner_e:
                    # If module level functions fail, fall back to direct API call
                    logger.warning(f"Module level fallback failed: {str(inner_e)}, using direct API call")
                    url = f"{get_ollama_url()}/embeddings"
                    response = requests.post(url, json={"model": model, "prompt": text})
                    if response.status_code == 200:
                        result = response.json()
                        embeddings = np.array(result['embedding'])
                    else:
                        logger.error(f"API error: {response.text}")
                        return np.zeros((1, 1536))
        
        return embeddings.reshape(1, -1)  # Ensure it's a 2D array
    except Exception as e:
        logger.error(f"An error occurred while getting token embeddings: {e}")
        st.error(f"An error occurred while getting token embeddings: {e}")
        return np.zeros((1, 1536))  # Return a default 2D array with 1536 features (common embedding size)

def check_function_calling(model, temperature, max_tokens, presence_penalty, frequency_penalty):
    prompt = "Define a function named 'add' that takes two numbers and returns their sum. Then call the function with arguments 5 and 3."
    result, _, _, _, _ = call_ollama_endpoint(model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)
    return "8" in result

def run_tool_test(model, tool_description, arguments):
    prompt = f"Test the function: {tool_description}. Arguments: {arguments}"
    result, _, _, _, _ = call_ollama_endpoint(model, prompt=prompt)
    return result

def pull_model(model_name):
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    st.write(f"📥 Pulling model: `{model_name}`")
    
    try:
        client = get_ollama_client()
        
        if client:
            # Newer version with Client class
            try:
                # Try the new streaming API approach (newer versions)
                try:
                    # First attempt: Use the new stream API without callback
                    stream_response = client.pull(model=model_name, stream=True)
                    # Process the stream responses
                    for response in stream_response:
                        # v0.4.8+: ProgressResponse objects, not dicts — normalize
                        if not isinstance(response, dict):
                            response = _normalize_response(response)
                        if isinstance(response, dict):
                            if "status" in response:
                                status_text.text(response["status"])
                                results.append(response)

                            # Handle progress updates
                            if "completed" in response and "total" in response and response["total"] > 0:
                                total = response["total"]
                                completed = response["completed"]
                                progress = completed / total
                                progress_bar.progress(progress)
                                status_text.text(f"Progress: {progress * 100:.2f}%")
                    
                    # Set progress to 100% when done
                    progress_bar.progress(1.0)
                    status_text.text("Download complete!")
                    return results
                    
                except TypeError as callback_error:
                    # If the API changed and doesn't take callback parameter
                    if "callback" in str(callback_error):
                        logger.warning("Client API doesn't support callback parameter, using alternative methods")
                    else:
                        # Re-raise if it's a different TypeError
                        raise callback_error
            except Exception as client_pull_error:
                logger.warning(f"Client pull with stream failed: {str(client_pull_error)}")
                
                # Try simple pull without streaming if streaming fails
                try:
                    # Use non-streaming pull as fallback
                    status_text.text("Using non-streaming pull method...")
                    response = client.pull(model=model_name)
                    # Simple response without progress updates
                    status_text.text("Download complete!")
                    progress_bar.progress(1.0)
                    return [{"status": "Download complete (non-streaming)"}]
                except Exception as non_stream_error:
                    logger.warning(f"Non-streaming pull failed: {str(non_stream_error)}")
                    # Continue to fallback methods
        
        # Direct API call fallback - should work with most Ollama API versions
        try:
            status_text.text("Using direct API call to pull model...")
            api_url = f"{get_ollama_url()}/pull"
            
            # Start a session for persistent connection
            session = requests.Session()
            
            # Make a streaming request
            response = session.post(
                api_url, 
                json={"name": model_name},
                stream=True  # Use streaming mode
            )
            
            if response.status_code == 200:
                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "status" in data:
                                status_text.text(data["status"])
                                results.append(data)
                            
                            # Handle progress updates
                            if "completed" in data and "total" in data and data["total"] > 0:
                                total = data["total"]
                                completed = data["completed"]
                                progress = completed / total
                                progress_bar.progress(progress)
                                status_text.text(f"Progress: {progress * 100:.2f}%")
                        except json.JSONDecodeError:
                            # Not valid JSON, just show the line
                            status_text.text(line.decode('utf-8'))
                
                # Set progress to 100% when done
                progress_bar.progress(1.0)
                status_text.text("Download complete!")
                return results
            else:
                raise Exception(f"API error: {response.status_code}")
        except Exception as api_error:
            logger.warning(f"Direct API call failed: {str(api_error)}")
                
        # Last resort: use CLI
        try:
            status_text.text("Using CLI to pull model...")
            import subprocess
            process = subprocess.Popen(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in iter(process.stdout.readline, ''):
                status_text.text(line.strip())
                results.append({"status": line.strip()})
                
                # Simple progress parsing from CLI output
                if "%" in line:
                    try:
                        percent = float(line.split("%")[0].split()[-1]) / 100
                        progress_bar.progress(percent)
                    except (ValueError, IndexError):
                        pass
            
            process.stdout.close()
            return_code = process.wait()
            
            if return_code == 0:
                progress_bar.progress(1.0)
                status_text.text("Download complete!")
                return results
            else:
                raise Exception(f"CLI command failed with exit code {return_code}")
        except Exception as cli_error:
            logger.error(f"CLI pull failed: {str(cli_error)}")
            st.error(f"Failed to pull model: {str(cli_error)}")
            return []
            
    except Exception as e:
        logger.error(f"Error pulling model: {str(e)}")
        st.error(f"Error pulling model: {str(e)}")
        return []

def show_model_info(model_name):
    """Get model info. Always returns a dict for backwards compatibility,
    converting v0.4.8+ ShowResponse objects as needed."""
    try:
        client = get_ollama_client()

        if client:
            # Newer version with Client class
            response = client.show(model=model_name)
            # v0.4.8+: ShowResponse object, not a dict — normalize it
            return _normalize_response(response)
        else:
            # Older version fallback - use module level functions
            try:
                if hasattr(ollama, 'show'):
                    response = ollama.show(model=model_name)
                    # v0.4.8+: ShowResponse object — normalize it
                    return _normalize_response(response)
                else:
                    # Last resort: fall back to direct API call
                    url = f"{get_ollama_url()}/show"
                    response = requests.post(url, json={"name": model_name})
                    if response.status_code == 200:
                        return response.json()
                    else:
                        logger.error(f"API error: {response.text}")
                        return {}
            except Exception as inner_e:
                # If module level functions fail, fall back to direct API call
                logger.warning(f"Module level fallback failed: {str(inner_e)}, using direct API call")
                url = f"{get_ollama_url()}/show"
                response = requests.post(url, json={"name": model_name})
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"API error: {response.text}")
                    return {}
    except Exception as e:
        logger.error(f"Error fetching model info: {str(e)}")
        st.error(f"Error fetching model info: {str(e)}")
        return {}

def remove_model(model_name):
    try:
        client = get_ollama_client()
        
        if client:
            # Newer version with Client class
            client.delete(model=model_name)
            return {"status": "success", "message": f"Model '{model_name}' removed successfully."}
        else:
            # Older version fallback - use module level functions
            try:
                if hasattr(ollama, 'delete'):
                    ollama.delete(model=model_name)
                    return {"status": "success", "message": f"Model '{model_name}' removed successfully."}
                else:
                    # Last resort: fall back to direct API call or CLI
                    # Try the API first
                    url = f"{get_ollama_url()}/delete"
                    response = requests.delete(url, json={"name": model_name})
                    if response.status_code == 200:
                        return {"status": "success", "message": f"Model '{model_name}' removed successfully."}
                    
                    # If API fails, try CLI
                    import subprocess
                    result = subprocess.run(["ollama", "rm", model_name], capture_output=True)
                    if result.returncode == 0:
                        return {"status": "success", "message": f"Model '{model_name}' removed successfully."}
                    else:
                        error_msg = result.stderr.decode()
                        return {"status": "error", "message": f"Failed to remove model via CLI: {error_msg}"}
            except Exception as inner_e:
                # If module level functions fail, fall back to direct API call
                logger.warning(f"Module level fallback failed: {str(inner_e)}, trying CLI")
                
                # Try CLI as last resort
                try:
                    import subprocess
                    result = subprocess.run(["ollama", "rm", model_name], capture_output=True)
                    if result.returncode == 0:
                        return {"status": "success", "message": f"Model '{model_name}' removed successfully."}
                    else:
                        error_msg = result.stderr.decode()
                        return {"status": "error", "message": f"Failed to remove model via CLI: {error_msg}"}
                except Exception as final_e:
                    return {"status": "error", "message": f"All remove methods failed: {str(final_e)}"}
    except Exception as e:
        logger.error(f"Failed to remove model '{model_name}': {str(e)}")
        return {"status": "error", "message": f"Failed to remove model '{model_name}': {str(e)}"}

def save_chat_history(chat_history, filename="chat_history.json"):
    with open(filename, "w") as f:
        json.dump(chat_history, f)

def load_chat_history(filename):
    with open(filename, "r") as f:
        return json.load(f)

def update_model_selection(selected_models, key):
    st.session_state[key] = selected_models

def preload_model(model_name):
    """Preloads a model into memory."""
    try:
        response = requests.post(f"{OLLAMA_URL}/generate", json={"model": model_name})
        response.raise_for_status()
        st.success(f"Model '{model_name}' preloaded successfully!")
    except requests.exceptions.RequestException as e:
        st.error(f"Error preloading model '{model_name}': {e}")

def stop_server():
    """Stops the Ollama server."""
    try:
        subprocess.run(["osascript", "-e", 'tell app "Ollama" to quit'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        st.success("Ollama server stopped.")
    except Exception as e:
        st.error(f"Error stopping Ollama server: {e}")

def apply_server_settings(host, origins, model_dir, global_keep_alive, max_loaded_models, num_parallel, max_queue):
    """Applies server settings using environment variables."""
    try:
        if host:
            os.environ["OLLAMA_HOST"] = host
        if origins:
            formatted_origins = " ".join([f"http://{origin.strip()}" for origin in origins.split(",")])
            os.environ["OLLAMA_ORIGINS"] = formatted_origins
        if model_dir:
            os.environ["OLLAMA_MODELS"] = model_dir
        if global_keep_alive:
            os.environ["OLLAMA_KEEP_ALIVE"] = global_keep_alive
        if max_loaded_models:
            os.environ["OLLAMA_MAX_LOADED_MODELS"] = str(max_loaded_models)
        if num_parallel:
            os.environ["OLLAMA_NUM_PARALLEL"] = str(num_parallel)
        if max_queue:
            os.environ["OLLAMA_MAX_QUEUE"] = str(max_queue)
        st.success("Server settings applied. Please restart the server.")
    except Exception as e:
        st.error(f"Error applying server settings: {e}")

def start_server():
    """Starts the Ollama server."""
    try:
        subprocess.Popen(["ollama", "serve"])
        st.success("Ollama server started.")
    except Exception as e:
        st.error(f"Error starting Ollama server: {e}")

def apply_model_keep_alive(model_name, keep_alive):
    """Applies keep-alive settings for a specific model using the ollama CLI."""
    try:
        if keep_alive:
            command = ["ollama", "run", model_name, "", "--keep-alive", keep_alive]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode == 0:
                st.success(f"Keep-Alive '{keep_alive}' applied for '{model_name}'!")
            else:
                st.error(f"Error applying Keep-Alive for '{model_name}': {result.stderr.decode()}")
    except Exception as e:
        st.error(f"Error applying Keep-Alive for '{model_name}': {e}")

def get_log_file_path():
    """Returns the platform-specific path to the Ollama server log file."""
    system = platform.system()
    if system == "Darwin":
        return os.path.expanduser("~/.ollama/logs/server.log")
    elif system == "Linux":
        return "/var/log/ollama/server.log"  # Assuming a standard location for Linux
    elif system == "Windows":
        return os.path.join(os.environ["LOCALAPPDATA"], "Ollama", "server.log")
    else:
        return None

def get_new_logs(last_position):
    """Fetches new log entries from the Ollama server log file."""
    log_file = get_log_file_path()
    if not log_file:
        return "", 0

    try:
        current_size = os.path.getsize(log_file)
        if current_size > last_position:
            with open(log_file, "r") as f:
                f.seek(last_position)
                new_logs = f.read()
                return new_logs, current_size
        else:
            return "", last_position
    except FileNotFoundError:
        return "", 0

def view_last_logs():
    """Displays the last 1000 lines of the Ollama server log file."""
    logs = get_server_logs()
    logs = logs[-1000:]
    log_text = "".join(logs)
    st.text_area("Last 1000 Lines of Server Logs", value=log_text, height=400, key="last_logs_view")

def get_server_logs():
    """Fetches server logs from the local Ollama log file."""
    system = platform.system()
    if system == "Darwin":
        log_file = os.path.expanduser("~/.ollama/logs/server.log")
    elif system == "Linux":
        # Assuming systemd service
        st.info("Please check the systemd journal using 'journalctl -u ollama' for logs.")
        return []
    elif system == "Windows":
        log_file = os.path.join(os.environ["LOCALAPPDATA"], "Ollama", "server.log")
    else:
        st.warning("Unsupported operating system. Unable to fetch server logs.")
        return []

    try:
        with open(log_file, "r") as f:
            logs = f.readlines()
        return logs
    except FileNotFoundError:
        st.warning(f"Server log file not found: {log_file}")
        return []

def get_resource_usage():
    """Fetches resource usage data from the Ollama API (placeholder)."""
    st.info("Real-time resource usage monitoring is not yet supported by the Ollama API.")
    return {}

@trace_llm_call(name="embedding_generation")
def generate_embeddings(model, text):
    """Generates embeddings for the given text using the specified model with comprehensive observability."""
    start_time = time.time()
    operation_id = f"embedding_{int(start_time * 1000)}"
    
    # Enhanced metadata for comprehensive observability
    if OBSERVABILITY_AVAILABLE:
        provider = "groq" if model in get_groq_models() else "openai" if model in get_openai_models() else "ollama"
        metadata = {
            "operation_id": operation_id,
            "model": model,
            "operation_type": "embedding",
            "text_length": len(text),
            "provider": provider,
            "word_count": len(text.split()),
            "character_count": len(text),
            "timestamp": time.time(),
            "session_id": getattr(st.session_state, 'session_id', 'unknown'),
            "user_id": getattr(st.session_state, 'user_id', 'anonymous')
        }
        add_trace_metadata(metadata)
        
        # Log structured operation start
        logger.info(f"Starting embedding generation", extra={
            "operation_id": operation_id,
            "model": model,
            "provider": provider,
            "text_length": len(text),
            "word_count": len(text.split())
        })
    
    try:
        if model in get_groq_models():
            # Use Groq API for embedding
            from .groq_utils import get_local_embeddings
            embeddings = get_local_embeddings(text)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Log usage for embeddings
            try:
                from ollama_workbench.models.model_management import log_model_usage
                log_model_usage(
                    model_name=model,
                    tokens_generated=len(text.split()),  # Approximate token count
                    response_time=elapsed_time,
                    operation_type="embedding"
                )
            except (ImportError, Exception) as log_error:
                logger.debug(f"Could not log embedding usage: {str(log_error)}")
                
            return embeddings
        elif model in get_openai_models():
            # Use OpenAI API for embedding
            response = call_openai_api(model, prompt=[{"role": "user", "content": text}], use_chat=False)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Log usage for embeddings
            try:
                from ollama_workbench.models.model_management import log_model_usage
                log_model_usage(
                    model_name=model,
                    tokens_generated=len(text.split()),  # Approximate token count
                    response_time=elapsed_time,
                    operation_type="embedding"
                )
            except (ImportError, Exception) as log_error:
                logger.debug(f"Could not log embedding usage: {str(log_error)}")
                
            return response
        else:
            # Default to Ollama API for embedding
            client = get_ollama_client()
            
            if client:
                # Newer version with Client class
                response = client.embeddings(model=model, prompt=text)
                # v0.4.8+: EmbedResponse object — normalize for dict access
                response = _normalize_response(response)

                # Calculate elapsed time
                elapsed_time = time.time() - start_time

                # Log usage for embeddings
                try:
                    from ollama_workbench.models.model_management import log_model_usage, log_model_performance
                    prompt_eval_count = response.get("prompt_eval_count", len(text.split()))
                    log_model_usage(
                        model_name=model,
                        tokens_generated=prompt_eval_count,
                        response_time=elapsed_time,
                        operation_type="embedding"
                    )

                    # Log performance if duration is available
                    if response.get("total_duration"):
                        total_duration = response.get("total_duration")
                        tokens_per_second = prompt_eval_count / (total_duration / (10**9)) if total_duration > 0 else 0
                        log_model_performance(
                            model_name=model,
                            prompt_text=text[:100] if len(text) > 100 else text,
                            tokens_per_second=tokens_per_second,
                            latency=elapsed_time,
                            temperature=0.0,  # Not applicable for embeddings
                            max_tokens=0      # Not applicable for embeddings
                        )
                except (ImportError, Exception) as log_error:
                    logger.debug(f"Could not log embedding usage: {str(log_error)}")

                return response.get("embedding", []), response.get("total_duration"), response.get("load_duration"), response.get("prompt_eval_count", 0)
            else:
                # Older version fallback - use module level functions
                try:
                    if hasattr(ollama, 'embeddings'):
                        response = ollama.embeddings(model=model, prompt=text)
                        # v0.4.8+: EmbedResponse object — normalize for dict access
                        response = _normalize_response(response)

                        # Calculate elapsed time
                        elapsed_time = time.time() - start_time

                        # Log usage for embeddings
                        try:
                            from ollama_workbench.models.model_management import log_model_usage, log_model_performance
                            prompt_eval_count = response.get("prompt_eval_count", len(text.split()))
                            log_model_usage(
                                model_name=model,
                                tokens_generated=prompt_eval_count,
                                response_time=elapsed_time,
                                operation_type="embedding"
                            )

                            # Log performance if duration is available
                            if response.get("total_duration"):
                                total_duration = response.get("total_duration")
                                tokens_per_second = prompt_eval_count / (total_duration / (10**9)) if total_duration > 0 else 0
                                log_model_performance(
                                    model_name=model,
                                    prompt_text=text[:100] if len(text) > 100 else text,
                                    tokens_per_second=tokens_per_second,
                                    latency=elapsed_time,
                                    temperature=0.0,  # Not applicable for embeddings
                                    max_tokens=0      # Not applicable for embeddings
                                )
                        except (ImportError, Exception) as log_error:
                            logger.debug(f"Could not log embedding usage: {str(log_error)}")

                        return response.get("embedding", []), response.get("total_duration"), response.get("load_duration"), response.get("prompt_eval_count", 0)
                    else:
                        # Last resort: fall back to direct API call
                        api_url = f"{get_ollama_url()}/embeddings"
                        response = requests.post(api_url, json={"model": model, "prompt": text})
                        response.raise_for_status()
                        embedding_data = response.json()
                        
                        # Calculate elapsed time
                        elapsed_time = time.time() - start_time
                        
                        # Log usage for embeddings
                        try:
                            from ollama_workbench.models.model_management import log_model_usage, log_model_performance
                            prompt_eval_count = embedding_data.get("prompt_eval_count", len(text.split()))
                            log_model_usage(
                                model_name=model,
                                tokens_generated=prompt_eval_count,
                                response_time=elapsed_time,
                                operation_type="embedding"
                            )
                            
                            # Log performance if duration is available
                            if embedding_data.get("total_duration"):
                                total_duration = embedding_data.get("total_duration")
                                tokens_per_second = prompt_eval_count / (total_duration / (10**9)) if total_duration > 0 else 0
                                log_model_performance(
                                    model_name=model,
                                    prompt_text=text[:100] if len(text) > 100 else text,
                                    tokens_per_second=tokens_per_second,
                                    latency=elapsed_time,
                                    temperature=0.0,  # Not applicable for embeddings
                                    max_tokens=0      # Not applicable for embeddings
                                )
                        except (ImportError, Exception) as log_error:
                            logger.debug(f"Could not log embedding usage: {str(log_error)}")
                        
                        return embedding_data["embedding"], embedding_data.get("total_duration"), embedding_data.get("load_duration"), embedding_data.get("prompt_eval_count", 0)
                except Exception as inner_e:
                    # If module level functions fail, fall back to direct API call
                    logger.warning(f"Module level fallback failed: {str(inner_e)}, using direct API call")
                    api_url = f"{get_ollama_url()}/embeddings"
                    response = requests.post(api_url, json={"model": model, "prompt": text})
                    response.raise_for_status()
                    embedding_data = response.json()
                    
                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time
                    
                    # Log usage for embeddings
                    try:
                        from ollama_workbench.models.model_management import log_model_usage, log_model_performance
                        prompt_eval_count = embedding_data.get("prompt_eval_count", len(text.split()))
                        log_model_usage(
                            model_name=model,
                            tokens_generated=prompt_eval_count,
                            response_time=elapsed_time,
                            operation_type="embedding"
                        )
                        
                        # Log performance if duration is available
                        if embedding_data.get("total_duration"):
                            total_duration = embedding_data.get("total_duration")
                            tokens_per_second = prompt_eval_count / (total_duration / (10**9)) if total_duration > 0 else 0
                            log_model_performance(
                                model_name=model,
                                prompt_text=text[:100] if len(text) > 100 else text,
                                tokens_per_second=tokens_per_second,
                                latency=elapsed_time,
                                temperature=0.0,  # Not applicable for embeddings
                                max_tokens=0      # Not applicable for embeddings
                            )
                    except (ImportError, Exception) as log_error:
                        logger.debug(f"Could not log embedding usage: {str(log_error)}")
                    
                    return embedding_data["embedding"], embedding_data.get("total_duration"), embedding_data.get("load_duration"), embedding_data.get("prompt_eval_count", 0)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error generating embeddings: {e}")
        st.error(f"Error generating embeddings: {e}")
        return None, None, None, None
    except Exception as e:
        # Enhanced error tracking with comprehensive context
        elapsed_time = time.time() - start_time
        error_context = {
            "operation_id": operation_id,
            "model": model,
            "text_length": len(text),
            "word_count": len(text.split()),
            "operation_type": "embedding",
            "provider": "groq" if model in get_groq_models() else "openai" if model in get_openai_models() else "ollama",
            "elapsed_time": elapsed_time,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "timestamp": time.time()
        }
        
        if OBSERVABILITY_AVAILABLE:
            log_trace_error(e, error_context)
        
        # Structured error logging
        logger.error(f"Embedding generation failed", extra={
            "operation_id": operation_id,
            "model": model,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "elapsed_time": elapsed_time,
            "text_length": len(text)
        })
        
        # Log error patterns for analysis
        try:
            from ollama_workbench.models.model_management import log_model_error
            log_model_error(
                model_name=model,
                error_type=type(e).__name__,
                error_message=str(e),
                operation_type="embedding",
                context=error_context
            )
        except (ImportError, Exception) as log_error:
            logger.debug(f"Could not log embedding error: {str(log_error)}")
        
        st.error(f"Unexpected error generating embeddings: {e}")
        return None, None, None, None

def get_local_models():
    """Gets locally available Ollama models.
    
    Returns:
        list: A list of dictionaries with 'name' key for each model
    """
    model_names = get_available_models()
    return [{"name": name} for name in model_names]
    
def log_model_stats(model_name, tokens_generated, response_time, operation_type="generate", **kwargs):
    """
    Enhanced utility function to manually log comprehensive model usage statistics from anywhere in the application.
    This is helpful for tracking usage in custom implementations or third-party integrations.
    
    Args:
        model_name (str): Name of the model being used
        tokens_generated (int): Number of tokens generated in the operation
        response_time (float): Time taken for the operation in seconds
        operation_type (str): Type of operation (generate, chat, embedding, vision)
        **kwargs: Additional metadata (operation_id, user_id, session_id, etc.)
    
    Returns:
        bool: True if logging was successful, False otherwise
    """
    operation_id = kwargs.get('operation_id', f"manual_{int(time.time() * 1000)}")
    
    try:
        # Log to model management system
        from ollama_workbench.models.model_management import log_model_usage
        log_model_usage(
            model_name=model_name,
            tokens_generated=tokens_generated,
            response_time=response_time,
            operation_type=operation_type
        )
        
        # Enhanced structured logging
        try:
            logger.info("Model usage logged", extra={
                "operation_id": operation_id,
                "model_name": model_name,
                "tokens_generated": tokens_generated,
                "response_time": response_time,
                "operation_type": operation_type,
                "tokens_per_second": tokens_generated / response_time if response_time > 0 else 0,
                "user_id": kwargs.get('user_id', 'unknown'),
                "session_id": kwargs.get('session_id', 'unknown')
            })
        except (KeyError, Exception):
            pass  # Logging failures should not affect functionality
        
        # Log to observability system if available
        if OBSERVABILITY_AVAILABLE:
            log_performance_metrics(
                response_time=response_time,
                token_count=tokens_generated,
                model_name=model_name,
                operation_type=operation_type,
                operation_id=operation_id,
                **kwargs
            )
        
        return True
    except (ImportError, Exception) as e:
        logger.debug(f"Could not log model usage: {str(e)}")
        return False

def get_operation_metrics(operation_id=None, model_name=None, time_range_hours=24):
    """
    Retrieve comprehensive operation metrics for analysis and monitoring.
    
    Args:
        operation_id (str, optional): Specific operation ID to retrieve
        model_name (str, optional): Filter by model name
        time_range_hours (int): Time range in hours to look back
    
    Returns:
        dict: Metrics data including performance stats, error rates, usage patterns
    """
    try:
        from ollama_workbench.models.model_management import get_model_metrics
        return get_model_metrics(
            operation_id=operation_id,
            model_name=model_name,
            time_range_hours=time_range_hours
        )
    except (ImportError, Exception) as e:
        logger.debug(f"Could not retrieve operation metrics: {str(e)}")
        return {}

def monitor_model_health(model_name, alert_thresholds=None):
    """
    Monitor model health and performance, triggering alerts when thresholds are exceeded.
    
    Args:
        model_name (str): Name of the model to monitor
        alert_thresholds (dict, optional): Custom thresholds for alerts
    
    Returns:
        dict: Health status and recommendations
    """
    if alert_thresholds is None:
        alert_thresholds = PERFORMANCE_THRESHOLDS
    
    try:
        # Get recent metrics
        metrics = get_operation_metrics(model_name=model_name, time_range_hours=1)
        
        health_status = {
            "model_name": model_name,
            "timestamp": time.time(),
            "status": "healthy",
            "alerts": [],
            "recommendations": [],
            "metrics_summary": metrics
        }
        
        # Analyze performance patterns
        if metrics:
            avg_response_time = metrics.get("avg_response_time", 0)
            avg_tokens_per_second = metrics.get("avg_tokens_per_second", 0)
            error_rate = metrics.get("error_rate", 0)
            
            # Check response time alerts
            if avg_response_time > alert_thresholds["very_slow_response_time"]:
                health_status["status"] = "critical"
                health_status["alerts"].append(f"Very slow response time: {avg_response_time:.2f}s")
                health_status["recommendations"].append("Consider switching to a faster model or optimizing prompts")
            elif avg_response_time > alert_thresholds["slow_response_time"]:
                health_status["status"] = "warning"
                health_status["alerts"].append(f"Slow response time: {avg_response_time:.2f}s")
                health_status["recommendations"].append("Monitor response times and consider optimization")
            
            # Check throughput alerts
            if avg_tokens_per_second < alert_thresholds["very_low_tokens_per_second"]:
                health_status["status"] = "critical"
                health_status["alerts"].append(f"Very low throughput: {avg_tokens_per_second:.2f} tokens/sec")
                health_status["recommendations"].append("Check system resources and model configuration")
            elif avg_tokens_per_second < alert_thresholds["low_tokens_per_second"]:
                if health_status["status"] == "healthy":
                    health_status["status"] = "warning"
                health_status["alerts"].append(f"Low throughput: {avg_tokens_per_second:.2f} tokens/sec")
                health_status["recommendations"].append("Monitor system performance")
            
            # Check error rate
            if error_rate > 0.1:  # 10% error rate
                health_status["status"] = "critical"
                health_status["alerts"].append(f"High error rate: {error_rate:.1%}")
                health_status["recommendations"].append("Investigate error patterns and model availability")
            elif error_rate > 0.05:  # 5% error rate
                if health_status["status"] == "healthy":
                    health_status["status"] = "warning"
                health_status["alerts"].append(f"Elevated error rate: {error_rate:.1%}")
                health_status["recommendations"].append("Monitor error logs for patterns")
        
        # Log health status
        logger.info(f"Model health check completed", extra={
            "model_name": model_name,
            "health_status": health_status["status"],
            "alert_count": len(health_status["alerts"]),
            "timestamp": health_status["timestamp"]
        })
        
        return health_status
        
    except Exception as e:
        logger.error(f"Model health monitoring failed", extra={
            "model_name": model_name,
            "error_type": type(e).__name__,
            "error_message": str(e)
        })
        return {
            "model_name": model_name,
            "timestamp": time.time(),
            "status": "unknown",
            "error": str(e)
        }

def get_dynamic_model_default():
    """Get the first available model as a dynamic default, with fallbacks."""
    try:
        # Get all available models
        all_models = get_all_models()
        
        if all_models:
            # Return the first available model
            default_model = all_models[0]
            logger.info(f"Dynamic default model selected: {default_model}")
            return default_model
        else:
            logger.warning("No models available - returning None")
            return None
    except Exception as e:
        logger.error(f"Error getting dynamic default model: {e}")
        return None

def validate_model_exists(model_name):
    """Check if a specific model exists in available models."""
    if not model_name:
        return False
        
    try:
        all_models = get_all_models()
        exists = model_name in all_models
        logger.info(f"Model '{model_name}' {'exists' if exists else 'does not exist'}")
        return exists
    except Exception as e:
        logger.error(f"Error validating model {model_name}: {e}")
        return False

def get_available_models_with_fallback():
    """Get available models with fallback message if none available."""
    try:
        models = get_available_models()
        if not models:
            return ["No models available - Please install Ollama models first"]
        return models
    except Exception as e:
        logger.error(f"Error getting models with fallback: {e}")
        return ["Error loading models - Check Ollama installation"]

def get_all_models():
    """Gets all available models, including Ollama, Groq, OpenAI, and Mistral with enhanced logging."""
    start_time = time.time()
    
    try:
        # Get Ollama models - these are already strings from get_available_models()
        ollama_model_names = get_available_models()
        
        # Combine all models - remove the headers that were causing selection issues
        # Just return the actual model names without the category headers
        groq_models = get_groq_models()
        openai_models = get_openai_models()
        mistral_models = get_mistral_models()
        all_models = ollama_model_names + groq_models + openai_models + mistral_models

        elapsed_time = time.time() - start_time

        # Log model discovery metrics
        logger.info(f"Model discovery completed", extra={
            "ollama_models_count": len(ollama_model_names),
            "groq_models_count": len(groq_models),
            "openai_models_count": len(openai_models),
            "mistral_models_count": len(mistral_models),
            "total_models_count": len(all_models),
            "discovery_time": elapsed_time
        })

        return all_models

    except Exception as e:
        logger.error(f"Model discovery failed", extra={
            "error_type": type(e).__name__,
            "error_message": str(e),
            "elapsed_time": time.time() - start_time
        })
        # Return at least the static models if Ollama discovery fails
        return get_groq_models() + get_openai_models() + get_mistral_models()

def create_observability_context(operation_type, model_name, **kwargs):
    """
    Create a standardized observability context for operations.
    
    Args:
        operation_type (str): Type of operation (generate, embedding, chat, etc.)
        model_name (str): Name of the model being used
        **kwargs: Additional context parameters
    
    Returns:
        dict: Standardized context dictionary
    """
    timestamp = time.time()
    operation_id = kwargs.get('operation_id', f"{operation_type}_{int(timestamp * 1000)}")
    
    context = {
        "operation_id": operation_id,
        "operation_type": operation_type,
        "model_name": model_name,
        "provider": "groq" if model_name in get_groq_models() else "openai" if model_name in get_openai_models() else "mistral" if model_name in get_mistral_models() else "ollama",
        "timestamp": timestamp,
        "session_id": getattr(st.session_state, 'session_id', kwargs.get('session_id', 'unknown')),
        "user_id": getattr(st.session_state, 'user_id', kwargs.get('user_id', 'anonymous')),
        "observability_enabled": OBSERVABILITY_AVAILABLE
    }
    
    # Add any additional context
    context.update(kwargs)
    
    return context
