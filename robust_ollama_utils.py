"""
Robust Ollama Utilities Module

This module provides all the functionality of the original ollama_utils.py
but with enhanced error handling, dependency management, and compatibility fixes
for NumPy and PyTorch.

CHECKPOINT: This is the robust version with complete functionality.
"""

import os
import json
import logging
import requests
import time
import importlib.util
import subprocess
import platform

# Set up logging with detailed checkpoints
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.info("CHECKPOINT: Loading robust_ollama_utils module")

# Safe imports with fallbacks
def safe_import(module_name, fallback=None):
    """Safely import a module with fallback value if not available."""
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(f"CHECKPOINT: Could not import {module_name}: {str(e)}")
        return fallback

# Import dependencies with fallbacks
st = safe_import("streamlit")
psutil = safe_import("psutil")

# Try to import model categorization
try:
    from model_categorization import (
        categorize_models, 
        create_categorized_model_ui
    )
    logger.info("CHECKPOINT: Successfully imported model categorization module")
    MODEL_CATEGORIZATION_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"CHECKPOINT: Model categorization module not available: {str(e)}")
    MODEL_CATEGORIZATION_AVAILABLE = False

# Handle NumPy compatibility issues
try:
    import numpy as np
    logger.info("CHECKPOINT: Successfully imported numpy")
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"CHECKPOINT: NumPy import error: {str(e)}")
    # Create minimal numpy replacement for essential functions
    class MinimalNumpy:
        def array(self, data):
            return data
    np = MinimalNumpy()
    logger.info("CHECKPOINT: Using minimal NumPy replacement")

# Handle ollama package with version detection
try:
    import ollama
    logger.info("CHECKPOINT: Successfully imported ollama package")
    OLLAMA_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"CHECKPOINT: Ollama package not available: {str(e)}")
    OLLAMA_AVAILABLE = False

# Try to import external model providers
try:
    from openai_utils import OPENAI_MODELS
    logger.info("CHECKPOINT: Successfully imported OpenAI utilities")
    OPENAI_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"CHECKPOINT: OpenAI utilities not available: {str(e)}")
    OPENAI_AVAILABLE = False
    OPENAI_MODELS = []

try:
    from groq_utils import GROQ_MODELS
    logger.info("CHECKPOINT: Successfully imported Groq utilities")
    GROQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"CHECKPOINT: Groq utilities not available: {str(e)}")
    GROQ_AVAILABLE = False
    GROQ_MODELS = []

try:
    from mistral_utils import MISTRAL_MODELS
    logger.info("CHECKPOINT: Successfully imported Mistral utilities")
    MISTRAL_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"CHECKPOINT: Mistral utilities not available: {str(e)}")
    MISTRAL_AVAILABLE = False
    MISTRAL_MODELS = []

# Try to import configuration
try:
    from config import CONFIG, get_config, update_config, set_api_key, get_api_key
    logger.info("CHECKPOINT: Successfully imported configuration module")
    CONFIG_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"CHECKPOINT: Configuration module not available: {str(e)}")
    CONFIG_AVAILABLE = False
    # Fallback configuration
    CONFIG = {
        "OLLAMA_HOST": "http://localhost:11434",
        "DEFAULT_MODEL": "llama3",
        "DEBUG": True
    }
    
    def get_config():
        """Get the current configuration."""
        return CONFIG

    def update_config(new_config):
        """Update the configuration with new values."""
        global CONFIG
        CONFIG.update(new_config)
        return CONFIG

    def set_api_key(service, key):
        """Set an API key for a service."""
        CONFIG[f"{service.upper()}_API_KEY"] = key

    def get_api_key(service):
        """Get an API key for a service."""
        return CONFIG.get(f"{service.upper()}_API_KEY", "")

# Legacy files - will be migrated to central config
API_KEYS_FILE = "api_keys.json"
MODEL_SETTINGS_FILE = "model_settings.json"

def load_api_keys():
    """Loads API keys from the JSON file."""
    logger.info("CHECKPOINT: Loading API keys")
    try:
        if os.path.exists(API_KEYS_FILE):
            with open(API_KEYS_FILE, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"CHECKPOINT: Error loading API keys: {str(e)}")
        return {}

def save_api_keys(api_keys):
    """Saves API keys to the JSON file."""
    try:
        with open(API_KEYS_FILE, "w") as f:
            json.dump(api_keys, f, indent=4)
        logger.info("CHECKPOINT: API keys saved successfully")
        return True
    except Exception as e:
        logger.error(f"CHECKPOINT: Error saving API keys: {str(e)}")
        return False

def load_model_settings():
    """Loads model settings from the JSON file."""
    try:
        if os.path.exists(MODEL_SETTINGS_FILE):
            with open(MODEL_SETTINGS_FILE, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"CHECKPOINT: Error loading model settings: {str(e)}")
        return {}

def save_model_settings(settings):
    """Saves model settings to the JSON file."""
    try:
        with open(MODEL_SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"CHECKPOINT: Error saving model settings: {str(e)}")
        return False

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
    if not OLLAMA_AVAILABLE:
        logger.warning("CHECKPOINT: ollama package not available, cannot create client")
        return None
        
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
                    logger.info("CHECKPOINT: Successfully created ollama client with new API")
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
                        logger.info("CHECKPOINT: Successfully configured ollama with module-level functions")
                        return None        # Return None to indicate module-level functions should be used
                except Exception as module_error:
                    logger.warning(f"Module-level functions failed: {module_error}")
                    
            # If we get here, both approaches failed
            logger.error("Both client and module-level functions failed")
            return None
                
        except Exception as e:
            logger.error(f"Error configuring ollama client: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting ollama client: {e}")
        return None

# Dynamic URL based on config
OLLAMA_URL = get_ollama_url()

def get_ollama_resource_usage():
    """Gets Ollama server resource usage."""
    logger.info("CHECKPOINT: Getting Ollama resource usage")
    
    if not psutil:
        logger.warning("CHECKPOINT: psutil not available, cannot get resource usage")
        return {"error": "psutil not available"}
    
    try:
        # Try to find the Ollama process
        ollama_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Check if this is an Ollama process
                if proc.info['name'] and 'ollama' in proc.info['name'].lower():
                    ollama_processes.append(proc)
                elif proc.info['cmdline'] and any('ollama' in cmd.lower() for cmd in proc.info['cmdline'] if cmd):
                    ollama_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        if not ollama_processes:
            logger.warning("CHECKPOINT: No Ollama processes found")
            return {"error": "No Ollama processes found"}
        
        # Get resource usage for the first Ollama process
        process = ollama_processes[0]
        cpu_percent = process.cpu_percent(interval=0.5)
        memory_info = process.memory_info()
        
        # Try to get the Ollama API status
        api_status = "Unknown"
        try:
            api_response = requests.get(f"{get_ollama_url()}/api/health", timeout=2)
            api_status = "Online" if api_response.status_code == 200 else f"Error: {api_response.status_code}"
        except Exception as e:
            api_status = f"Error: {str(e)}"
        
        return {
            "pid": process.pid,
            "cpu_percent": cpu_percent,
            "memory_mb": memory_info.rss / (1024 * 1024),  # Convert to MB
            "api_status": api_status
        }
    except Exception as e:
        logger.error(f"CHECKPOINT: Error getting resource usage: {str(e)}")
        return {"error": str(e)}

def get_available_models():
    """
    Get a list of available Ollama models.
    
    Returns:
        list: A list of model names (strings)
    """
    logger.info("CHECKPOINT: Getting available Ollama models")
    
    # Try using the client first if available
    if OLLAMA_AVAILABLE:
        try:
            client = get_ollama_client()
            if client and hasattr(client, "list"):
                # Use client.list() for newer versions
                response = client.list()
                if isinstance(response, dict) and "models" in response:
                    models = [model["name"] for model in response["models"]]
                elif isinstance(response, list):
                    models = [model["name"] for model in response]
                logger.info(f"CHECKPOINT: Found {len(models)} models using client API")
                if models:  # Only return if we actually found models
                    return models
            elif hasattr(ollama, "list"):
                # Use module-level function for older versions
                response = ollama.list()
                if isinstance(response, dict) and "models" in response:
                    models = [model["name"] for model in response["models"]]
                elif isinstance(response, list):
                    models = [model["name"] for model in response]
                logger.info(f"CHECKPOINT: Found {len(models)} models using module API")
                if models:  # Only return if we actually found models
                    return models
        except Exception as e:
            logger.warning(f"CHECKPOINT: Error getting models with ollama package: {str(e)}")
            # Continue to fallback method
    
    # Fallback to direct API call
    try:
        # Make sure we have the correct URL with /api prefix if needed
        base_url = get_ollama_url()
        # Try both /api/tags and /tags endpoints
        for endpoint in [f"{base_url}/api/tags", f"{base_url}/tags"]:
            try:
                logger.info(f"CHECKPOINT: Trying to fetch models from {endpoint}")
                response = requests.get(endpoint, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if "models" in data:
                        models = [model["name"] for model in data["models"]]
                    else:
                        models = [model["name"] for model in data.get("models", [])]
                    logger.info(f"CHECKPOINT: Found {len(models)} models using direct API call to {endpoint}")
                    if models:  # Only return if we actually found models
                        return models
            except Exception as e:
                logger.warning(f"CHECKPOINT: Error with endpoint {endpoint}: {str(e)}")
                continue
    except Exception as e:
        logger.error(f"CHECKPOINT: Error getting models with direct API: {str(e)}")
    
    # Last resort: try to get models from local file system
    try:
        # Check multiple possible locations
        possible_dirs = [
            os.path.expanduser("~/.ollama/models"),
            "/usr/share/ollama/models",
            "C:\\Program Files\\Ollama\\models",
            os.path.join(os.path.expanduser("~"), "AppData", "Local", "Ollama", "models")
        ]
        
        for ollama_dir in possible_dirs:
            if os.path.exists(ollama_dir):
                logger.info(f"CHECKPOINT: Checking for models in {ollama_dir}")
                # List directories in the models folder
                for item in os.listdir(ollama_dir):
                    if os.path.isdir(os.path.join(ollama_dir, item)) and not item.startswith('.'):
                        models.append(item)
                if models:
                    logger.info(f"CHECKPOINT: Found {len(models)} models from file system at {ollama_dir}")
                    return models
    except Exception as e:
        logger.error(f"CHECKPOINT: Error getting models from file system: {str(e)}")
    
    # If still no models, provide some default suggestions
    if not models:
        logger.warning("CHECKPOINT: No models found, using default suggestions")
        models = ["llama3", "mistral", "gemma", "llama2"]
    
    return models

def call_ollama_endpoint(model, prompt=None, image=None, temperature=0.5, max_tokens=150, 
                        presence_penalty=0.0, frequency_penalty=0.0, context=None, 
                        tools=None, episodic_memory=None, format=None, capture_metrics=False):
    """
    Call the Ollama API to generate text or process images
    
    Args:
        model: The model to use
        prompt: The text prompt
        image: Optional image data (base64 encoded)
        temperature: Temperature parameter (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
        presence_penalty: Presence penalty parameter
        frequency_penalty: Frequency penalty parameter
        context: Optional context for continued conversations
        tools: Optional tools/functions for the model to use
        episodic_memory: Optional memory context
        format: Optional output format specifier
        capture_metrics: Whether to capture detailed performance metrics
        
    Returns:
        Tuple of (response_text, total_duration, load_duration, eval_count)
    """
    logger.info(f"CHECKPOINT: Calling Ollama endpoint with model {model}")
    
    # Record start time for metrics
    start_time = time.time()
    
    # Handle image input
    if image:
        logger.info("CHECKPOINT: Image input detected, using multimodal endpoint")
        try:
            # Try using the client for multimodal
            client = get_ollama_client()
            if client and hasattr(client, "generate"):
                response = client.generate(
                    model=model,
                    prompt=prompt or "",
                    images=[image] if image else None,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                )
                elapsed_time = time.time() - start_time
                logger.info(f"CHECKPOINT: Multimodal response generated in {elapsed_time:.2f}s")
                return response.get("response", ""), elapsed_time, None, None
        except Exception as mm_error:
            logger.warning(f"CHECKPOINT: Error with multimodal request: {str(mm_error)}")
    
    # Try using the ollama client
    if OLLAMA_AVAILABLE:
        try:
            client = get_ollama_client()
            if client and hasattr(client, "generate"):
                # Prepare options
                options = {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
                
                if context:
                    options["context"] = context
                
                # Call the API
                response = client.generate(
                    model=model,
                    prompt=prompt,
                    options=options
                )
                
                # Extract response
                response_text = response.get("response", "")
                total_duration = response.get("total_duration", None)
                load_duration = response.get("load_duration", None)
                eval_count = response.get("eval_count", None)
                
                elapsed_time = time.time() - start_time
                logger.info(f"CHECKPOINT: Successfully generated response with client (tokens: {eval_count}, time: {elapsed_time:.2f}s)")
                
                # Log detailed metrics if requested
                if capture_metrics and st is not None:
                    if "metrics" not in st.session_state:
                        st.session_state.metrics = []
                    st.session_state.metrics.append({
                        "model": model,
                        "tokens": eval_count,
                        "time": elapsed_time,
                        "tokens_per_second": eval_count / elapsed_time if elapsed_time > 0 else 0
                    })
                
                return response_text, total_duration, load_duration, eval_count
        except Exception as client_error:
            logger.warning(f"CHECKPOINT: Error generating with client: {str(client_error)}")
    
    # Fallback to direct API call
    try:
        # Prepare request data
        request_data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty
        }
        
        if context:
            request_data["context"] = context
        
        if tools:
            request_data["tools"] = tools
            
        if format:
            request_data["format"] = format
        
        # Make the API call
        response = requests.post(
            f"{get_ollama_url()}/generate",
            json=request_data
        )
        
        if response.status_code == 200:
            data = response.json()
            response_text = data.get("response", "")
            total_duration = data.get("total_duration", None)
            load_duration = data.get("load_duration", None)
            eval_count = data.get("eval_count", None)
            
            elapsed_time = time.time() - start_time
            logger.info(f"CHECKPOINT: Successfully generated response with API (tokens: {eval_count}, time: {elapsed_time:.2f}s)")
            
            # Log metrics if requested
            if capture_metrics and st is not None:
                if "metrics" not in st.session_state:
                    st.session_state.metrics = []
                st.session_state.metrics.append({
                    "model": model,
                    "tokens": eval_count,
                    "time": elapsed_time,
                    "tokens_per_second": eval_count / elapsed_time if elapsed_time > 0 else 0
                })
            
            return response_text, total_duration, load_duration, eval_count
        else:
            logger.error(f"CHECKPOINT: API error: {response.status_code} - {response.text}")
            return f"Error: {response.status_code} - {response.text}", None, None, None
    except Exception as api_error:
        logger.error(f"CHECKPOINT: Error with API request: {str(api_error)}")
        return f"Error: {str(api_error)}", None, None, None


# ============================================================================
# Model Management and Integration Functions
# ============================================================================

def get_all_models():
    """Gets all available models, including Ollama, Groq, OpenAI, and Mistral."""
    logger.info("CHECKPOINT: Getting all available models")
    
    # Get Ollama models
    ollama_model_names = get_available_models()
    logger.info(f"CHECKPOINT: Found {len(ollama_model_names)} Ollama models")
    
    # Combine all models from different providers
    all_models = ollama_model_names
    
    # Add other providers if available
    if GROQ_AVAILABLE and GROQ_MODELS:
        logger.info(f"CHECKPOINT: Adding {len(GROQ_MODELS)} Groq models")
        all_models.extend(GROQ_MODELS)
    
    if OPENAI_AVAILABLE and OPENAI_MODELS:
        logger.info(f"CHECKPOINT: Adding {len(OPENAI_MODELS)} OpenAI models")
        all_models.extend(OPENAI_MODELS)
        
    if MISTRAL_AVAILABLE and MISTRAL_MODELS:
        logger.info(f"CHECKPOINT: Adding {len(MISTRAL_MODELS)} Mistral models")
        all_models.extend(MISTRAL_MODELS)
    
    # If we have no models at all, add some default suggestions
    if not all_models:
        logger.warning("CHECKPOINT: No models found from any provider, adding default suggestions")
        all_models = ["llama3", "mistral", "gemma", "llama2"]
    
    logger.info(f"CHECKPOINT: Found {len(all_models)} total models across all providers")
    return all_models


def pull_model(model_name):
    """Pull a model from Ollama."""
    logger.info(f"CHECKPOINT: Pulling model {model_name}")
    
    if not model_name:
        return {"error": "No model name provided"}
    
    # First try using the client if available
    if OLLAMA_AVAILABLE:
        try:
            client = get_ollama_client()
            if client and hasattr(client, "pull"):
                # Use client.pull() for newer versions
                logger.info(f"CHECKPOINT: Pulling model {model_name} using client API")
                client.pull(model_name)
                return {"success": True, "message": f"Model {model_name} pulled successfully"}
            elif hasattr(ollama, "pull"):
                # Use module-level function for older versions
                logger.info(f"CHECKPOINT: Pulling model {model_name} using module API")
                ollama.pull(model_name)
                return {"success": True, "message": f"Model {model_name} pulled successfully"}
        except Exception as e:
            logger.warning(f"CHECKPOINT: Error pulling model with ollama package: {str(e)}")
            # Continue to fallback method
    
    # Fallback to subprocess
    try:
        logger.info(f"CHECKPOINT: Pulling model {model_name} using subprocess")
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            check=True
        )
        return {"success": True, "message": result.stdout}
    except subprocess.CalledProcessError as e:
        error_message = e.stderr or str(e)
        logger.error(f"CHECKPOINT: Error pulling model with subprocess: {error_message}")
        return {"success": False, "error": error_message}
    except Exception as e:
        logger.error(f"CHECKPOINT: Error pulling model with subprocess: {str(e)}")
        return {"success": False, "error": str(e)}


def remove_model(model_name):
    """Remove a model from Ollama."""
    logger.info(f"CHECKPOINT: Removing model {model_name}")
    
    if not model_name:
        return {"error": "No model name provided"}
    
    # Try using the client first
    if OLLAMA_AVAILABLE:
        try:
            client = get_ollama_client()
            if client and hasattr(client, "delete"):
                logger.info(f"CHECKPOINT: Removing model {model_name} with client API")
                # Use client.delete() for newer versions
                response = client.delete(model_name)
                return {"success": True, "message": f"Model {model_name} removed successfully"}
        except Exception as e:
            logger.warning(f"CHECKPOINT: Error removing model with client: {str(e)}")
            # Continue to fallback method
    
    # Fallback to subprocess
    try:
        logger.info(f"CHECKPOINT: Removing model {model_name} with subprocess")
        process = subprocess.Popen(
            ["ollama", "rm", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logger.info(f"CHECKPOINT: Model {model_name} removed successfully")
            return {"success": True, "message": f"Model {model_name} removed successfully", "output": stdout}
        else:
            logger.error(f"CHECKPOINT: Error removing model: {stderr}")
            return {"success": False, "error": stderr}
    except Exception as e:
        logger.error(f"CHECKPOINT: Error removing model with subprocess: {str(e)}")
        return {"success": False, "error": str(e)}


def show_model_info(model_name):
    """Show information about a model."""
    logger.info(f"CHECKPOINT: Getting info for model {model_name}")
    
    if not model_name:
        return {"error": "No model name provided"}
    
    # Try using the client first
    if OLLAMA_AVAILABLE:
        try:
            client = get_ollama_client()
            if client and hasattr(client, "show"):
                logger.info(f"CHECKPOINT: Getting info for model {model_name} with client API")
                # Use client.show() for newer versions
                response = client.show(model_name)
                return {"success": True, "info": response}
        except Exception as e:
            logger.warning(f"CHECKPOINT: Error getting model info with client: {str(e)}")
            # Continue to fallback method
    
    # Fallback to subprocess
    try:
        logger.info(f"CHECKPOINT: Getting info for model {model_name} with subprocess")
        process = subprocess.Popen(
            ["ollama", "show", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logger.info(f"CHECKPOINT: Got info for model {model_name}")
            return {"success": True, "info": stdout}
        else:
            logger.error(f"CHECKPOINT: Error getting model info: {stderr}")
            return {"success": False, "error": stderr}
    except Exception as e:
        logger.error(f"CHECKPOINT: Error getting model info with subprocess: {str(e)}")
        return {"success": False, "error": str(e)}


# ============================================================================
# Workspace Integration Functions
# ============================================================================

def save_ai_content_to_workspace(content):
    """
    Save AI-generated content to the workspace.
    This function is a bridge to the chat_workspace module.
    
    Args:
        content: The content to save
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("CHECKPOINT: Attempting to save content to workspace")
    if not content or not isinstance(content, str):
        logger.warning(f"CHECKPOINT: Invalid content type: {type(content)}")
        return False
        
    # Check if content contains code blocks
    if "```" not in content:
        logger.info("CHECKPOINT: No code blocks found in content, skipping workspace save")
        return True  # Not an error, just nothing to save
    
    try:
        # Import streamlit here when needed
        import streamlit as st
        
        # Import here to avoid circular imports
        from chat_workspace import save_ai_content_to_workspace as save_to_workspace
        
        # Call the actual implementation
        save_to_workspace(content)
        
        # Show a success notification if we're in a Streamlit context
        if hasattr(st, '_is_running') and st._is_running:
            st.toast("Code saved to Workspace", icon="✅")
            
        logger.info("CHECKPOINT: Successfully saved content to workspace")
        return True
    except ImportError as ie:
        # This is a more specific error for when the chat_workspace module is not available
        logger.warning(f"CHECKPOINT: Workspace module not available: {str(ie)}")
        return False
    except Exception as e:
        logger.error(f"CHECKPOINT: Error saving content to workspace: {str(e)}")
        logger.exception(e)
        return False


# ============================================================================
# Model Selection UI Functions
# ============================================================================

def get_model_categories(models):
    """Categorize models by provider and type."""
    logger.info("CHECKPOINT: Categorizing models")
    
    if MODEL_CATEGORIZATION_AVAILABLE:
        # Use the dedicated module if available
        return categorize_models(models)
    
    # Fallback implementation if module not available
    categories = {
        "ollama": [],
        "openai": [],
        "groq": [],
        "mistral": [],
        "anthropic": []
    }
    
    # Simple categorization based on name prefixes
    for model in models:
        if model.startswith("gpt-") or model.startswith("text-"):
            categories["openai"].append(model)
        elif model.startswith("llama2-") or model.startswith("mixtral-") or model.startswith("gemma-"):
            categories["groq"].append(model)
        elif model.startswith("mistral-"):
            categories["mistral"].append(model)
        elif model.startswith("claude-"):
            categories["anthropic"].append(model)
        else:
            categories["ollama"].append(model)
    
    return categories

def create_model_selection_ui(available_models, current_model=None):
    """Create a UI for selecting models with proper categorization."""
    logger.info("CHECKPOINT: Creating model selection UI")
    
    if not st:
        logger.warning("CHECKPOINT: Streamlit not available, cannot create UI")
        return current_model or (available_models[0] if available_models else "llama3"), False
    
    if MODEL_CATEGORIZATION_AVAILABLE:
        # Use the dedicated module if available
        return create_categorized_model_ui(available_models, current_model)
    
    # Fallback implementation
    st.subheader("Model Selection")
    
    # Ensure we have a valid current model
    if not current_model or current_model not in available_models:
        current_model = available_models[0] if available_models else "llama3"
    
    # Find the current model index
    try:
        model_index = available_models.index(current_model)
    except (ValueError, IndexError):
        model_index = 0
    
    # Create a form to prevent auto-rerun
    with st.form(key="model_selection_form"):
        model_choice = st.selectbox(
            "📦 Model:",
            available_models,
            index=model_index
        )
        
        # Form submit button
        submit_button = st.form_submit_button(label="Select Model")
        
        # Only update on form submission
        if submit_button and model_choice != current_model:
            # Log the change
            logger.info(f"CHECKPOINT: Model changed from '{current_model}' to '{model_choice}'")
            return model_choice, True
    
    return current_model, False

# ============================================================================
# Server Management Functions
# ============================================================================

def get_log_file_path():
    """Returns the platform-specific path to the Ollama server log file."""
    system = platform.system()
    if system == "Darwin":  # macOS
        return os.path.expanduser("~/Library/Logs/Ollama/ollama.log")
    elif system == "Linux":
        return "/var/log/ollama/ollama.log"
    elif system == "Windows":
        return os.path.join(os.environ.get("APPDATA", ""), "Ollama", "logs", "ollama.log")
    else:
        return None


def get_server_logs():
    """Fetches server logs from the local Ollama log file."""
    logger.info("CHECKPOINT: Getting Ollama server logs")
    try:
        log_path = get_log_file_path()
        if not log_path or not os.path.exists(log_path):
            logger.warning(f"CHECKPOINT: Log file not found at {log_path}")
            return "Log file not found"
            
        # Read the last 1000 lines of the log file
        with open(log_path, "r") as f:
            lines = f.readlines()
            return "".join(lines[-1000:]) if lines else "No logs found"
    except Exception as e:
        logger.error(f"CHECKPOINT: Error reading log file: {str(e)}")
        return f"Error reading log file: {str(e)}"


def start_server():
    """Starts the Ollama server."""
    logger.info("CHECKPOINT: Starting Ollama server")
    try:
        # Start the server and don't wait for it to complete
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return {"success": True, "message": "Ollama server started"}
    except Exception as e:
        logger.error(f"CHECKPOINT: Error starting Ollama server: {str(e)}")
        return {"success": False, "error": str(e)}


def stop_server():
    """Stops the Ollama server."""
    logger.info("CHECKPOINT: Stopping Ollama server")
    try:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"])
        else:
            subprocess.run(["pkill", "ollama"])
        return {"success": True, "message": "Ollama server stopped"}
    except Exception as e:
        logger.error(f"CHECKPOINT: Error stopping Ollama server: {str(e)}")
        return {"success": False, "error": str(e)}
