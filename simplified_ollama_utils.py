"""
Enhanced Simplified Ollama Utilities Module

This module provides essential functions for the Ollama Workbench to operate
without dependencies on other modules that might be missing, while still
supporting all core features including workspace integration.

CHECKPOINT: This is the enhanced version with workspace support.
"""

import os
import json
import logging
import requests
import psutil
import subprocess
import importlib.util
# Only import streamlit when needed in functions that use it
# This avoids the unused import warning

# Set up logging
logger = logging.getLogger(__name__)
logger.info("CHECKPOINT: Loading simplified_ollama_utils module")

# Check if ollama package is available
OLLAMA_AVAILABLE = importlib.util.find_spec("ollama") is not None
if OLLAMA_AVAILABLE:
    import ollama
    logger.info("CHECKPOINT: Successfully imported ollama package")
else:
    logger.warning("CHECKPOINT: ollama package not available")

# Configuration
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
        
        # Create client
        client = ollama.Client(host=host)
        
        # Test a simple API call to verify the client works
        try:
            _ = client.list()  # Try a simple API call
            logger.info("CHECKPOINT: Successfully created and tested ollama client")
            return client      # Only return client if it works
        except Exception as test_error:
            logger.warning(f"CHECKPOINT: Client created but test failed: {str(test_error)}")
            return None
            
    except Exception as e:
        logger.error(f"CHECKPOINT: Error creating ollama client: {str(e)}")
        return None

def get_ollama_resource_usage():
    """Gets Ollama server resource usage."""
    logger.info("CHECKPOINT: Getting Ollama resource usage")
    try:
        # Get CPU and memory usage
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # Check if GPU is available
        gpu_percent = None
        try:
            # Try to get GPU usage if available
            # This is a simplified approach and might not work on all systems
            if os.name == 'posix':  # Linux/Mac
                nvidia_smi = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
                if nvidia_smi.returncode == 0:
                    gpu_percent = float(nvidia_smi.stdout.strip())
        except Exception as gpu_error:
            logger.debug(f"CHECKPOINT: GPU info not available: {str(gpu_error)}")
        
        # Build result
        result = {
            "cpu": cpu_percent,
            "memory": memory_percent
        }
        
        if gpu_percent is not None:
            result["gpu"] = gpu_percent
            
        return result
    except Exception as e:
        logger.error(f"CHECKPOINT: Error getting resource usage: {str(e)}")
        return {"cpu": 0, "memory": 0}

def get_available_models():
    """
    Get a list of available Ollama models.
    
    Returns:
        list: A list of model names (strings)
    """
    logger.info("CHECKPOINT: Getting available models")
    models = []
    
    # Try using the ollama client first
    client = get_ollama_client()
    if client:
        try:
            models_data = client.list()
            if isinstance(models_data, dict) and "models" in models_data:
                models = [model["name"] for model in models_data["models"]]
            elif isinstance(models_data, list):
                models = [model["name"] for model in models_data]
            logger.info(f"CHECKPOINT: Found {len(models)} models using client")
            return models
        except Exception as client_error:
            logger.warning(f"CHECKPOINT: Error getting models with client: {str(client_error)}")
    
    # Fallback to direct API call
    try:
        response = requests.get(f"{get_ollama_url()}/tags")
        if response.status_code == 200:
            data = response.json()
            if "models" in data:
                models = [model["name"] for model in data["models"]]
            logger.info(f"CHECKPOINT: Found {len(models)} models using API")
            return models
    except Exception as api_error:
        logger.warning(f"CHECKPOINT: Error getting models with API: {str(api_error)}")
    
    # If no models found, return some default models
    if not models:
        logger.warning("CHECKPOINT: No models found, using default list")
        models = ["llama3", "llama3:8b", "llama3:70b", "mistral", "gemma", "phi"]
    
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
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        presence_penalty: Penalizes repeated tokens
        frequency_penalty: Penalizes frequent tokens
        context: Optional context for the model
        tools: Optional tools for function calling
        episodic_memory: Optional episodic memory
        format: Optional output format (json)
        capture_metrics: Whether to capture performance metrics
    
    Returns:
        tuple: (response_text, total_duration, load_duration, eval_count)
    """
    logger.info(f"CHECKPOINT: Calling Ollama endpoint with model {model}")
    
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
                return response.get("response", ""), None, None, None
        except Exception as mm_error:
            logger.warning(f"CHECKPOINT: Error with multimodal request: {str(mm_error)}")
    
    # Try using the ollama client
    if OLLAMA_AVAILABLE:
        try:
            client = get_ollama_client()
            if client:
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
                
                logger.info(f"CHECKPOINT: Successfully generated response with client (tokens: {eval_count})")
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
            
            logger.info(f"CHECKPOINT: Successfully generated response with API (tokens: {eval_count})")
            return response_text, total_duration, load_duration, eval_count
        else:
            logger.error(f"CHECKPOINT: API error: {response.status_code} - {response.text}")
            return f"Error: {response.status_code} - {response.text}", None, None, None
    except Exception as api_error:
        logger.error(f"CHECKPOINT: Error with API request: {str(api_error)}")
        return f"Error: {str(api_error)}", None, None, None

def get_all_models():
    """Gets all available models, including Ollama models."""
    # Get Ollama models
    ollama_model_names = get_available_models()
    
    # You can add other model sources here if needed
    # For now, just return Ollama models
    return ollama_model_names


# ============================================================================
# Workspace Integration Functions
# ============================================================================

def save_ai_content_to_workspace(content):
    """
    Save AI-generated content to the workspace.
    This function is a bridge to the chat_workspace module.
    
    Args:
        content: The content to save
    """
    logger.info("CHECKPOINT: Attempting to save content to workspace")
    try:
        # Import streamlit here when needed
        import streamlit as st
        
        # Import here to avoid circular imports
        from chat_workspace import save_ai_content_to_workspace as save_to_workspace
        
        # Call the actual implementation
        save_to_workspace(content)
        
        # Show a success notification if we're in a Streamlit context
        if st._is_running:
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
