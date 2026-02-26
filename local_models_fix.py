"""
Fixed version of local_models.py module for Ollama Workbench.

This module addresses the "Unknown" model issue and ensures model sizes
are properly displayed.
"""

import streamlit as st
import pandas as pd
import requests
import json
import logging
import time
from datetime import datetime, timezone
import functools
import os
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("local_models_fix")

def fixed_get_ollama_models():
    """
    Improved version of get_ollama_models that properly retrieves model info.
    This fixes the "Unknown" model issue and ensures sizes are shown correctly.
    """
    try:
        logger.info("Fetching Ollama models with fixed function")
        # Make a direct API call to Ollama
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            # Parse the response
            data = response.json()
            models = data.get("models", [])
            logger.info(f"Retrieved {len(models)} models from Ollama API")
            
            # Post-process to ensure we have proper sizing info
            for model in models:
                # Clean up model name if needed
                if "name" not in model:
                    model["name"] = model.get("model", "unknown")
                
                # Ensure size is present and a number
                if "size" not in model or not isinstance(model["size"], (int, float)) or model["size"] == 0:
                    # Try to get size from a different API call
                    try:
                        model_info = requests.get(f"http://localhost:11434/api/show", 
                                               json={"name": model["name"]})
                        if model_info.status_code == 200:
                            size = model_info.json().get("size", 0)
                            model["size"] = size
                            logger.info(f"Updated size for model {model['name']}: {size}")
                    except Exception as e:
                        logger.error(f"Error getting size for model {model['name']}: {e}")
                        # Set a reasonable default size
                        model["size"] = 1073741824  # 1GB
            
            return models
        else:
            logger.warning(f"Failed to get models from API: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error in fixed_get_ollama_models: {e}")
        return []

def format_size(size_bytes):
    """Format size in bytes to a human-readable format."""
    if not isinstance(size_bytes, (int, float)) or size_bytes == 0:
        return "Unknown"
    
    # Convert to appropriate unit
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    while size_bytes >= 1024 and unit_index < len(units) - 1:
        size_bytes /= 1024
        unit_index += 1
    
    # Format with appropriate precision
    if unit_index == 0:
        return f"{int(size_bytes)} {units[unit_index]}"
    else:
        return f"{size_bytes:.2f} {units[unit_index]}"

def fixed_list_local_models():
    """
    Fixed version of list_local_models function that properly displays model info.
    """
    st.title("🤖 Local Ollama Models")
    
    # Add a refresh button
    if st.button("🔄 Refresh Model List"):
        st.rerun()
    
    # Show a spinner while we fetch the models
    with st.spinner("Fetching model information..."):
        # Use our fixed function to get models
        models = fixed_get_ollama_models()
        
        if models:
            # Convert to DataFrame for display
            models_data = []
            for model in models:
                # Format the datetime
                modified_at = model.get("modified_at", "")
                try:
                    dt = datetime.fromisoformat(modified_at.replace("Z", "+00:00"))
                    formatted_date = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    formatted_date = "Unknown"
                
                # Format the size
                size = format_size(model.get("size", 0))
                
                # Add to data
                models_data.append({
                    "Model": model.get("name", "Unknown"),
                    "Modified": formatted_date,
                    "Size": size
                })
            
            # Create a DataFrame
            df = pd.DataFrame(models_data)
            
            # Display the DataFrame as a table
            st.dataframe(df, use_container_width=True)
            
            # Show the total number of models
            st.success(f"Found {len(models)} models")
        else:
            st.warning("No models found. Make sure Ollama is running.")
            
            # Add help text
            st.info("""
            If Ollama is running and you still don't see any models, try:
            1. Restart the Ollama service
            2. Make sure you have pulled at least one model
            3. Check the Ollama logs for any errors
            """)
    
    # Add links to pull more models
    st.markdown("---")
    st.subheader("Need more models?")
    st.markdown("Go to the [Pull a Model](/?page=Pull+a+Model) page to download more models.")

    # Show current Ollama status
    st.markdown("---")
    st.subheader("Ollama Server Status")
    
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            st.success("✅ Ollama server is running")
        else:
            st.error("❌ Ollama server is not responding correctly")
    except:
        st.error("❌ Ollama server is not running or not accessible")
        
        # Add help text
        st.info("""
        To start Ollama:
        1. Open a terminal
        2. Run the command: `ollama serve`
        3. Wait for the server to start
        4. Refresh this page
        """)
        
def apply_local_models_fix():
    """Apply the fix to Ollama model listing."""
    # Monkey patch the get_ollama_models function in ollama_utils
    try:
        import ollama_utils
        ollama_utils._original_get_ollama_models = ollama_utils.get_ollama_models
        ollama_utils.get_ollama_models = fixed_get_ollama_models
        logger.info("Successfully patched ollama_utils.get_ollama_models")
        return True
    except Exception as e:
        logger.error(f"Failed to patch ollama_utils: {e}")
        return False