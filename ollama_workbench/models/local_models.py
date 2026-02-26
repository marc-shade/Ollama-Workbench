# local_models.py
import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime
from ollama_workbench.providers.ollama_utils import *

def list_local_models():
    st.title("🤖 Local Ollama Models")
    
    # Clear cache to ensure fresh results
    get_available_models.clear()
    
    # Add a refresh button
    if st.button("🔄 Refresh Model List"):
        st.rerun()
    
    # Try multiple methods to get models for maximum reliability
    try:
        # First try using our improved get_available_models function
        available_models = get_available_models()
        
        if available_models:
            # Get detailed info for each model using direct API or client
            models_info = []
            
            # Show progress while loading model details
            with st.spinner("Loading model details..."):
                for model_name in available_models:
                    try:
                        # Get detailed info for this model
                        model_info = show_model_info(model_name)
                        if model_info:
                            models_info.append(model_info)
                    except Exception as e:
                        st.warning(f"Could not load details for model {model_name}: {str(e)}")
            
            # If we got detailed info, use it
            if models_info:
                models = models_info
            else:
                # Fallback to direct API call
                try:
                    response = requests.get(f"{OLLAMA_URL}/tags", timeout=5)
                    response.raise_for_status()
                    models = response.json().get("models", [])
                except:
                    # Final fallback - construct basic info from available_models
                    models = [{"name": model} for model in available_models]
        else:
            # Try direct API call if get_available_models returned empty
            try:
                response = requests.get(f"{OLLAMA_URL}/tags", timeout=5)
                response.raise_for_status()
                models = response.json().get("models", [])
            except Exception as e:
                st.error(f"Error fetching models: {str(e)}")
                models = []
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        models = []
    
    if not models:
        st.warning("No local models available. Use the 'Pull Model' tool to install models.")
        return

    # Prepare data for the dataframe
    data = []
    for model in models:
        # Handle both detailed model info and basic model info formats
        model_name = model.get('name', model if isinstance(model, str) else "Unknown")
        
        # Extract size if available (with fallback handling)
        size_bytes = model.get('size', 0) if isinstance(model, dict) else 0
        size_gb = size_bytes / (1024**3) if size_bytes else 0  # Convert bytes to GB
        
        # Extract modified time if available
        modified_at = model.get('modified_at', 'Unknown') if isinstance(model, dict) else 'Unknown'
        if modified_at != 'Unknown':
            try:
                modified_at = datetime.fromisoformat(modified_at).strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass  # Keep as is if parsing fails
                
        # Extract capabilities if available
        from .model_capability_registry import get_model_capabilities
        capabilities = []
        try:
            model_caps = get_model_capabilities(model_name)
            if model_caps.get("vision", False):
                capabilities.append("Vision")
            if model_caps.get("tools", False):
                capabilities.append("Tools")
            if model_caps.get("embedding", False):
                capabilities.append("Embeddings")
        except:
            pass
            
        data.append({
            "Model Name": model_name,
            "Size (GB)": f"{size_gb:.2f}",
            "Modified At": modified_at,
            "Capabilities": ", ".join(capabilities) if capabilities else "Text",
            "Preload": False,
            "Keep-Alive": ""
        })

    # Create a pandas dataframe
    df = pd.DataFrame(data)

    # Display the editable dataframe
    edited_df = st.data_editor(
        df,
        hide_index=True,
        column_config={
            "Model Name": st.column_config.TextColumn("Model Name", disabled=True),
            "Size (GB)": st.column_config.NumberColumn("Size (GB)", format="%.2f", disabled=True),
            "Modified At": st.column_config.TextColumn("Modified At", disabled=True),
            "Capabilities": st.column_config.TextColumn("Capabilities", disabled=True),
            "Preload": st.column_config.CheckboxColumn("Preload"),
            "Keep-Alive": st.column_config.TextColumn("Keep-Alive", help="Value like 5m, 1h (empty for default)")
        },
        column_order=("Model Name", "Size (GB)", "Capabilities", "Modified At", "Preload", "Keep-Alive"),
        key="model_editor"
    )

    # Add Apply Changes button
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Apply Settings", use_container_width=True):
            with st.spinner("Applying model settings..."):
                for index, row in edited_df.iterrows():
                    model_name = row["Model Name"]
                    if row["Preload"]:
                        preload_model(model_name)
                    if row["Keep-Alive"]:
                        apply_model_keep_alive(model_name, row["Keep-Alive"])
                st.success("✅ Model settings applied successfully!")
    
    with col2:
        if st.button("📊 Show Server Status", use_container_width=True):
            with st.spinner("Checking server status..."):
                usage = get_ollama_resource_usage()
                st.info(f"Server Status: {usage['status']}")
                st.metric("CPU Usage", usage["cpu_usage"])
                st.metric("Memory Usage", usage["memory_usage"])
                if usage["gpu_usage"] != "N/A":
                    st.metric("GPU Usage", usage["gpu_usage"])