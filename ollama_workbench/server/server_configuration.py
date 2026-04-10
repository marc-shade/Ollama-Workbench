# server_configuration.py
import streamlit as st
import os
import platform
import json
import shutil
import subprocess
from ollama_workbench.core.config import server_config_ui, get_config, update_config, CONFIG

def get_default_model_dir():
    system = platform.system()
    if system == "Darwin":
        return os.path.expanduser("~/.ollama/models")
    elif system == "Linux":
        return "/usr/share/ollama/.ollama/models"
    elif system == "Windows":
        return os.path.join(os.environ["USERPROFILE"], ".ollama", "models")
    else:
        return ""

def get_default_max_loaded_models():
    system = platform.system()
    if system == "Windows" and platform.machine().endswith("64"):
        return 1
    else:
        try:
            import GPUtil
            num_gpus = len(GPUtil.getGPUs())
            return 3 * num_gpus if num_gpus > 0 else 3
        except ImportError:
            return 3

def stop_server():
    """
    Stop the Ollama server using the appropriate method for the platform.
    """
    system = platform.system()
    
    try:
        if system == "Linux":
            # Try systemctl first
            if shutil.which("systemctl"):
                subprocess.run(["sudo", "systemctl", "stop", "ollama"], check=False)
            # Fall back to killall
            elif shutil.which("killall"):
                subprocess.run(["killall", "ollama"], check=False)
        elif system == "Darwin":  # macOS
            if shutil.which("pkill"):
                subprocess.run(["pkill", "ollama"], check=False)
        elif system == "Windows":
            # For Windows, try to use taskkill
            subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], check=False)
        
        st.success("Ollama server has been stopped.")
    except Exception as e:
        st.error(f"Failed to stop Ollama server: {str(e)}")

def apply_server_settings(host, origins, model_dir, global_keep_alive, max_loaded_models, num_parallel, max_queue):
    """
    Apply server settings to Ollama configuration file.
    """
    config = {
        "OLLAMA_HOST": host,
        "OLLAMA_ORIGINS": origins,
        "OLLAMA_MODELS": model_dir,
        "OLLAMA_KEEP_ALIVE": global_keep_alive,
        "OLLAMA_MAX_LOADED_MODELS": max_loaded_models,
        "OLLAMA_NUM_PARALLEL": num_parallel,
        "OLLAMA_MAX_QUEUE": max_queue
    }
    
    system = platform.system()
    
    try:
        if system == "Linux":
            config_path = "/etc/ollama/config.json"
            with open(config_path, "w") as config_file:
                json.dump(config, config_file, indent=2)
        elif system == "Darwin":  # macOS
            config_path = os.path.expanduser("~/.ollama/config.json")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as config_file:
                json.dump(config, config_file, indent=2)
        elif system == "Windows":
            config_path = os.path.join(os.environ["USERPROFILE"], ".ollama", "config.json")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as config_file:
                json.dump(config, config_file, indent=2)
        
        # Also add these settings to our Workbench configuration
        update_config({"OLLAMA_HOST": host})
        
        st.success(f"Server settings applied to {config_path}")
    except Exception as e:
        st.error(f"Failed to apply server settings: {str(e)}")

def start_server():
    """
    Start the Ollama server using the appropriate method for the platform.
    """
    system = platform.system()
    
    try:
        if system == "Linux":
            # Try systemctl first
            if shutil.which("systemctl"):
                subprocess.run(["sudo", "systemctl", "start", "ollama"], check=False)
            # Fall back to running ollama directly
            elif shutil.which("ollama"):
                # Run in background
                subprocess.Popen(["ollama", "serve"], start_new_session=True)
        elif system == "Darwin":  # macOS
            if shutil.which("ollama"):
                # Run in background
                subprocess.Popen(["ollama", "serve"], start_new_session=True)
        elif system == "Windows":
            # For Windows, try to run ollama.exe
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        
        st.success("Ollama server has been started.")
    except Exception as e:
        st.error(f"Failed to start Ollama server: {str(e)}")

def get_server_status():
    """
    Check if the Ollama server is running.
    
    Returns:
        bool: True if server is running, False otherwise
    """
    import requests
    import psutil
    
    # First check: Try using the API
    config = get_config()
    host = config.get("OLLAMA_HOST", "http://localhost:11434")
    
    # Ensure the host has http:// prefix
    if not host.startswith("http://") and not host.startswith("https://"):
        host = f"http://{host}"
    
    # Try multiple endpoints for more robust detection
    try:
        # First try the main API endpoint
        response = requests.get(f"{host}/api/tags", timeout=2)
        if response.status_code == 200:
            return True
    except Exception:
        pass  # Continue to next check if this fails

    try:
        # Try the health check endpoint
        response = requests.get(f"{host}/api/version", timeout=2)
        if response.status_code == 200:
            return True
    except Exception:
        pass  # Continue to next check if this fails

    # Second check: Look for Ollama process
    try:
        for proc in psutil.process_iter(['name']):
            proc_name = proc.info['name'].lower()
            if proc_name == 'ollama' or proc_name == 'ollama.exe':
                return True
    except Exception:
        pass  # Continue to next check if this fails

    # Third check: Try direct localhost without config
    try:
        direct_response = requests.get("http://localhost:11434/api/tags", timeout=1)
        if direct_response.status_code == 200:
            return True
    except Exception:
        pass  # Continue to final result if this fails
    
    # If all checks fail, the server is likely not running
    return False

def server_configuration():
    st.header("⚙️ Ollama Server Configuration")
    
    # Check if Ollama server is running
    server_running = get_server_status()
    
    # Display server status
    if server_running:
        st.success("✅ Ollama server is running")
    else:
        st.error("❌ Ollama server is not running")
    
    # Create tabs for Ollama-specific and workbench configs
    tab1, tab2 = st.tabs(["Ollama Server Settings", "Workbench Configuration"])
    
    with tab1:
        st.subheader("Host/Bind Address")
        host = st.text_input("OLLAMA_HOST", value=CONFIG.get("OLLAMA_HOST", "127.0.0.1"))
        
        st.subheader("Allowed Origins")
        origins = st.text_input("OLLAMA_ORIGINS", value="127.0.0.1, 0.0.0.0")
        
        st.subheader("Model Directory")
        model_dir = st.text_input("OLLAMA_MODELS", value=get_default_model_dir())
        
        st.subheader("Global Keep-Alive")
        global_keep_alive = st.text_input("OLLAMA_KEEP_ALIVE", value="5m")
        
        st.subheader("Concurrency Control")
        col1, col2, col3 = st.columns(3)
        with col1:
            max_loaded_models = st.number_input("OLLAMA_MAX_LOADED_MODELS", value=get_default_max_loaded_models(), min_value=1)
        with col2:
            num_parallel = st.number_input("OLLAMA_NUM_PARALLEL", value=4, min_value=1)
        with col3:
            max_queue = st.number_input("OLLAMA_MAX_QUEUE", value=512, min_value=1)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply Settings"):
                apply_server_settings(host, origins, model_dir, global_keep_alive, max_loaded_models, num_parallel, max_queue)
        
        with col2:
            if server_running:
                if st.button("Stop Ollama Server"):
                    stop_server()
            else:
                if st.button("Start Ollama Server"):
                    start_server()
        
        st.info("Changing some settings may require restarting the Ollama server to take effect.")
        if st.button("Restart Ollama Server"):
            stop_server()
            # Add a small delay to ensure the server has time to stop
            import time
            time.sleep(2)
            start_server()
    
    with tab2:
        # Use the centralized server config UI
        server_config_ui()