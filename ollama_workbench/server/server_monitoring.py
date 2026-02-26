import streamlit as st
import subprocess
import json
import os
import platform
import psutil

def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            return result.stdout
        else:
            st.error(f"Error running command: {result.stderr}")
            return None
    except Exception as e:
        st.error(f"Exception running command: {e}")
        return None

def get_ollama_ps():
    return run_command("ollama ps")

def get_server_logs():
    log_file_path = get_log_file_path()
    if log_file_path:
        return run_command(f"tail -n 1000 {log_file_path}")
    else:
        return None

def get_log_file_path():
    system = platform.system()
    if system == "Darwin":
        return os.path.expanduser("~/.ollama/logs/server.log")
    elif system == "Linux":
        return "/var/log/ollama/server.log"
    elif system == "Windows":
        return os.path.join(os.environ["LOCALAPPDATA"], "Ollama", "server.log")
    else:
        st.warning("Unsupported operating system. Unable to fetch server logs.")
        return None

def get_ollama_resource_usage():
    ollama_process = None
    for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']):
        if proc.info['name'] == 'ollama':
            ollama_process = proc
            break

    if ollama_process:
        cpu_usage = ollama_process.cpu_percent(interval=1)
        memory_usage = ollama_process.memory_percent()
        status = "Running"
    else:
        cpu_usage = 0
        memory_usage = 0
        status = "Not Running"

    # GPU usage (this is a placeholder, as getting GPU usage is more complex and system-dependent)
    gpu_usage = "N/A"

    return {
        "status": status,
        "cpu_usage": f"{cpu_usage:.2f}%",
        "memory_usage": f"{memory_usage:.2f}%",
        "gpu_usage": gpu_usage
    }

def server_monitoring():
    st.header("🖥️ Ollama Server Monitoring")

    # Resource Usage
    st.subheader("Resource Usage")
    usage = get_ollama_resource_usage()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", usage["status"])
    col2.metric("CPU Usage", usage["cpu_usage"])
    col3.metric("Memory Usage", usage["memory_usage"])
    col4.metric("GPU Usage", usage["gpu_usage"])

    # Running Models
    st.subheader("Running Models")
    running_models = get_ollama_ps()
    if running_models:
        st.text_area("Running Models", value=running_models, height=200)

    # Live Log Stream
    st.subheader("Server Logs (Last 1000 Lines)")
    if st.button("Refresh Logs"):
        st.rerun()
    logs = get_server_logs()
    if logs:
        st.text_area("Server Logs", value=logs, height=400, key="server_logs")

    # Server Configuration
    st.subheader("Server Configuration")
    if platform.system() == "Linux":
        config_path = "/etc/ollama/config.json"
    elif platform.system() == "Darwin":
        config_path = os.path.expanduser("~/Library/Application Support/Ollama/config.json")
    elif platform.system() == "Windows":
        config_path = os.path.join(os.environ["APPDATA"], "Ollama", "config.json")
    else:
        st.warning("Unsupported operating system. Unable to fetch server configuration.")
        config_path = None

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            st.json(config)
        
        st.download_button("Download Config", json.dumps(config, indent=4), file_name="ollama_config.json")
    else:
        st.warning("Configuration file not found.")