# server_monitoring.py
import streamlit as st
import subprocess
import json
import os
import platform

def get_ollama_resource_usage():
    try:
        result = subprocess.run(["ollama", "resource-usage", "--json"], capture_output=True, text=True)
        if result.returncode == 0:
            usage = json.loads(result.stdout)
            return usage
        else:
            st.error(f"Error getting resource usage: {result.stderr}")
            return {"status": "Unknown", "cpu_usage": "Unknown", "memory_usage": "Unknown", "gpu_usage": "Unknown"}
    except json.JSONDecodeError as e:
        st.error(f"Error decoding resource usage: {e}")
        return {"status": "Unknown", "cpu_usage": "Unknown", "memory_usage": "Unknown", "gpu_usage": "Unknown"}

def get_server_logs():
    log_file_path = get_log_file_path()
    if log_file_path:
        try:
            logs = subprocess.check_output(['tail', '-n', '1000', log_file_path]).decode('utf-8').split('\n')
            # Format logs with line breaks within the text_area
            formatted_logs = [f"{log}\n" for log in logs]
            return formatted_logs
        except subprocess.CalledProcessError:
            st.warning(f"Unable to access log file: {log_file_path}")
            return []
    else:
        return []

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
        st.warning("Unsupported operating system. Unable to fetch server logs.")
        return None

def server_monitoring():
    st.header("🖥️ Server Monitoring")

    # Resource Usage
    st.subheader("Resource Usage")
    usage = get_ollama_resource_usage()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", usage["status"])
    col2.metric("CPU Usage", usage["cpu_usage"])
    col3.metric("Memory Usage", usage["memory_usage"])
    col4.metric("GPU Usage", usage["gpu_usage"])

    # Live Log Stream
    st.subheader("Server Logs (Last 1000 Lines)")

    # Button to refresh the logs
    if st.button("Refresh Logs"):
        st.rerun()

    # Display the last 1000 lines of the log file
    logs = get_server_logs()
    logs = logs[-1000:]
    log_text = "".join(logs)
    st.text_area("Server Logs", value=log_text, height=400, key="server_logs")