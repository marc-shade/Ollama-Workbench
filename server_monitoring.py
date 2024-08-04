# server_monitoring.py
import streamlit as st
import subprocess
import json

def get_ollama_resource_usage():
    result = subprocess.run(["ollama", "resource-usage"], capture_output=True, text=True)
    usage = json.loads(result.stdout)
    return usage

def get_server_logs():
    return subprocess.check_output(['tail', '-n', '1000', '/var/log/ollama/server.log']).decode('utf-8').split('\n')

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