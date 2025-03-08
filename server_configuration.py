# server_configuration.py
import streamlit as st
import os
import platform
import json

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
    os.system("sudo systemctl stop ollama-server")

def apply_server_settings(host, origins, model_dir, global_keep_alive, max_loaded_models, num_parallel, max_queue):
    config = {
        "OLLAMA_HOST": host,
        "OLLAMA_ORIGINS": origins,
        "OLLAMA_MODELS": model_dir,
        "OLLAMA_KEEP_ALIVE": global_keep_alive,
        "OLLAMA_MAX_LOADED_MODELS": max_loaded_models,
        "OLLAMA_NUM_PARALLEL": num_parallel,
        "OLLAMA_MAX_QUEUE": max_queue
    }
    config_path = "/etc/ollama/config.json"
    with open(config_path, "w") as config_file:
        json.dump(config, config_file)
    st.success("Server settings applied.")

def start_server():
    os.system("sudo systemctl start ollama-server")

def server_configuration():
    st.header("⚙️ Ollama Server Configuration")

    st.subheader("Host/Bind Address")
    host = st.text_input("OLLAMA_HOST", value="127.0.0.1")

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

    if st.button("Stop Ollama Server"):
        stop_server()

    if st.button("Apply Settings"):
        apply_server_settings(host, origins, model_dir, global_keep_alive, max_loaded_models, num_parallel, max_queue)

    st.info("Click the button below to manually restart the Ollama server with the applied settings.")
    if st.button("Restart Ollama Server"):
        start_server()
