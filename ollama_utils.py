# ollama_utils.py

import requests
import json
import io

import streamlit as st
import numpy as np
import ollama
import psutil
import platform
import subprocess
import os

from openai_utils import (
    call_openai_api,
    OPENAI_MODELS
)
from groq_utils import get_local_embeddings, GROQ_MODELS
from mistral_utils import MISTRAL_MODELS

API_KEYS_FILE = "api_keys.json"
MODEL_SETTINGS_FILE = "model_settings.json"

def load_api_keys():
    """Loads API keys from the JSON file."""
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_api_keys(api_keys):
    """Saves API keys to the JSON file."""
    with open(API_KEYS_FILE, "w") as f:
        json.dump(api_keys, f, indent=4)

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

OLLAMA_URL = "http://localhost:11434/api"

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

@st.cache_data
def get_available_models():
    response = requests.get(f"{OLLAMA_URL}/tags")
    response.raise_for_status()
    models = [
        model["name"]
        for model in response.json()["models"]
        if "embed" not in model["name"]
    ]
    return models

def call_ollama_endpoint(model, prompt=None, image=None, temperature=0.5, max_tokens=150, presence_penalty=0.0, frequency_penalty=0.0, context=None, tools=None, episodic_memory=None):
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
    if image:
        # Read image data into BytesIO
        image_bytesio = io.BytesIO(image.read())

        # Determine image format and filename
        image_format = "image/jpeg" if image.type == "image/jpeg" else "image/png"
        filename = "image.jpg" if image.type == "image/jpeg" else "image.png"

        # Send image data using multipart/form-data
        files = {"file": (filename, image_bytesio, image_format)}
        response = requests.post(f"{OLLAMA_URL}/generate", data=payload, files=files, stream=True)
    else:
        response = requests.post(f"{OLLAMA_URL}/generate", json=payload, stream=True)
    try:
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {str(e)}", None, None, None

    response_parts = []
    eval_count = None
    eval_duration = None
    for line in response.iter_lines():
        part = json.loads(line)
        response_parts.append(part.get("response", ""))
        if part.get("done", False):
            eval_count = part.get("eval_count", None)
            eval_duration = part.get("eval_duration", None)
            break
    return "".join(response_parts), part.get("context", None), eval_count, eval_duration

def check_json_handling(model, temperature, max_tokens, presence_penalty, frequency_penalty):
    prompt = "Return the following data in JSON format: name: John, age: 30, city: New York"
    result, _, _, _ = call_ollama_endpoint(model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)
    try:
        json.loads(result)
        return True
    except json.JSONDecodeError:
        return False

def get_token_embeddings(model: str, text: str, api_keys: dict) -> np.ndarray:
    """Gets embeddings for each token in the text and returns a 2D array."""
    try:
        if model in OPENAI_MODELS:
            response = call_openai_api(
                "text-embedding-ada-002",  # OpenAI's embedding model
                prompt=[{"role": "user", "content": text}],
                openai_api_key=api_keys.get("openai_api_key"),
                use_chat=False
            )
            embeddings = np.array(response['data'][0]['embedding'])
        elif model in GROQ_MODELS:
            embeddings = get_local_embeddings(text)
        else:
            response = ollama.embeddings(model=model, prompt=text)
            embeddings = np.array(response['embedding'])
        
        return embeddings.reshape(1, -1)  # Ensure it's a 2D array
    except Exception as e:
        st.error(f"An error occurred while getting token embeddings: {e}")
        return np.zeros((1, 1536))  # Return a default 2D array with 1536 features (common embedding size)

def check_function_calling(model, temperature, max_tokens, presence_penalty, frequency_penalty):
    prompt = "Define a function named 'add' that takes two numbers and returns their sum. Then call the function with arguments 5 and 3."
    result, _, _, _ = call_ollama_endpoint(model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)
    return "8" in result

def run_tool_test(model, description, tool_description, test_function, arguments):
    prompt = f"Test the function: {tool_description}. Arguments: {arguments}"
    result, _, _, _ = call_ollama_endpoint(model, prompt=prompt)
    return result

def pull_model(model_name):
    payload = {"name": model_name, "stream": True}
    response = requests.post(f"{OLLAMA_URL}/pull", json=payload, stream=True)
    response.raise_for_status()
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    total = None
    st.write(f"📥 Pulling model: `{model_name}`")
    for line in response.iter_lines():
        line = line.decode("utf-8")
        try:
            data = json.loads(line)
            
            if "total" in data and "completed" in data:
                total = data["total"]
                completed = data["completed"]
                progress = completed / total
                progress_bar.progress(progress)
                status_text.text(f"Progress: {progress * 100:.2f}%")
            elif "status" in data:
                if not data["status"].startswith("pulling"):
                    status_text.text(data["status"])
                if data["status"] == "success":
                    break
            else:
                # Handle cases where neither "total" nor "status" is present
                status_text.text("Processing...")
            
            results.append(data)
        except json.JSONDecodeError:
            st.warning(f"Failed to parse JSON: {line}")
        
    return results

def show_model_info(model_name):
    payload = {"name": model_name}
    response = requests.post(f"{OLLAMA_URL}/show", json=payload)
    response.raise_for_status()
    return response.json()

def remove_model(model_name):
    payload = {"name": model_name}
    response = requests.delete(f"{OLLAMA_URL}/delete", json=payload)
    if response.status_code == 200:
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"status": "success", "message": f"Model '{model_name}' removed successfully."}
    else:
        return {"status": "error", "message": f"Failed to remove model '{model_name}'. Status code: {response.status_code}"}

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

def generate_embeddings(model, text):
    """Generates embeddings for the given text using the specified model."""
    try:
        if model in GROQ_MODELS:
            # Use Groq API for embedding
            return get_local_embeddings(text)
        elif model in OPENAI_MODELS:
            # Use OpenAI API for embedding
            return call_openai_api(model, prompt=[{"role": "user", "content": text}], use_chat=False)
        else:
            # Default to Ollama API for embedding
            response = requests.post(f"{OLLAMA_URL}/embed", json={"model": model, "text": text})
            response.raise_for_status()
            embedding_data = response.json()
            return embedding_data["embedding"], embedding_data["total_duration"], embedding_data["load_duration"], embedding_data["prompt_eval_count"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error generating embeddings: {e}")
        return None, None, None, None

def get_all_models():
    """Gets all available models, including Ollama, Groq, OpenAI, and Mistral."""
    ollama_models = ["🦙 Ollama Models"] + get_available_models()
    groq_models = ["🚀 Groq Models"] + GROQ_MODELS
    openai_models = ["🤖 OpenAI Models"] + OPENAI_MODELS
    mistral_models = ["🌟 Mistral Models"] + MISTRAL_MODELS
    return ollama_models + groq_models + openai_models + mistral_models
