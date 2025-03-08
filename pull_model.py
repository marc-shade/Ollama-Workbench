# pull_model.py
import streamlit as st
import requests
import json
import time
import humanize

def parse_model_name(input_string: str) -> str:
    """Parse and clean the model name from the user input."""
    parts = input_string.split()
    if len(parts) > 1 and parts[0].lower() in ["ollama", "run", "pull"]:
        return parts[-1]
    return input_string.strip()

def format_size(size_bytes: int) -> str:
    """Format the file size in a human-readable format."""
    return humanize.naturalsize(size_bytes, binary=True)

def pull_model(model_name):
    """
    Pull a model from the Ollama library.

    Args:
    model_name (str): The name of the model to pull.

    Yields:
    dict: Status updates during the pull process, including progress information.
    """
    url = "http://localhost:11434/api/pull"
    payload = {"name": model_name, "stream": True}
    headers = {"Content-Type": "application/json"}

    try:
        with requests.post(url, json=payload, headers=headers, stream=True) as response:
            response.raise_for_status()
            total_size = 0
            downloaded_size = 0
            start_time = time.time()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "status" in data:
                        status = data["status"]
                        current_time = time.time()

                        if status == "pulling manifest":
                            yield {"status": "Initializing...", "progress": 0}
                        elif "total" in data and "completed" in data:
                            total_size = data.get("total", total_size)
                            downloaded_size = data.get("completed", downloaded_size)
                            progress = downloaded_size / total_size if total_size > 0 else 0

                            elapsed_time = current_time - start_time
                            speed = downloaded_size / elapsed_time if elapsed_time > 0 else 0
                            remaining_time = (total_size - downloaded_size) / speed if speed > 0 else 0

                            yield {
                                "status": "Downloading",
                                "progress": progress,
                                "details": f"{format_size(downloaded_size)} of {format_size(total_size)} downloaded",
                                "speed": format_size(speed) + "/s",
                                "remaining_time": humanize.naturaldelta(remaining_time)
                            }
                        elif status in ["verifying sha256 digest", "writing manifest", "removing any unused layers"]:
                            yield {"status": "Finalizing...", "progress": 1}
                        elif status == "success":
                            yield {"status": "Complete", "progress": 1}
    except requests.RequestException as e:
        yield {"status": f"Error: {str(e)}", "progress": 0}

def pull_models():
    st.header("⬇ Pull a Model from Ollama Library")
    st.write("""
    Enter the exact name of the model you want to pull from the Ollama library. 
    You can just paste the whole model snippet from the model library page like 'ollama run llava-phi3' 
    or you can just enter the model name like 'llava-phi3' and then click 'Pull Model' to begin the download. 
    The progress of the download will be displayed below.
    """)

    col1, col2 = st.columns([4, 1], vertical_alignment="bottom")

    with col1:
        model_input = st.text_input("Enter the name of the model you want to pull:")
    
    with col2:
        pull_button = st.button("Pull Model", key="pull_model")

    if pull_button:
        if model_input:
            model_name = parse_model_name(model_input)
            progress_bar = st.progress(0)
            status_text = st.empty()
            details_text = st.empty()
            speed_text = st.empty()
            remaining_time_text = st.empty()

            for update in pull_model(model_name):
                progress_bar.progress(int(update["progress"] * 100))
                status_text.text(update["status"])
                
                if "details" in update:
                    details_text.text(update["details"])
                
                if "speed" in update:
                    speed_text.text(f"Speed: {update['speed']}")
                    remaining_time_text.text(f"Estimated Time Remaining: {update['remaining_time']}")
                
                time.sleep(0.1)  # Small delay to prevent overwhelming the UI

            if update["status"] == "Complete":
                st.success(f"Model '{model_name}' pulled successfully!")
            elif "Error" in update["status"]:
                st.error(update["status"])
        else:
            st.error("Please enter a model name.")

    st.markdown("---")
    st.subheader("ℹ About Model Pulling")
    st.write("""
    Pulling a model downloads it from the Ollama library to your local machine. 
    This process may take several minutes for large models. 
    If the progress seems stuck, the download is likely still ongoing in the background.
    """)

if __name__ == "__main__":
    pull_models()
