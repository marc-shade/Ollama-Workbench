import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import json
import time
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import ollama  # Import the ollama library
import os

# Set plot style based on Streamlit theme
if st.get_option("theme.base") == "light":
    plt.style.use('default')  # Use default white background for light mode
else:
    plt.style.use('dark_background')  # Use dark background for dark mode

OLLAMA_URL = "http://localhost:11434/api"

def get_available_models():
    response = requests.get(f"{OLLAMA_URL}/tags")
    response.raise_for_status()
    models = [
        model["name"]
        for model in response.json()["models"]
        if "embed" not in model["name"]
    ]
    return models

def get_model_hash(model_name):
    """Gets the hash of a model using the ollama.show endpoint."""
    payload = {"name": model_name}
    response = requests.post(f"{OLLAMA_URL}/show", json=payload)
    response.raise_for_status()
    model_info = response.json()
    return model_info.get("hash", None)

def get_latest_model_hash(model_name):
    """Gets the latest hash of a model from the ollama.tags endpoint."""
    response = requests.get(f"{OLLAMA_URL}/tags")
    response.raise_for_status()
    tags_info = response.json()
    for model in tags_info.get("models", []):
        if model["name"] == model_name:
            return model.get("hash", None)
    return None

def call_ollama_endpoint(model, prompt=None, image=None, temperature=0.5, max_tokens=150, presence_penalty=0.0, frequency_penalty=0.0, context=None):
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "context": context if context is not None else [],
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
        return f"An error occurred: {str(e)}", None, None, None  # Return None for eval_count and eval_duration

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

def performance_test(models, prompt, temperature=0.5, max_tokens=150, presence_penalty=0.0, frequency_penalty=0.0, context=None):
    results = {}
    for model in models:
        start_time = time.time()
        result, _, eval_count, eval_duration = call_ollama_endpoint(
            model,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            context=context,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        results[model] = (result, elapsed_time, eval_count, eval_duration)
        time.sleep(0.1)
    return results

def vision_test(models, image_file, temperature=0.5, max_tokens=150, presence_penalty=0.0, frequency_penalty=0.0, context=None):
    results = {}
    for model in models:
        start_time = time.time()
        try:
            # Read image data into BytesIO
            image_bytesio = io.BytesIO(image_file.read())
            result, _, eval_count, eval_duration = call_ollama_endpoint(
                model,
                image=image_bytesio,
                temperature=temperature,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                context=context,
            )
            print(f"Model: {model}, Result: {result}")  # Debug statement
        except Exception as e:
            result = f"An error occurred: {str(e)}"
        end_time = time.time()
        elapsed_time = end_time - start_time
        results[model] = (result, elapsed_time, eval_count, eval_duration)
        time.sleep(0.1)
    return results

def check_json_handling(model, temperature, max_tokens, presence_penalty, frequency_penalty):
    prompt = "Return the following data in JSON format: name: John, age: 30, city: New York"
    result, _, _, _ = call_ollama_endpoint(model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)
    try:
        json.loads(result)
        return True
    except json.JSONDecodeError:
        return False

def check_function_calling(model, temperature, max_tokens, presence_penalty, frequency_penalty):
    prompt = "Define a function named 'add' that takes two numbers and returns their sum. Then call the function with arguments 5 and 3."
    result, _, _, _ = call_ollama_endpoint(model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)
    return "8" in result

def list_local_models():
    response = requests.get(f"{OLLAMA_URL}/tags")
    response.raise_for_status()
    models = response.json().get("models", [])
    if not models:
        st.write("No local models available.")
        return
    
    # Prepare data for the dataframe
    data = []
    for model in models:
        size_gb = model.get('size', 0) / (1024**3)  # Convert bytes to GB
        modified_at = model.get('modified_at', 'Unknown')
        if modified_at != 'Unknown':
            modified_at = datetime.fromisoformat(modified_at).strftime('%Y-%m-%d %H:%M:%S')
        data.append({
            "Model Name": model['name'],
            "Size (GB)": size_gb,
            "Modified At": modified_at
        })
    
    # Create a pandas dataframe
    df = pd.DataFrame(data)

    # Calculate height based on the number of rows
    row_height = 35  # Set row height
    height = row_height * len(df) + 35  # Calculate height
    
    # Display the dataframe with Streamlit
    st.dataframe(df, use_container_width=True, height=height, hide_index=True)

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
        line = line.decode("utf-8")  # Decode the line from bytes to str
        data = json.loads(line)
        
        if "total" in data and "completed" in data:
            total = data["total"]
            completed = data["completed"]
            progress = completed / total
            progress_bar.progress(progress)
            status_text.text(f"Progress: {progress * 100:.2f}%")
        else:
            progress = None
            if not data["status"].startswith("pulling"):
                status_text.text(data["status"])
        
        if data["status"] == "success":
            break
        
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

def model_comparison_test():
    st.header("Model Comparison by Response Quality")
    available_models = get_available_models()
    selected_models = st.multiselect("Select the models you want to compare:", available_models)
    temperature = st.slider("Select the temperature:", min_value=0.0, max_value=1.0, value=0.5)
    max_tokens = st.number_input("Max tokens:", value=150)
    presence_penalty = st.number_input("Presence penalty:", value=0.0)
    frequency_penalty = st.number_input("Frequency penalty:", value=0.0)
    prompt = st.text_area("Enter the prompt:", value="Write a short story about a brave knight.")

    if st.button("Compare Models", key="compare_models"):
        results = performance_test(selected_models, prompt, temperature, max_tokens, presence_penalty, frequency_penalty)
        
        # Prepare data for visualization
        models = list(results.keys())  # Get models from results
        times = [results[model][1] for model in models]
        tokens_per_second = [
            results[model][2] / (results[model][3] / (10**9)) if results[model][2] and results[model][3] else 0
            for model in models
        ]

        df = pd.DataFrame({"Model": models, "Time (seconds)": times, "Tokens/second": tokens_per_second})

        # Plot the results using st.bar_chart
        st.bar_chart(df, x="Model", y=["Time (seconds)", "Tokens/second"], color=["#4CAF50", "#FFC107"])  # Green and amber
        
        for model, (result, elapsed_time, eval_count, eval_duration) in results.items():
            st.subheader(f"Results for {model} (Time taken: {elapsed_time:.2f} seconds, Tokens/second: {tokens_per_second[models.index(model)]:.2f}):")
            st.write(result)
            st.write("JSON Handling Capability: ", "✅" if check_json_handling(model, temperature, max_tokens, presence_penalty, frequency_penalty) else "❌")
            st.write("Function Calling Capability: ", "✅" if check_function_calling(model, temperature, max_tokens, presence_penalty, frequency_penalty) else "❌")

def contextual_response_test():
    st.header("Contextual Response Test by Model")
    available_models = get_available_models()
    selected_model = st.selectbox("Select the model you want to test:", available_models)
    prompts = st.text_area("Enter the prompts (one per line):", value="Hi, how are you?\nWhat's your name?\nTell me a joke.")
    temperature = st.slider("Select the temperature:", min_value=0.0, max_value=1.0, value=0.5)
    max_tokens = st.number_input("Max tokens:", value=150)
    presence_penalty = st.number_input("Presence penalty:", value=0.0)
    frequency_penalty = st.number_input("Frequency penalty:", value=0.0)

    if st.button("Start Contextual Test", key="start_contextual_test"):
        prompt_list = [p.strip() for p in prompts.split("\n")]
        context = []
        times = []
        tokens_per_second_list = []
        for i, prompt in enumerate(prompt_list):
            start_time = time.time()
            result, context, eval_count, eval_duration = call_ollama_endpoint(
                selected_model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                context=context,
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
            tokens_per_second = eval_count / (eval_duration / (10**9)) if eval_count and eval_duration else 0
            tokens_per_second_list.append(tokens_per_second)
            st.subheader(f"Prompt {i+1}: {prompt} (Time taken: {elapsed_time:.2f} seconds, Tokens/second: {tokens_per_second:.2f}):")
            st.write(f"Response: {result}")

        # Prepare data for visualization
        data = {"Prompt": prompt_list, "Time (seconds)": times, "Tokens/second": tokens_per_second_list}
        df = pd.DataFrame(data)

        # Plot the results using st.bar_chart
        st.bar_chart(df, x="Prompt", y=["Time (seconds)", "Tokens/second"], color=["#4CAF50", "#FFC107"])  # Green and amber

        st.write("JSON Handling Capability: ", "✅" if check_json_handling(selected_model, temperature, max_tokens, presence_penalty, frequency_penalty) else "❌")
        st.write("Function Calling Capability: ", "✅" if check_function_calling(selected_model, temperature, max_tokens, presence_penalty, frequency_penalty) else "❌")

def feature_test():
    st.header("Model Feature Test")
    available_models = get_available_models()
    selected_model = st.selectbox("Select the model you want to test:", available_models)
    temperature = st.slider("Select the temperature:", min_value=0.0, max_value=1.0, value=0.5)
    max_tokens = st.number_input("Max tokens:", value=150)
    presence_penalty = st.number_input("Presence penalty:", value=0.0)
    frequency_penalty = st.number_input("Frequency penalty:", value=0.0)

    if st.button("Run Feature Test", key="run_feature_test"):
        json_result = check_json_handling(selected_model, temperature, max_tokens, presence_penalty, frequency_penalty)
        function_result = check_function_calling(selected_model, temperature, max_tokens, presence_penalty, frequency_penalty)

        st.markdown(f"### JSON Handling Capability: {'✅ Success!' if json_result else '❌ Failure!'}")
        st.markdown(f"### Function Calling Capability: {'✅ Success!' if function_result else '❌ Failure!'}")

        # Prepare data for visualization
        data = {"Feature": ['JSON Handling', 'Function Calling'], "Result": [json_result, function_result]}
        df = pd.DataFrame(data)

        # Plot the results using st.bar_chart
        st.bar_chart(df, x="Feature", y="Result", color="#4CAF50")

def vision_comparison_test():
    st.header("Vision Model Comparison")
    available_models = get_available_models()
    # Ensure 'llava' is in available models before setting it as default
    default_models = ['llava'] if 'llava' in available_models else []
    selected_models = st.multiselect("Select the models you want to compare:", available_models, default=default_models)
    temperature = st.slider("Select the temperature:", min_value=0.0, max_value=1.0, value=0.5)
    max_tokens = st.number_input("Max tokens:", value=150)
    presence_penalty = st.number_input("Presence penalty:", value=0.0)
    frequency_penalty = st.number_input("Frequency penalty:", value=0.0)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if st.button("Compare Vision Models", key="compare_vision_models") and uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        results = {}
        for model in selected_models:
            # Reset file pointer to the beginning
            uploaded_file.seek(0)  # Add this line

            start_time = time.time()
            try:
                # Use ollama.chat for vision tests
                response = ollama.chat(
                    model=model,
                    messages=[
                        {
                            'role': 'user',
                            'content': 'Describe this image:',
                            'images': [uploaded_file]
                        }
                    ]
                )
                result = response['message']['content']
                print(f"Model: {model}, Result: {result}")  # Debug statement
            except Exception as e:
                result = f"An error occurred: {str(e)}"
            end_time = time.time()
            elapsed_time = end_time - start_time
            results[model] = (result, elapsed_time)
            time.sleep(0.1)

        # Display the LLM response text and time taken
        for model, (result, elapsed_time) in results.items():
            st.subheader(f"Results for {model} (Time taken: {elapsed_time:.2f} seconds):")
            st.write(result)

        # Prepare data for visualization (after displaying responses)
        models = list(results.keys())
        times = [results[model][1] for model in models]
        df = pd.DataFrame({"Model": models, "Time (seconds)": times})

        # Plot the results
        st.bar_chart(df, x="Model", y="Time (seconds)", color="#4CAF50")

def list_models():
    st.header("List Local Models")
    models = list_local_models()
    if models:
        # Prepare data for the dataframe
        data = []
        for model in models:
            size_gb = model.get('size', 0) / (1024**3)  # Convert bytes to GB
            modified_at = model.get('modified_at', 'Unknown')
            if modified_at != 'Unknown':
                modified_at = datetime.fromisoformat(modified_at).strftime('%Y-%m-%d %H:%M:%S')
            data.append({
                "Model Name": model['name'],
                "Size (GB)": size_gb,
                "Modified At": modified_at
            })
        
        # Create a pandas dataframe
        df = pd.DataFrame(data)

        # Calculate height based on the number of rows
        row_height = 35  # Set row height
        height = row_height * len(df) + 35  # Calculate height
        
        # Display the dataframe with Streamlit
        st.dataframe(df, use_container_width=True, height=height, hide_index=True)

def pull_models():
    st.header("Pull a Model from Ollama Library")
    model_name = st.text_input("Enter the name of the model you want to pull:")
    if st.button("Pull Model", key="pull_model"):
        if model_name:
            result = pull_model(model_name)
            for status in result:
                st.write(status)
        else:
            st.error("Please enter a model name.")

def show_model_details():
    st.header("Show Model Information")
    available_models = get_available_models()
    selected_model = st.selectbox("Select the model you want to show details for:", available_models)
    if st.button("Show Model Information", key="show_model_information"):
        details = show_model_info(selected_model)
        st.json(details)

def remove_model_ui():
    st.header("Remove a Model")
    available_models = get_available_models()
    selected_model = st.selectbox("Select the model you want to remove:", available_models)
    confirm_label = f"❌ Confirm removal of model `{selected_model}`"
    confirm = st.checkbox(confirm_label)
    if st.button("Remove Model", key="remove_model") and confirm:
        if selected_model:
            result = remove_model(selected_model)
            st.write(result["message"])
            # Update the list of available models
            st.session_state.available_models = get_available_models()
            st.rerun()
        else:
            st.error("Please select a model.")

def update_models():
    st.header("Update Local Models")
    available_models = get_available_models()
    if st.button("Update All Models"):
        for model_name in available_models:
            local_hash = get_model_hash(model_name)

            # Check if the model is legitimate (has a valid hash)
            if local_hash:
                st.write(f"Updating model: `{model_name}`")
                pull_model(model_name)  # Pull the model regardless of hash match
            else:
                st.write(f"Skipping model with invalid hash: `{model_name}`")
        st.success("All models updated.")

def save_chat_history(chat_history, filename="chat_history.json"):
    with open(filename, "w") as f:
        json.dump(chat_history, f)

def load_chat_history(filename):
    with open(filename, "r") as f:
        return json.load(f)

def chat_interface():
    st.header("Chat with a Model")
    available_models = get_available_models()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = available_models[0] if available_models else None

    selected_model = st.selectbox("Select a model:", available_models, key="selected_model")
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.5)
    max_tokens = st.number_input("Max Tokens:", value=150)
    presence_penalty = st.number_input("Presence Penalty:", value=0.0)
    frequency_penalty = st.number_input("Frequency Penalty:", value=0.0)

    # Save chat history
    if st.button("Save Chat"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"chat_history_{timestamp}.json"
        chat_name = st.text_input("Enter a name for the chat:", value=default_filename)
        if chat_name:
            save_chat_history(st.session_state.chat_history, chat_name)
            st.success(f"Chat history saved to {chat_name}")

    # Load/Rename/Delete chat history
    st.sidebar.subheader("Saved Chats")
    saved_chats = [f for f in os.listdir() if f.endswith(".json")]

    # State variable to track which chat is being renamed
    if "rename_chat" not in st.session_state:
        st.session_state.rename_chat = None

    for chat in saved_chats:
        col1, col2, col3 = st.sidebar.columns([3, 1, 1])
        with col1:
            # Display chat name without .json extension
            chat_name = os.path.splitext(chat)[0] 
            if st.button(chat_name):
                st.session_state.chat_history = load_chat_history(chat)
                st.rerun()
        with col2:
            if st.button("✏️", key=f"rename_{chat}"):
                st.session_state.rename_chat = chat  # Set chat to be renamed
                st.rerun()
        with col3:
            if st.button("🗑️", key=f"delete_{chat}"):
                os.remove(chat)
                st.success(f"Chat {chat} deleted.")
                st.rerun()

    # Text input for renaming (outside the loop)
    if st.session_state.rename_chat:
        # Display current name without .json extension
        current_name = os.path.splitext(st.session_state.rename_chat)[0]
        new_name = st.text_input("Rename chat:", value=current_name)
        if new_name:
            # Add .json extension to the new name
            new_name = new_name + ".json" 
            if new_name != st.session_state.rename_chat:
                os.rename(st.session_state.rename_chat, new_name)
                st.success(f"Chat renamed to {new_name}")
                st.session_state.rename_chat = None  # Reset rename_chat
                st.cache_resource.clear()
                st.rerun()

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("What is up my person?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response using ollama library
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            for response_chunk in ollama.generate(selected_model, prompt, stream=True):
                full_response += response_chunk["response"]
                response_placeholder.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

def main():
    if 'selected_test' not in st.session_state:
        st.session_state.selected_test = None

    with st.sidebar:
        st.markdown(
            '<div style="text-align: left;">'
            '<h1 class="logo" style="font-size: 50px;">🦙 Ollama <span style="color: orange;">Workbench</span></h1>'
            "</div>",
            unsafe_allow_html=True,
        )

        st.subheader("Chat") # Chat Section
        st.markdown('<style>div.row-widget.stButton > button {width:100%;}</style>', unsafe_allow_html=True)
        if st.button("Chat", key="button_chat"):
            st.session_state.selected_test = "Chat"

        # Maintain Section (Collapsible)
        with st.expander("Maintain", expanded=False):
            st.markdown('<style>div.row-widget.stButton > button {width:100%;}</style>', unsafe_allow_html=True)
            if st.button("List Local Models", key="button_list_models"):
                st.session_state.selected_test = "List Local Models"
            if st.button("Show Model Information", key="button_show_model_info"):
                st.session_state.selected_test = "Show Model Information"
            if st.button("Pull a Model", key="button_pull_model"):
                st.session_state.selected_test = "Pull a Model"
            if st.button("Remove a Model", key="button_remove_model"):
                st.session_state.selected_test = "Remove a Model"
            # Add Update Models button
            if st.button("Update Models", key="button_update_models"):
                st.session_state.selected_test = "Update Models"

        # Test Section (Collapsible)
        with st.expander("Test", expanded=False):
            st.markdown('<style>div.row-widget.stButton > button {width:100%;}</style>', unsafe_allow_html=True)
            if st.button("Model Feature Test", key="button_feature_test"):
                st.session_state.selected_test = "Model Feature Test"
            if st.button("Model Comparison by Response Quality", key="button_model_comparison"):
                st.session_state.selected_test = "Model Comparison by Response Quality"
            if st.button("Contextual Response Test by Model", key="button_contextual_response"):
                st.session_state.selected_test = "Contextual Response Test by Model"
            if st.button("Vision Model Comparison", key="button_vision_model_comparison"):
                st.session_state.selected_test = "Vision Model Comparison"

    if st.session_state.selected_test == "Model Comparison by Response Quality":
        model_comparison_test()
    elif st.session_state.selected_test == "Contextual Response Test by Model":
        contextual_response_test()
    elif st.session_state.selected_test == "Model Feature Test":
        feature_test()
    elif st.session_state.selected_test == "List Local Models":
        list_models()
    elif st.session_state.selected_test == "Pull a Model":
        pull_models()
    elif st.session_state.selected_test == "Show Model Information":
        show_model_details()
    elif st.session_state.selected_test == "Remove a Model":
        remove_model_ui()
    elif st.session_state.selected_test == "Vision Model Comparison":
        vision_comparison_test()
    elif st.session_state.selected_test == "Chat":
        chat_interface()
    elif st.session_state.selected_test == "Update Models":
        update_models()
    else:
        st.write("""
            ### Welcome to the Ollama Workbench!
            Use the sidebar to select a test or maintenance function.

            #### Maintain
            - **List Local Models**: View a list of all locally available models, including their size and last modified date.
            - **Show Model Information**: Display detailed information about a selected model.
            - **Pull a Model**: Download a new model from the Ollama library.
            - **Remove a Model**: Delete a selected model from the local storage.

            #### Test
            - **Model Feature Test**: Test a model's capability to handle JSON and function calls.
            - **Model Comparison by Response Quality**: Compare the response quality of multiple models for a given prompt.
            - **Contextual Response Test by Model**: Test how well a model maintains context across multiple prompts.
            - **Vision Model Comparison**: Compare the performance of vision models using the same test image.
            - **Chat**: Engage in a real-time chat with a selected model.
        """)

if __name__ == "__main__":
    main()
