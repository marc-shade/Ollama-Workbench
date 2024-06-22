# ui_elements.py
import streamlit as st
import pandas as pd
from datetime import datetime
import os
import ollama
from ollama_utils import *
from model_tests import *
import requests

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

def update_model_selection(selected_models, key):
    """Callback function to update session state during form submission."""
    st.session_state[key] = selected_models

@st.cache_data  # Cache the comparison and visualization logic
def run_comparison(selected_models, prompt, temperature, max_tokens, presence_penalty, frequency_penalty):
    results = performance_test(selected_models, prompt, temperature, max_tokens, presence_penalty, frequency_penalty)

    # Prepare data for visualization
    models = list(results.keys())  # Get models from results
    times = [results[model][1] for model in models]
    tokens_per_second = [
        results[model][2] / (results[model][3] / (10**9)) if results[model][2] and results[model][3] else 0
        for model in models
    ]

    df = pd.DataFrame({"Model": models, "Time (seconds)": times, "Tokens/second": tokens_per_second})

    return results, df, tokens_per_second, models  # Return models

def model_comparison_test():
    st.header("Model Comparison by Response Quality")

    # Refresh available_models list
    available_models = get_available_models()

    # Initialize selected_models in session state if it doesn't exist
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []

    # Pass the session state variable as the default for st.multiselect
    selected_models = st.multiselect(
        "Select the models you want to compare:",
        available_models,
        default=st.session_state.selected_models,  # Use session state for default
        key="model_comparison_models"  # Unique key for this multiselect
    )

    temperature = st.slider("Select the temperature:", min_value=0.0, max_value=1.0, value=0.5)
    max_tokens = st.number_input("Max tokens:", value=150)
    presence_penalty = st.number_input("Presence penalty:", value=0.0)
    frequency_penalty = st.number_input("Frequency penalty:", value=0.0)
    prompt = st.text_area("Enter the prompt:", value="Write a short story about a brave knight.")

    # Check if the button is clicked
    if st.button(label='Compare Models'):
        if selected_models:
            # Run the comparison and get the results, dataframe, tokens_per_second, and models
            results, df, tokens_per_second, models = run_comparison(selected_models, prompt, temperature, max_tokens, presence_penalty, frequency_penalty)

            # Plot the results using st.bar_chart
            st.bar_chart(df, x="Model", y=["Time (seconds)", "Tokens/second"], color=["#4CAF50", "#FFC107"])  # Green and amber

            for model, (result, elapsed_time, eval_count, eval_duration) in results.items():
                st.subheader(f"Results for {model} (Time taken: {elapsed_time:.2f} seconds, Tokens/second: {tokens_per_second[models.index(model)]:.2f}):")
                st.write(result)
                st.write("JSON Handling Capability: ", "✅" if check_json_handling(model, temperature, max_tokens, presence_penalty, frequency_penalty) else "❌")
                st.write("Function Calling Capability: ", "✅" if check_function_calling(model, temperature, max_tokens, presence_penalty, frequency_penalty) else "❌")
        else:
            st.warning("Please select at least one model.")

def vision_comparison_test():
    st.header("Vision Model Comparison")

    # Refresh available_models list
    available_models = get_available_models()

    # Initialize selected_vision_models in session state if it doesn't exist
    if "selected_vision_models" not in st.session_state:
        st.session_state.selected_vision_models = []

    # Pass the session state variable as the default for st.multiselect
    selected_models = st.multiselect(
        "Select the models you want to compare:",
        available_models,
        default=st.session_state.selected_vision_models,  # Use session state for default
        key="vision_comparison_models"  # Unique key for this multiselect
    )
    temperature = st.slider("Select the temperature:", min_value=0.0, max_value=1.0, value=0.5)
    max_tokens = st.number_input("Max tokens:", value=150)
    presence_penalty = st.number_input("Presence penalty:", value=0.0)
    frequency_penalty = st.number_input("Frequency penalty:", value=0.0)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    # Check if the button is clicked
    if st.button(label='Compare Vision Models'):
        if uploaded_file is not None:
            if selected_models:
                # Display the uploaded image
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

                results = {}
                for model in selected_models:
                    # Reset file pointer to the beginning
                    uploaded_file.seek(0)

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
            else:
                st.warning("Please select at least one model.")
        else:
            st.warning("Please upload an image.")

def contextual_response_test():
    st.header("Contextual Response Test by Model")

    # Refresh available_models list
    available_models = get_available_models()

    # Initialize selected_model in session state if it doesn't exist
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = available_models[0] if available_models else None

    # Use a separate key for the selectbox
    selectbox_key = "contextual_test_model_selector"

    # Update selected_model when selectbox changes
    if selectbox_key in st.session_state:
        st.session_state.selected_model = st.session_state[selectbox_key]

    selected_model = st.selectbox(
        "Select the model you want to test:", 
        available_models, 
        key=selectbox_key,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
    )

    # Display the currently selected model
    st.write(f"Currently selected model: {selected_model}")

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
    
    # Refresh available_models list
    available_models = get_available_models()

    # Initialize selected_model in session state if it doesn't exist
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = available_models[0] if available_models else None

    # Use a separate key for the selectbox
    selectbox_key = "feature_test_model_selector"

    # Update selected_model when selectbox changes
    if selectbox_key in st.session_state:
        st.session_state.selected_model = st.session_state[selectbox_key]

    selected_model = st.selectbox(
        "Select the model you want to test:", 
        available_models, 
        key=selectbox_key,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
    )

    # Display the currently selected model
    st.write(f"Currently selected model: {selected_model}")

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
    
    # Refresh available_models list
    available_models = get_available_models()

    # Initialize selected_model in session state if it doesn't exist
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = available_models[0] if available_models else None

    # Use a separate key for the selectbox
    selectbox_key = "show_model_details_model_selector"

    # Update selected_model when selectbox changes
    if selectbox_key in st.session_state:
        st.session_state.selected_model = st.session_state[selectbox_key]

    selected_model = st.selectbox(
        "Select the model you want to show details for:", 
        available_models, 
        key=selectbox_key,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
    )

    # Display the currently selected model
    st.write(f"Currently selected model: {selected_model}")

    if st.button("Show Model Information", key="show_model_information"):
        details = show_model_info(selected_model)
        st.json(details)

def remove_model_ui():
    st.header("Remove a Model")
    
    # Refresh available_models list
    available_models = get_available_models()

    # Initialize selected_model in session state if it doesn't exist
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = available_models[0] if available_models else None

    # Use a separate key for the selectbox
    selectbox_key = "remove_model_ui_model_selector"

    # Update selected_model when selectbox changes
    if selectbox_key in st.session_state:
        st.session_state.selected_model = st.session_state[selectbox_key]

    selected_model = st.selectbox(
        "Select the model you want to remove:", 
        available_models, 
        key=selectbox_key,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
    )

    confirm_label = f"❌ Confirm removal of model `{selected_model}`"
    confirm = st.checkbox(confirm_label)
    if st.button("Remove Model", key="remove_model") and confirm:
        if selected_model:
            result = remove_model(selected_model)
            st.write(result["message"])
            # Update the list of available models
            st.session_state.available_models = get_available_models()
            # Update selected_model if it was removed
            if selected_model not in st.session_state.available_models:
                st.session_state.selected_model = st.session_state.available_models[0] if st.session_state.available_models else None
            st.rerun()
        else:
            st.error("Please select a model.")

def update_models():
    st.header("Update Local Models")
    available_models = get_available_models()
    if st.button("Update All Models"):
        for model_name in available_models:
            # Skip custom models (those with a ':' in the name)
            if 'gpt' in model_name:
                st.write(f"Skipping custom model: `{model_name}`")
                continue
            st.write(f"Updating model: `{model_name}`")
            pull_model(model_name)
        st.success("All models updated successfully!")

def chat_interface():
    st.header("Chat with a Model")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Refresh available_models list
    available_models = get_available_models()

    # Initialize selected_model in session state if it doesn't exist
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = available_models[0] if available_models else None

    # Use a separate key for the selectbox
    selectbox_key = "chat_model_selector"

    # Update selected_model when selectbox changes
    if selectbox_key in st.session_state:
        st.session_state.selected_model = st.session_state[selectbox_key]

    selected_model = st.selectbox(
        "Select a model:", 
        available_models, 
        key=selectbox_key,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
    )

    # Display the currently selected model
    st.write(f"Currently selected model: {selected_model}")

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

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            for response_chunk in ollama.generate(st.session_state.selected_model, prompt, stream=True):
                full_response += response_chunk["response"]
                response_placeholder.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})