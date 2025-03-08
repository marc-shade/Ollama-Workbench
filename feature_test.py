# feature_test.py
from model_tests import model_tool_test_ui
import streamlit as st
import asyncio
from ollama_utils import get_available_models as get_ollama_models, check_json_handling, check_function_calling, run_tool_test
from openai_utils import OPENAI_MODELS, call_openai_api
from groq_utils import GROQ_MODELS, call_groq_api
from ollama_utils import load_api_keys
import ollama

def feature_test():
    st.header("🏟️ AI Model Feature Test")
    model_tool_test_ui() # Call it inside feature_test
    
    # Combining models from all sources
    all_models = {
        "Ollama Models": get_ollama_models(),
        "OpenAI Models": OPENAI_MODELS,
        "Groq Models": GROQ_MODELS
    }

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = list(all_models.values())[0][0] if all_models else None

    selectbox_key = "feature_test_model_selector"
    
    if selectbox_key in st.session_state:
        st.session_state.selected_model = st.session_state[selectbox_key]

    # Model selection
    selected_provider = st.selectbox("Select Model Provider:", list(all_models.keys()), key="model_provider_selector")
    selected_model = st.selectbox(
        "Select the model you want to test:", 
        all_models[selected_provider], 
        key=selectbox_key,
        index=all_models[selected_provider].index(st.session_state.selected_model) if st.session_state.selected_model in all_models[selected_provider] else 0
    )

    # Adjust parameters for the API calls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    with col2:
        max_tokens = st.slider("Max Tokens", min_value=1000, max_value=128000, value=4000, step=1000)
    with col3:
        presence_penalty = st.slider("Presence Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
    with col4:
        frequency_penalty = st.slider("Frequency Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

    # Test function for tool testing
    def test_function(arg1: str, arg2: int) -> str:
        return f"Argument 1: {arg1}, Argument 2: {arg2}"

    tool_description = "A test function that takes two arguments: a string and an integer."
    arguments = {
        "arg1": {"type": "string", "description": "The first argument (a string)."},
        "arg2": {"type": "integer", "description": "The second argument (an integer)."}
    }

    if st.button("Run Feature Test", key="run_feature_test"):
        # Load API keys
        api_keys = load_api_keys()
        
        # Execute tests based on the selected provider
        if selected_provider == "Ollama Models":
            # Proper tool testing for Ollama models
            try:
                client = ollama.Client()
                prompt = f"Here is a simple test function in Python that takes two arguments: {arguments['arg1']['description']} and {arguments['arg2']['description']}."
                response = client.generate(model=selected_model, prompt=prompt)
                if 'response' in response:
                    tool_result = response['response']  # Extracting the result from the response
                    st.markdown(f"### 🧰 Ollama Tool Test Result: {tool_result}")
                else:
                    st.error(f"Unexpected response structure: {response}")
            except Exception as e:
                st.error(f"An error occurred during the Ollama tool test: {e}")

            # JSON handling
            json_result = check_json_handling(selected_model, temperature, max_tokens, presence_penalty, frequency_penalty)
            st.markdown(f"### 📦 Ollama JSON Handling Capability: {'✅ Success!' if json_result else '❌ Failure!'}")

            # Function calling
            function_result = check_function_calling(selected_model, temperature, max_tokens, presence_penalty, frequency_penalty)
            st.markdown(f"### ⚙️ Ollama Function Calling Capability: {'✅ Success!' if function_result else '❌ Failure!'}")

        elif selected_provider == "OpenAI Models":
            prompt = "Test the tool function with structured data."
            try:
                # Tool testing (simulating tool usage through API call)
                tool_prompt = "Run a function to calculate the sum of two numbers: 2 and 3."
                tool_result = call_openai_api(
                    selected_model,
                    [{"role": "user", "content": tool_prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    openai_api_key=api_keys.get("openai_api_key")
                )
                st.markdown(f"### 🧰 OpenAI Tool Test Result: {tool_result}")

                # JSON handling (using the API to process JSON input/output)
                json_prompt = "Convert the following JSON into a summary: {\"name\": \"Alice\", \"age\": 30, \"city\": \"New York\"}"
                json_result = call_openai_api(
                    selected_model,
                    [{"role": "user", "content": json_prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    openai_api_key=api_keys.get("openai_api_key")
                )
                st.markdown(f"### 📦 OpenAI JSON Handling Capability: {json_result}")

                # Function calling (simulating function execution)
                function_prompt = "Call a function with the arguments: 'hello', 42"
                function_result = call_openai_api(
                    selected_model,
                    [{"role": "user", "content": function_prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    openai_api_key=api_keys.get("openai_api_key")
                )
                st.markdown(f"### ⚙️ OpenAI Function Calling Capability: {function_result}")

            except Exception as e:
                st.error(f"An error occurred during the OpenAI model test: {e}")

        elif selected_provider == "Groq Models":
            prompt = "Test the tool function with structured data."
            try:
                # Tool testing (simulating tool usage through API call)
                tool_prompt = "Run a function to calculate the sum of two numbers: 2 and 3."
                tool_result = call_groq_api(
                    selected_model,
                    tool_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    groq_api_key=api_keys.get("groq_api_key")
                )
                st.markdown(f"### 🧰 Groq Tool Test Result: {tool_result}")

                # JSON handling (using the API to process JSON input/output)
                json_prompt = "Convert the following JSON into a summary: {\"name\": \"Alice\", \"age\": 30, \"city\": \"New York\"}"
                json_result = call_groq_api(
                    selected_model,
                    json_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    groq_api_key=api_keys.get("groq_api_key")
                )
                st.markdown(f"### 📦 Groq JSON Handling Capability: {json_result}")

                # Function calling (simulating function execution)
                function_prompt = "Call a function with the arguments: 'hello', 42"
                function_result = call_groq_api(
                    selected_model,
                    function_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    groq_api_key=api_keys.get("groq_api_key")
                )
                st.markdown(f"### ⚙️ Groq Function Calling Capability: {function_result}")

            except Exception as e:
                st.error(f"An error occurred during the Groq model test: {e}")

# Main function to run the app
    
if __name__ == "__main__":
    feature_test()
