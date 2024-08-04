# feature_test.py
import streamlit as st
import asyncio
from ollama_utils import get_available_models, check_json_handling, check_function_calling, run_tool_test

def feature_test():
    st.header("🧪 Model Feature Test")
    
    available_models = get_available_models()

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = available_models[0] if available_models else None

    selectbox_key = "feature_test_model_selector"

    if selectbox_key in st.session_state:
        st.session_state.selected_model = st.session_state[selectbox_key]

    selected_model = st.selectbox(
        "Select the model you want to test:", 
        available_models, 
        key=selectbox_key,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    with col2:
        max_tokens = st.slider("Max Tokens", min_value=1000, max_value=128000, value=4000, step=1000)
    with col3:
        presence_penalty = st.slider("Presence Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
    with col4:
        frequency_penalty = st.slider("Frequency Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

    def test_function(arg1: str, arg2: int) -> str:
        return f"Argument 1: {arg1}, Argument 2: {arg2}"

    tool_description = "A test function that takes two arguments: a string and an integer."
    arguments = {
        "arg1": {"type": "string", "description": "The first argument (a string)."},
        "arg2": {"type": "integer", "description": "The second argument (an integer)."}
    }

    if st.button("Run Feature Test", key="run_feature_test"):
        try:
            result = asyncio.run(run_tool_test(selected_model, "Test the tool function.", tool_description, test_function, arguments))

            st.markdown(f"### 🧰 Tool Test Result: {result}")
        except Exception as e:
            st.error(f"An error occurred during the tool test: {e}")

        json_result = check_json_handling(selected_model, temperature, max_tokens, presence_penalty, frequency_penalty)
        function_result = check_function_calling(selected_model, temperature, max_tokens, presence_penalty, frequency_penalty)

        st.markdown(f"### 📦 JSON Handling Capability: {'✅ Success!' if json_result else '❌ Failure!'}")
        st.markdown(f"### ⚙️ Function Calling Capability: {'✅ Success!' if function_result else '❌ Failure!'}")
