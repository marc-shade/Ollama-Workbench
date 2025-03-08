# contextual_response.py
import streamlit as st
import pandas as pd
import time
from ollama_utils import get_available_models, call_ollama_endpoint, check_json_handling, check_function_calling

def contextual_response_test():
    st.header("💬 Contextual Response Test by Model")

    available_models = get_available_models()

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = available_models[0] if available_models else None

    selectbox_key = "contextual_test_model_selector"

    if selectbox_key in st.session_state:
        st.session_state.selected_model = st.session_state[selectbox_key]

    selected_model = st.selectbox(
        "Select the model you want to test:", 
        available_models, 
        key=selectbox_key,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
    )

    prompts = st.text_area("Enter the prompts (one per line):", value="Hi, how are you?\nWhat's your name?\nTell me a joke.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    with col2:
        max_tokens = st.slider("Max Tokens", min_value=100, max_value=32000, value=4000, step=100)
    with col3:
        presence_penalty = st.slider("Presence Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
    with col4:
        frequency_penalty = st.slider("Frequency Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

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

        data = {"Prompt": prompt_list, "Time (seconds)": times, "Tokens/second": tokens_per_second_list}
        df = pd.DataFrame(data)

        st.bar_chart(df, x="Prompt", y=["Time (seconds)", "Tokens/second"], color=["#4CAF50", "#FFC107"])

        st.write("📦 JSON Handling Capability: ", "✅" if check_json_handling(selected_model, temperature, max_tokens, presence_penalty, frequency_penalty) else "❌")
        st.write("⚙️ Function Calling Capability: ", "✅" if check_function_calling(selected_model, temperature, max_tokens, presence_penalty, frequency_penalty) else "❌")
