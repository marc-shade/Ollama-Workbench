# model_comparison.py
import streamlit as st
import pandas as pd
from ollama_utils import get_available_models, check_json_handling, check_function_calling
from model_tests import performance_test

@st.cache_data
def run_comparison(selected_models, prompt, temperature, max_tokens, presence_penalty, frequency_penalty):
    results = performance_test(selected_models, prompt, temperature, max_tokens, presence_penalty, frequency_penalty)

    # Prepare data for visualization
    models = list(results.keys())
    times = [results[model][1] for model in models]
    tokens_per_second = [
        results[model][2] / (results[model][3] / (10**9)) if results[model][2] and results[model][3] else 0
        for model in models
    ]

    df = pd.DataFrame({"Model": models, "Time (seconds)": times, "Tokens/second": tokens_per_second})

    return results, df, tokens_per_second, models

def model_comparison_test():
    st.header("🎯 Model Comparison by Response Quality")

    available_models = get_available_models()

    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []

    selected_models = st.multiselect(
        "Select the models you want to compare:",
        available_models,
        default=st.session_state.selected_models,
        key="model_comparison_models"
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

    prompt = st.text_area("Enter the prompt:", value="Write a short story about a brave knight.")

    if st.button(label='Compare Models'):
        if selected_models:
            results, df, tokens_per_second, models = run_comparison(selected_models, prompt, temperature, max_tokens, presence_penalty, frequency_penalty)

            st.bar_chart(df, x="Model", y=["Time (seconds)", "Tokens/second"], color=["#4CAF50", "#FFC107"])

            for model, (result, elapsed_time, eval_count, eval_duration) in results.items():
                st.subheader(f"Results for {model} (Time taken: {elapsed_time:.2f} seconds, Tokens/second: {tokens_per_second[models.index(model)]:.2f}):")
                st.write(result)
                st.write("📦 JSON Handling Capability: ", "✅" if check_json_handling(model, temperature, max_tokens, presence_penalty, frequency_penalty) else "❌")
                st.write("⚙️ Function Calling Capability: ", "✅" if check_function_calling(model, temperature, max_tokens, presence_penalty, frequency_penalty) else "❌")
        else:
            st.warning("Please select at least one model.")
