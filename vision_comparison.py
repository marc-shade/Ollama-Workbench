# vision_comparison.py
import streamlit as st
import pandas as pd
import time
from ollama_utils import get_available_models
import ollama

def vision_comparison_test():
    st.header("👁️ Vision Model Comparison")

    available_models = get_available_models()

    if "selected_vision_models" not in st.session_state:
        st.session_state.selected_vision_models = []

    selected_models = st.multiselect(
        "Select the models you want to compare:",
        available_models,
        default=st.session_state.selected_vision_models,
        key="vision_comparison_models"
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

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if st.button(label='Compare Vision Models'):
        if uploaded_file is not None:
            if selected_models:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

                results = {}
                for model in selected_models:
                    uploaded_file.seek(0)

                    start_time = time.time()
                    try:
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
                        print(f"Model: {model}, Result: {result}")
                    except Exception as e:
                        result = f"An error occurred: {str(e)}"
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    results[model] = (result, elapsed_time)
                    time.sleep(0.1)

                for model, (result, elapsed_time) in results.items():
                    st.subheader(f"Results for {model} (Time taken: {elapsed_time:.2f} seconds):")
                    st.write(result)

                models = list(results.keys())
                times = [results[model][1] for model in models]
                df = pd.DataFrame({"Model": models, "Time (seconds)": times})

                st.bar_chart(df, x="Model", y="Time (seconds)", color="#4CAF50")
            else:
                st.warning("Please select at least one model.")
        else:
            st.warning("Please upload an image.")
