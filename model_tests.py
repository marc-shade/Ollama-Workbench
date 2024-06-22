# model_tests.py
import streamlit as st
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
from ollama_utils import call_ollama_endpoint

# Set plot style based on Streamlit theme
if st.get_option("theme.base") == "light":
    plt.style.use('default')  # Use default white background for light mode
else:
    plt.style.use('dark_background')  # Use dark background for dark mode

def performance_test(models, prompt, temperature=0.5, max_tokens=150, presence_penalty=0.0, frequency_penalty=0.0, context=None):
    results = {}
    if models:  # Check if any models are selected
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
    else:
        return {}  # Return an empty dictionary if no models are selected

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