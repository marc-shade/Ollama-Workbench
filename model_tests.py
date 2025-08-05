# model_tests.py
import streamlit as st
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
from ollama_utils import call_ollama_endpoint
import io # Added import
import asyncio
from ollama import AsyncClient
from typing import Callable
from groq_utils import load_api_keys, GROQ_MODELS

# Set plot style based on Streamlit theme
if st.get_option("theme.base") == "light":
    plt.style.use('default')  # Use default white background for light mode
else:
    plt.style.use('dark_background')  # Use dark background for dark mode

def performance_test(models, prompt, temperature=0.7, max_tokens=1000, presence_penalty=0.0, frequency_penalty=0.0, context=None):
    results = {}
    api_keys = load_api_keys()
    if models:
        for model in models:
            start_time = time.time()
            if model.startswith("gpt-"):
                result = call_openai_api(model, [{"role": "user", "content": prompt}], temperature, max_tokens, api_keys.get("openai_api_key"))
                response_text = result.get('choices')[0].get('text') if result.get('choices') else None
                results[model] = (response_text, None, None, None)  # Adjust as necessary
            elif model in GROQ_MODELS:
                result = call_groq_api(model, prompt, temperature, max_tokens, api_keys.get("groq_api_key"))
                response_text = result.get('choices')[0].get('text') if result.get('choices') else None
                results[model] = (response_text, None, None, None)  # Adjust as necessary
            else:
                result, _, eval_count, eval_duration = call_ollama_endpoint(
                    model,
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    context=context,
                )
                response_text = result  # Assuming result is the response text
                results[model] = (response_text, None, eval_count, eval_duration)

            end_time = time.time()
            elapsed_time = end_time - start_time
            if len(results[model]) > 1:
                results[model] = (results[model][0], elapsed_time, results[model][2], results[model][3])

        # Prepare data for visualization
        models = list(results.keys())
        times = [results[model][1] for model in models]
        tokens_per_second = [
            results[model][2] / (results[model][3] / (10**9)) if len(results[model]) > 3 and results[model][2] and results[model][3] else 0
            for model in models
        ]

        return results
    else:
        return {}

def vision_test(models, image_file, temperature=0.7, max_tokens=1000, presence_penalty=0.0, frequency_penalty=0.0, context=None):
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

async def run_tool_test(model: str, prompt: str, tool_description: str, function_to_call: Callable, arguments: dict):
    """Runs a test using Ollama's tool calling feature."""
    client = AsyncClient()
    messages = [{'role': 'user', 'content': prompt}]

    response = await client.chat(
        model=model,
        messages=messages,
        tools=[
            {
                'type': 'function',
                'function': {
                    'name': 'tool_function',
                    'description': tool_description,
                    'parameters': {
                        'type': 'object',
                        'properties': arguments,
                        'required': list(arguments.keys()),
                    },
                },
            },
        ],
    )

    messages.append(response['message'])

    if not response['message'].get('tool_calls'):
        print("The model didn't use the function. Its response was:")
        print(response['message']['content'])
        return response['message']['content']

    if response['message'].get('tool_calls'):
        for tool in response['message']['tool_calls']:
            function_response = function_to_call(**tool['function']['arguments'])
            messages.append(
                {
                    'role': 'tool',
                    'content': function_response,
                }
            )

    final_response = await client.chat(model=model, messages=messages)
    return final_response['message']['content']

def model_tool_test_ui(): # Encapsulate the tool test UI in a function
    prompt_input = st.text_input("Enter a prompt for the model tool test", key="tool_prompt_input") # Changed prompt text to be specific
    if st.button("Run Model Tool Test"): # Changed button label to be specific
        model = "your_model_name"  # You can also make this dynamic
        prompt = prompt_input  # Use the existing dynamic prompt
        tool_description = "Your tool description here"  # Adjust as necessary
        arguments = {}  # Add any necessary arguments here
        generated_text = asyncio.run(run_tool_test(model, prompt, tool_description, lambda: "This is a test response", arguments))
        st.markdown(f"### Model Tool Test Result: {generated_text}") # Changed st.write to st.markdown for better formatting and clarity and specific text