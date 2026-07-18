# model_onboarding.py

import streamlit as st
import pandas as pd
import time
import json
import io
import re
import asyncio
import matplotlib.pyplot as plt
import ollama
from ollama_workbench.providers.ollama_utils import get_available_models, call_ollama_endpoint, generate_embeddings, check_json_handling, check_function_calling
from .model_tests import performance_test, vision_test
from typing import List, Dict, Any, Optional, Tuple

# Set plot style based on Streamlit theme
if st.get_option("theme.base") == "light":
    plt.style.use('default')  # Use default white background for light mode
else:
    plt.style.use('dark_background')  # Use dark background for dark mode

# Define test categories and benchmark prompts
TEST_CATEGORIES = {
    "General Knowledge": [
        "Explain the theory of relativity in simple terms.",
        "What were the major causes of World War II?",
        "Describe how photosynthesis works in plants."
    ],
    "Reasoning": [
        "If a ball costs $3 and a bat costs $7 more than the ball, how much do they cost together?",
        "A train travels at 60 mph. How far will it go in 2.5 hours?",
        "If 5 people can build 5 widgets in 5 minutes, how long will it take 10 people to build 10 widgets?"
    ],
    "Creative Writing": [
        "Write a short story about a robot discovering emotions.",
        "Compose a poem about autumn leaves.",
        "Create a dialogue between the sun and the moon."
    ],
    "Coding": [
        "Write a Python function to check if a string is a palindrome.",
        "Create a JavaScript function to sort an array of objects by a specific property.",
        "Implement a simple React component that displays a counter with increment and decrement buttons."
    ],
    "Function Calling": [
        "Calculate the sum of 256 and 789.",
        "Convert 25 degrees Celsius to Fahrenheit.",
        "Calculate the area of a circle with radius 5.2 meters."
    ],
    "JSON Handling": [
        "Convert this data to JSON: name=John, age=35, city=New York, languages=[Python, JavaScript]",
        "Parse this JSON and explain what it represents: {\"temperature\": 22.5, \"conditions\": \"sunny\", \"forecast\": [\"clear\", \"partly cloudy\", \"rainy\"]}",
        "Create a JSON object representing a book with title, author, publication year, and genres."
    ]
}

# Define vision test prompts
VISION_TEST_PROMPTS = {
    "Object Recognition": "What objects do you see in this image?",
    "Scene Description": "Describe this image in detail.",
    "Text Recognition": "Read and transcribe any text in this image.",
    "Spatial Understanding": "Describe the spatial relationships between objects in this image.",
    "Color Analysis": "What are the main colors in this image and where do they appear?"
}

def run_test_suite(model: str, categories: List[str], prompts_per_category: int = 1) -> Dict[str, Any]:
    """Run a complete test suite on a given model"""
    results = {"model": model, "categories": {}}
    
    for category in categories:
        if category not in TEST_CATEGORIES:
            continue
            
        category_prompts = TEST_CATEGORIES[category]
        selected_prompts = category_prompts[:prompts_per_category]
        
        category_results = []
        for prompt in selected_prompts:
            start_time = time.time()
            try:
                # Use the version-independent API
                from ollama_workbench.providers.ollama_utils import get_ollama_client, call_ollama_endpoint
                client = get_ollama_client()
                
                if client:
                    # New Client API
                    response = client.generate(
                        model=model,
                        prompt=prompt,
                        options={
                            "temperature": 0.7,
                            "num_predict": 2000
                        }
                    )
                    result_text = response["response"]
                else:
                    # Fallback to version-independent function
                    result_text, _, _, _, _ = call_ollama_endpoint(
                        model=model,
                        prompt=prompt,
                        temperature=0.7,
                        max_tokens=2000
                    )
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # Check word count as a simple evaluation metric
                word_count = len(result_text.split())
                
                category_results.append({
                    "prompt": prompt,
                    "response": result_text,
                    "time": elapsed_time,
                    "words": word_count
                })
            except Exception as e:
                category_results.append({
                    "prompt": prompt,
                    "response": f"Error: {str(e)}",
                    "time": 0,
                    "words": 0
                })
        
        results["categories"][category] = category_results
    
    return results

def run_vision_test(model: str, image_file, prompts: List[str]) -> Dict[str, Any]:
    """Run vision capabilities test on a given model"""
    if not image_file:
        return {"error": "No image provided"}
        
    results = {"model": model, "prompts": {}}
    
    for prompt in prompts:
        image_file.seek(0)
        start_time = time.time()
        try:
            # Use the version-independent API
            from ollama_workbench.providers.ollama_utils import get_ollama_client, call_ollama_endpoint
            client = get_ollama_client()
            
            if client:
                # New Client API
                try:
                    response = client.chat(
                        model=model,
                        messages=[
                            {
                                'role': 'user',
                                'content': prompt,
                                'images': [image_file]
                            }
                        ]
                    )
                    result_text = response['message']['content']
                except Exception as chat_e:
                    # Fallback in case chat API doesn't support images
                    st.warning(f"Vision API error: {str(chat_e)}. Attempting direct generation.")
                    result_text = f"Vision test encountered an error: {str(chat_e)}. This model might not support multimodal inputs."
            else:
                # Fallback for older versions
                result_text = f"This version of the Ollama API doesn't support vision capabilities. Please update to a newer version."
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Check word count as a simple evaluation metric
            word_count = len(result_text.split())
            
            results["prompts"][prompt] = {
                "response": result_text,
                "time": elapsed_time,
                "words": word_count
            }
        except Exception as e:
            results["prompts"][prompt] = {
                "response": f"Error: {str(e)}",
                "time": 0,
                "words": 0
            }
            
    return results

def run_function_call_test(model: str) -> Dict[str, Any]:
    """Test the model's capability to use function calling"""
    results = {"model": model}
    
    # Simple calculator function definition
    calculator_tool = {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform basic arithmetic operations",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The operation to perform"
                    },
                    "a": {
                        "type": "number",
                        "description": "The first number"
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number"
                    }
                },
                "required": ["operation", "a", "b"]
            }
        }
    }
    
    prompt = "Calculate 24 * 7"
    
    try:
        # Use the version-independent API
        from ollama_workbench.providers.ollama_utils import get_ollama_client, call_ollama_endpoint
        client = get_ollama_client()
        
        if not client:
            # Fallback for older versions
            results["tool_support"] = False
            results["tool_call_success"] = False
            results["response"] = "This version of the Ollama API doesn't support function calling. Please update to a newer version."
            return results
        
        # Try with the client
        try:
            response = client.chat(
                model=model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    }
                ],
                tools=[calculator_tool]
            )
            
            has_function_call = False
            has_tool_calls = False
            function_call_content = ""
            
            if "message" in response:
                if "content" in response["message"]:
                    function_call_content = response["message"]["content"]
                if "tool_calls" in response["message"]:
                    has_tool_calls = len(response["message"]["tool_calls"]) > 0
                if "function_call" in response["message"]:
                    has_function_call = True
            
            # Set initial results
            results["tool_support"] = has_tool_calls or has_function_call
            results["response"] = function_call_content
            
            # Now handle tool calls if present
            if has_tool_calls and "message" in response and 'tool_calls' in response['message']:
                tool_call = response['message']['tool_calls'][0]
                function_name = tool_call['function']['name']
                arguments = json.loads(tool_call['function']['arguments'])
                
                results["tool_call_success"] = True
                results["function_name"] = function_name
                results["arguments"] = arguments
                
                # Calculate result to simulate function call
                if arguments["operation"] == "add":
                    result = arguments["a"] + arguments["b"]
                elif arguments["operation"] == "subtract":
                    result = arguments["a"] - arguments["b"]
                elif arguments["operation"] == "multiply":
                    result = arguments["a"] * arguments["b"]
                elif arguments["operation"] == "divide":
                    result = arguments["a"] / arguments["b"]
                else:
                    result = "Unknown operation"
                    
                try:
                    # Send the result back
                    final_response = client.chat(
                        model=model,
                        messages=[
                            {'role': 'user', 'content': prompt},
                            response['message'],
                            {
                                'role': 'tool',
                                'tool_call_id': tool_call['id'],
                                'name': function_name,
                                'content': str(result)
                            }
                        ]
                    )
                    
                    results["final_response"] = final_response['message']['content']
                except Exception as tool_e:
                    results["final_response"] = f"Error during tool response: {str(tool_e)}"
            else:
                results["tool_call_success"] = False
                results["model_response"] = function_call_content
                
        except Exception as chat_e:
            # Fallback if chat API doesn't support tools
            results["tool_support"] = False
            results["tool_call_success"] = False
            results["response"] = f"Function calling test encountered an error: {str(chat_e)}. This model might not support function calls."
            
    except Exception as e:
        results["tool_support"] = False
        results["tool_call_success"] = False
        results["error"] = str(e)
        results["response"] = f"Error during function calling test: {str(e)}"
        
    return results

def run_json_format_test(model: str) -> Dict[str, Any]:
    """Test the model's capability to produce formatted JSON output"""
    results = {"model": model}
    
    prompt = "Return information about a fictional user profile in JSON format with fields: username, age, email, and interests (as an array)."
    
    try:
        client = ollama.Client()
        response = client.generate(
            model=model,
            prompt=prompt,
            format="json"
        )
        
        result_text = response["response"]
        
        # Try to parse the response as JSON
        try:
            parsed_json = json.loads(result_text)
            results["json_valid"] = True
            results["parsed_json"] = parsed_json
            
            # Check if the expected fields are present
            expected_fields = ["username", "age", "email", "interests"]
            missing_fields = [field for field in expected_fields if field not in parsed_json]
            
            results["missing_fields"] = missing_fields
            results["fields_coverage"] = (len(expected_fields) - len(missing_fields)) / len(expected_fields)
            
            # Check if interests is an array
            if "interests" in parsed_json:
                results["interests_is_array"] = isinstance(parsed_json["interests"], list)
            else:
                results["interests_is_array"] = False
                
        except json.JSONDecodeError:
            results["json_valid"] = False
            results["raw_response"] = result_text
    except Exception as e:
        results["json_valid"] = False
        results["error"] = str(e)
        
    return results

def calculate_model_scores(test_results: Dict[str, Any]) -> Dict[str, float]:
    """Calculate scores for each capability based on test results"""
    scores = {}
    
    # General capabilities from standard test suite
    if "categories" in test_results:
        for category, results in test_results["categories"].items():
            # Check response quality (very basic - just ensures no errors and reasonable length)
            response_quality = []
            response_speed = []
            
            for result in results:
                if "Error:" not in result["response"] and result["words"] > 20:
                    # Simple scoring: 0-1 based on word count (capped at 300 words)
                    quality_score = min(result["words"] / 300, 1.0)
                    response_quality.append(quality_score)
                    
                    # Speed scoring: response time (lower is better)
                    # Convert to score where 10 seconds -> 0, 0 seconds -> 1
                    speed_score = max(0, 1 - (result["time"] / 10))
                    response_speed.append(speed_score)
            
            # Calculate average scores
            if response_quality:
                scores[f"{category} Quality"] = sum(response_quality) / len(response_quality)
                scores[f"{category} Speed"] = sum(response_speed) / len(response_speed)
            else:
                scores[f"{category} Quality"] = 0.0
                scores[f"{category} Speed"] = 0.0
    
    # Vision capabilities
    if "prompts" in test_results:
        vision_quality = []
        vision_speed = []
        
        for prompt, result in test_results["prompts"].items():
            if "Error:" not in result["response"] and result["words"] > 20:
                quality_score = min(result["words"] / 200, 1.0)
                vision_quality.append(quality_score)
                
                speed_score = max(0, 1 - (result["time"] / 10))
                vision_speed.append(speed_score)
        
        if vision_quality:
            scores["Vision Quality"] = sum(vision_quality) / len(vision_quality)
            scores["Vision Speed"] = sum(vision_speed) / len(vision_speed)
        else:
            scores["Vision Quality"] = 0.0
            scores["Vision Speed"] = 0.0
    
    # Function calling
    if "tool_call_success" in test_results:
        scores["Function Calling"] = 1.0 if test_results["tool_call_success"] else 0.0
    
    # JSON formatting
    if "json_valid" in test_results:
        json_score = 0.0
        if test_results["json_valid"]:
            # Base score for valid JSON
            json_score = 0.5
            # Additional score for field coverage
            if "fields_coverage" in test_results:
                json_score += 0.5 * test_results["fields_coverage"]
        scores["JSON Formatting"] = json_score
    
    return scores

def visualize_model_comparison(models_scores: Dict[str, Dict[str, float]]):
    """Create visualizations to compare model performances"""
    if not models_scores:
        st.warning("No data available for visualization.")
        return
    
    # Find all unique categories
    all_categories = set()
    for model_scores in models_scores.values():
        all_categories.update(model_scores.keys())
    
    # Create a DataFrame for easy visualization
    df_data = []
    for model, scores in models_scores.items():
        model_row = {"Model": model}
        for category in all_categories:
            model_row[category] = scores.get(category, 0.0)
        df_data.append(model_row)
    
    df = pd.DataFrame(df_data)
    
    # Create radar chart of model capabilities
    st.subheader("Model Capabilities Comparison")
    
    # Create grouped bar chart for different categories
    for category_prefix in ["General Knowledge", "Reasoning", "Creative Writing", "Coding", "Vision", "Function", "JSON"]:
        relevant_columns = [col for col in df.columns if col.startswith(category_prefix)]
        if relevant_columns:
            st.subheader(f"{category_prefix} Comparison")
            chart_df = df[["Model"] + relevant_columns]
            st.bar_chart(chart_df, x="Model")
    
    # Create a summary score
    df["Overall Score"] = df.iloc[:, 1:].mean(axis=1)
    st.subheader("Overall Model Performance")
    st.bar_chart(df, x="Model", y="Overall Score")
    
    # Show the raw data
    st.subheader("Raw Scores")
    st.dataframe(df)

def onboarding_test_process():
    """Main function for the model onboarding test process UI"""
    st.title("🚀 Model Onboarding Test Process")
    st.write("Test your AI models' capabilities with this comprehensive onboarding process.")
    
    # Get available models
    available_models = get_available_models()
    
    # Model selection
    selected_models = st.multiselect(
        "Select models to test:",
        available_models,
        key="onboarding_selected_models"
    )
    
    # Test categories selection
    test_categories = st.multiselect(
        "Select test categories:",
        list(TEST_CATEGORIES.keys()),
        default=["General Knowledge", "Reasoning"],
        key="onboarding_test_categories"
    )
    
    # Number of prompts per category
    prompts_per_category = st.slider(
        "Prompts per category:",
        min_value=1,
        max_value=3,
        value=1,
        key="onboarding_prompts_per_category"
    )
    
    # Vision test
    vision_test_enabled = st.checkbox("Include vision capabilities test", value=True)
    
    if vision_test_enabled:
        vision_test_image = st.file_uploader(
            "Upload an image for vision test:",
            type=["jpg", "jpeg", "png"],
            key="onboarding_vision_test_image"
        )
        
        vision_test_prompts = st.multiselect(
            "Select vision test prompts:",
            list(VISION_TEST_PROMPTS.values()),
            default=[VISION_TEST_PROMPTS["Scene Description"]],
            key="onboarding_vision_test_prompts"
        )
    
    # Advanced capabilities tests
    function_call_test = st.checkbox("Test function calling capability", value=True)
    json_format_test = st.checkbox("Test JSON formatting capability", value=True)
    
    if st.button("Start Onboarding Tests", key="start_onboarding_tests"):
        if not selected_models:
            st.warning("Please select at least one model to test.")
            return
            
        if not test_categories and not vision_test_enabled and not function_call_test and not json_format_test:
            st.warning("Please select at least one test category or capability to test.")
            return
            
        # Check vision test requirements
        if vision_test_enabled and not vision_test_image:
            st.warning("Please upload an image for the vision test.")
            return
            
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Store test results
        all_test_results = {}
        model_scores = {}
        
        # Run tests for each model
        for i, model in enumerate(selected_models):
            model_results = {"model": model}
            status_text.text(f"Testing model: {model}")
            
            # Standard test suite
            if test_categories:
                status_text.text(f"Running category tests for {model}...")
                category_results = run_test_suite(model, test_categories, prompts_per_category)
                model_results.update(category_results)
            
            # Vision test
            if vision_test_enabled and vision_test_image:
                status_text.text(f"Running vision tests for {model}...")
                vision_results = run_vision_test(model, vision_test_image, vision_test_prompts)
                model_results["vision_test"] = vision_results
            
            # Function calling test
            if function_call_test:
                status_text.text(f"Testing function calling for {model}...")
                function_results = run_function_call_test(model)
                model_results["function_test"] = function_results
            
            # JSON formatting test
            if json_format_test:
                status_text.text(f"Testing JSON formatting for {model}...")
                json_results = run_json_format_test(model)
                model_results["json_test"] = json_results
            
            # Calculate scores
            model_scores[model] = calculate_model_scores(model_results)
            all_test_results[model] = model_results
            
            # Update progress
            progress = (i + 1) / len(selected_models)
            progress_bar.progress(progress)
        
        # Tests completed
        progress_bar.progress(1.0)
        status_text.text("All tests completed!")
        
        # Visualize results
        st.subheader("Test Results")
        
        # Model comparison
        if len(selected_models) > 1:
            visualize_model_comparison(model_scores)
        
        # Detailed results for each model
        st.subheader("Detailed Results")
        for model, results in all_test_results.items():
            with st.expander(f"Detailed results for {model}"):
                # Show standard test results
                if "categories" in results:
                    for category, category_results in results["categories"].items():
                        st.write(f"### {category}")
                        for i, result in enumerate(category_results):
                            st.write(f"**Prompt {i+1}:** {result['prompt']}")
                            st.write(f"**Response:** {result['response']}")
                            st.write(f"**Time:** {result['time']:.2f} seconds")
                            st.write(f"**Word count:** {result['words']} words")
                            st.write("---")
                
                # Show vision test results
                if "vision_test" in results and "prompts" in results["vision_test"]:
                    st.write("### Vision Test")
                    for prompt, result in results["vision_test"]["prompts"].items():
                        st.write(f"**Prompt:** {prompt}")
                        st.write(f"**Response:** {result['response']}")
                        st.write(f"**Time:** {result['time']:.2f} seconds")
                        st.write(f"**Word count:** {result['words']} words")
                        st.write("---")
                
                # Show function calling test results
                if "function_test" in results:
                    st.write("### Function Calling Test")
                    function_test = results["function_test"]
                    if function_test.get("tool_call_success", False):
                        st.write("✅ Successfully used function calling")
                        st.write(f"**Function:** {function_test.get('function_name', 'N/A')}")
                        st.write(f"**Arguments:** {json.dumps(function_test.get('arguments', {}), indent=2)}")
                        st.write(f"**Final response:** {function_test.get('final_response', 'N/A')}")
                    else:
                        st.write("❌ Failed to use function calling")
                        if "error" in function_test:
                            st.write(f"**Error:** {function_test['error']}")
                        if "model_response" in function_test:
                            st.write(f"**Model response:** {function_test['model_response']}")
                
                # Show JSON formatting test results
                if "json_test" in results:
                    st.write("### JSON Formatting Test")
                    json_test = results["json_test"]
                    if json_test.get("json_valid", False):
                        st.write("✅ Successfully formatted JSON")
                        st.write(f"**Parsed JSON:** {json.dumps(json_test.get('parsed_json', {}), indent=2)}")
                        st.write(f"**Fields coverage:** {json_test.get('fields_coverage', 0) * 100:.1f}%")
                        if json_test.get("missing_fields"):
                            st.write(f"**Missing fields:** {', '.join(json_test['missing_fields'])}")
                        st.write(f"**'interests' is array:** {json_test.get('interests_is_array', False)}")
                    else:
                        st.write("❌ Failed to format valid JSON")
                        if "error" in json_test:
                            st.write(f"**Error:** {json_test['error']}")
                        if "raw_response" in json_test:
                            st.write(f"**Raw response:** {json_test['raw_response']}")
        
        # Save results option
        if st.button("Save Results"):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"model_onboarding_results_{timestamp}.json"
            
            with open(filename, "w") as f:
                json.dump({
                    "test_results": all_test_results,
                    "model_scores": model_scores
                }, f, indent=2)
                
            st.success(f"Results saved to {filename}")

if __name__ == "__main__":
    onboarding_test_process()