import streamlit as st
import ollama
import json
import time
import pandas as pd
import plotly.express as px
import re
from typing import Dict, Any, List, Optional, Tuple
from ollama_workbench.providers.ollama_utils import get_ollama_client, get_local_models
from ollama_workbench.core.error_handling import handle_api_error, capture_exceptions
from ollama_workbench.core.config import CONFIG

# Define model capability categories and tests
MODEL_CAPABILITIES = {
    "Basic": [
        {
            "name": "Text Generation",
            "description": "Generate coherent text from a prompt",
            "test_prompt": "Write a short poem about the moon.",
            "evaluation": "success_if_longer_than",
            "threshold": 50
        },
        {
            "name": "Context Understanding",
            "description": "Understand and respond to context in a conversation",
            "test_prompt": "The sun is a star. What is the sun?",
            "evaluation": "contains_any",
            "keywords": ["star", "celestial", "body", "center", "solar", "system"]
        }
    ],
    "Reasoning": [
        {
            "name": "Arithmetic",
            "description": "Perform basic arithmetic calculations",
            "test_prompt": "What is 42 + 28?",
            "evaluation": "contains",
            "keywords": ["70"]
        },
        {
            "name": "Logic",
            "description": "Apply logical reasoning to solve problems",
            "test_prompt": "If all apples are fruits, and no fruits are vegetables, are apples vegetables?",
            "evaluation": "contains_any",
            "keywords": ["no", "not", "false", "incorrect"]
        },
        {
            "name": "Multi-step Reasoning",
            "description": "Solve problems requiring multiple reasoning steps",
            "test_prompt": "John has 5 apples. He gives 2 to Mary, who then gives 1 to Tom. How many apples does John have now?",
            "evaluation": "contains",
            "keywords": ["3"]
        }
    ],
    "Language": [
        {
            "name": "Grammar",
            "description": "Generate text with correct grammar",
            "test_prompt": "Write a grammatically correct sentence using the words: dog, chase, cat, tree.",
            "evaluation": "grammar_check"
        },
        {
            "name": "Summarization",
            "description": "Summarize longer text into key points",
            "test_prompt": "Summarize the following in 2-3 sentences: The Apollo 11 mission was the first manned mission to land on the Moon. It was launched on July 16, 1969, and carried Commander Neil Armstrong, Command Module Pilot Michael Collins, and Lunar Module Pilot Edwin 'Buzz' Aldrin. On July 20, Armstrong and Aldrin became the first humans to land on the Moon, while Collins orbited the Moon. The mission fulfilled President Kennedy's goal of reaching the Moon before the end of the 1960s.",
            "evaluation": "success_if_between",
            "min_length": 50,
            "max_length": 300
        }
    ],
    "Knowledge": [
        {
            "name": "General Knowledge",
            "description": "Demonstrate knowledge of common facts",
            "test_prompt": "What is the capital of France?",
            "evaluation": "contains_any",
            "keywords": ["Paris"]
        },
        {
            "name": "Science",
            "description": "Demonstrate knowledge of scientific concepts",
            "test_prompt": "Explain what photosynthesis is.",
            "evaluation": "contains_all",
            "keywords": ["plant", "light", "energy", "carbon", "dioxide", "oxygen"]
        }
    ],
    "Coding": [
        {
            "name": "Code Generation",
            "description": "Generate working code from a description",
            "test_prompt": "Write a Python function to check if a number is prime.",
            "evaluation": "contains_all",
            "keywords": ["def", "return", "for", "if", "prime"]
        },
        {
            "name": "Code Explanation",
            "description": "Explain what a code snippet does",
            "test_prompt": "Explain what this code does: for i in range(1, 101): if i % 3 == 0 and i % 5 == 0: print('FizzBuzz') elif i % 3 == 0: print('Fizz') elif i % 5 == 0: print('Buzz') else: print(i)",
            "evaluation": "contains_all",
            "keywords": ["fizz", "buzz", "divisible", "3", "5", "print"]
        }
    ],
    "Multimodal": [
        {
            "name": "Image Understanding",
            "description": "Understand and describe image content",
            "test_type": "image",
            "evaluation": "manual"
        },
        {
            "name": "Charts Analysis",
            "description": "Analyze and extract information from charts and graphs",
            "test_type": "image",
            "image_type": "chart",
            "evaluation": "manual"
        }
    ],
    "Function Calling": [
        {
            "name": "Tool Use",
            "description": "Use tools/functions when appropriate",
            "test_prompt": "Calculate the square root of 144.",
            "evaluation": "contains",
            "keywords": ["12"]
        },
        {
            "name": "JSON Output",
            "description": "Generate properly formatted JSON data",
            "test_prompt": "Generate a JSON object with fields for name, age, and city for a person named John who is 30 years old and lives in New York.",
            "evaluation": "valid_json_with_keys",
            "required_keys": ["name", "age", "city"]
        }
    ]
}

def evaluate_response(response: str, test: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Evaluate a model response against a test.
    
    Args:
        response: The model's response
        test: Test definition
        
    Returns:
        Tuple[bool, str]: (success, reason)
    """
    eval_type = test.get("evaluation", "manual")
    
    if eval_type == "manual":
        # Manual evaluation requires human judgment
        return None, "Requires manual evaluation"
    
    elif eval_type == "success_if_longer_than":
        threshold = test.get("threshold", 10)
        if len(response) > threshold:
            return True, f"Response length ({len(response)}) exceeds threshold ({threshold})"
        else:
            return False, f"Response too short: {len(response)}/{threshold} characters"
    
    elif eval_type == "success_if_between":
        min_length = test.get("min_length", 10)
        max_length = test.get("max_length", 1000)
        if min_length <= len(response) <= max_length:
            return True, f"Response length ({len(response)}) is within range ({min_length}-{max_length})"
        else:
            return False, f"Response length ({len(response)}) outside range ({min_length}-{max_length})"
    
    elif eval_type == "contains":
        keywords = test.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [keywords]
        
        response_lower = response.lower()
        all_found = all(keyword.lower() in response_lower for keyword in keywords)
        
        if all_found:
            return True, f"Response contains all required keywords: {', '.join(keywords)}"
        else:
            missing = [k for k in keywords if k.lower() not in response_lower]
            return False, f"Missing keywords: {', '.join(missing)}"
    
    elif eval_type == "contains_any":
        keywords = test.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [keywords]
        
        response_lower = response.lower()
        found = any(keyword.lower() in response_lower for keyword in keywords)
        
        if found:
            found_keywords = [k for k in keywords if k.lower() in response_lower]
            return True, f"Response contains at least one keyword: {', '.join(found_keywords)}"
        else:
            return False, f"None of the keywords found: {', '.join(keywords)}"
    
    elif eval_type == "contains_all":
        keywords = test.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [keywords]
        
        response_lower = response.lower()
        all_found = all(keyword.lower() in response_lower for keyword in keywords)
        
        if all_found:
            return True, f"Response contains all required keywords: {', '.join(keywords)}"
        else:
            missing = [k for k in keywords if k.lower() not in response_lower]
            return False, f"Missing keywords: {', '.join(missing)}"
    
    elif eval_type == "grammar_check":
        # Very basic grammar check - could be improved
        sentences = re.split(r'[.!?]+', response)
        valid_sentences = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check for capitalization at the beginning
            if sentence[0].isupper():
                valid_sentences += 1
        
        if valid_sentences > 0:
            return True, f"Found {valid_sentences} grammatically valid sentences"
        else:
            return False, "No grammatically valid sentences found"
    
    elif eval_type == "valid_json_with_keys":
        required_keys = test.get("required_keys", [])
        
        # Extract JSON from response
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without code block
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                return False, "No JSON found in response"
        
        try:
            json_obj = json.loads(json_str)
            missing_keys = [key for key in required_keys if key not in json_obj]
            
            if not missing_keys:
                return True, f"Valid JSON with all required keys: {', '.join(required_keys)}"
            else:
                return False, f"JSON missing keys: {', '.join(missing_keys)}"
        except json.JSONDecodeError:
            return False, "Invalid JSON format"
    
    else:
        return None, f"Unknown evaluation type: {eval_type}"

@capture_exceptions
def test_model_capability(model: str, test: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test a specific capability of a model.
    
    Args:
        model: Model name
        test: Test definition
        
    Returns:
        Dict[str, Any]: Test results
    """
    client = get_ollama_client()
    
    # Start timing
    start_time = time.time()
    
    # Handle test type
    test_type = test.get("test_type", "text")
    
    if test_type == "text":
        # Text prompt test
        prompt = test.get("test_prompt", "")
        
        # Use version-independent call_ollama_endpoint function
        from ollama_workbench.providers.ollama_utils import call_ollama_endpoint
        response_text, _, _, _, _ = call_ollama_endpoint(
            model=model,
            prompt=prompt,
            temperature=0.7,
            max_tokens=256
        )
    elif test_type == "image":
        # Image test requires manual testing
        response_text = "Image testing requires manual evaluation"
    else:
        response_text = f"Unknown test type: {test_type}"
    
    # Calculate time
    elapsed_time = time.time() - start_time
    
    # Evaluate response
    success, reason = evaluate_response(response_text, test)
    
    # Return results
    result = {
        "model": model,
        "capability": test.get("name", "Unknown"),
        "description": test.get("description", ""),
        "prompt": test.get("test_prompt", ""),
        "response": response_text,
        "success": success,
        "reason": reason,
        "time": elapsed_time
    }
    
    return result

def batch_test_model_capabilities(model: str, categories: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Test multiple capabilities of a model.
    
    Args:
        model: Model name
        categories: Optional list of capability categories to test
        
    Returns:
        Dict[str, List[Dict[str, Any]]]: Test results by category
    """
    results = {}
    
    for category, tests in MODEL_CAPABILITIES.items():
        if categories and category not in categories:
            continue
        
        category_results = []
        
        for test in tests:
            # Skip image tests for automatic testing
            if test.get("test_type") == "image":
                continue
                
            result = test_model_capability(model, test)
            category_results.append(result)
        
        results[category] = category_results
    
    return results

def model_capabilities_ui():
    """UI for discovering model capabilities."""
    st.title("🔎 Model Capabilities Discovery")
    st.write("Discover and compare the capabilities of different models")
    
    # Get available models
    local_models = get_local_models()
    # Handle both Model objects (v0.4.8+) and dicts
    model_names = []
    for model in local_models:
        if hasattr(model, 'model'):
            model_names.append(model.model)
        elif isinstance(model, dict):
            model_names.append(model.get("name", str(model)))
        else:
            model_names.append(str(model))
    
    if not model_names:
        st.warning("No models found. Please pull some models first.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Discover Capabilities", "Compare Models", "Detailed Tests"])
    
    with tab1:
        st.subheader("Discover Model Capabilities")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            model_names,
            key="capability_model_selector"
        )
        
        # Category selection
        selected_categories = st.multiselect(
            "Select Categories to Test",
            list(MODEL_CAPABILITIES.keys()),
            default=["Basic", "Reasoning"],
            key="capability_category_selector"
        )
        
        # Test button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Test Capabilities", key="test_capabilities_button"):
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Calculate total tests
                total_tests = sum(
                    len([t for t in tests if t.get("test_type") != "image"])
                    for category, tests in MODEL_CAPABILITIES.items()
                    if category in selected_categories
                )
                
                if total_tests == 0:
                    st.warning("No tests selected.")
                    return
                
                # Initialize results in session state
                if "capability_results" not in st.session_state:
                    st.session_state.capability_results = {}
                
                # Create model entry if it doesn't exist
                if selected_model not in st.session_state.capability_results:
                    st.session_state.capability_results[selected_model] = {}
                
                # Run tests
                completed = 0
                
                for category, tests in MODEL_CAPABILITIES.items():
                    if category not in selected_categories:
                        continue
                    
                    category_results = []
                    
                    for test in tests:
                        # Skip image tests for automatic testing
                        if test.get("test_type") == "image":
                            continue
                            
                        # Update progress
                        status_text.text(f"Testing: {category} - {test.get('name')}")
                        
                        # Run test
                        result = test_model_capability(selected_model, test)
                        category_results.append(result)
                        
                        # Update progress
                        completed += 1
                        progress_bar.progress(completed / total_tests)
                    
                    # Store results
                    st.session_state.capability_results[selected_model][category] = category_results
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
                
                st.success("Capability testing completed!")
                
                # Force re-render
                st.rerun()
        
        # Show results if available
        if "capability_results" in st.session_state and selected_model in st.session_state.capability_results:
            model_results = st.session_state.capability_results[selected_model]
            
            # Calculate success rates by category
            category_success = {}
            for category, results in model_results.items():
                if not results:
                    continue
                
                success_count = sum(1 for r in results if r.get("success") is True)
                total_count = len(results)
                if total_count > 0:
                    success_rate = success_count / total_count
                else:
                    success_rate = 0
                
                category_success[category] = {
                    "success_rate": success_rate,
                    "success_count": success_count,
                    "total_count": total_count
                }
            
            # Show summary as bar chart
            if category_success:
                st.subheader("Capability Summary")
                
                # Prepare data for chart
                chart_data = {
                    "Category": [],
                    "Success Rate": []
                }
                
                for category, data in category_success.items():
                    if category not in selected_categories:
                        continue
                    
                    chart_data["Category"].append(category)
                    chart_data["Success Rate"].append(data["success_rate"] * 100)
                
                # Create DataFrame
                df = pd.DataFrame(chart_data)
                
                # Create bar chart
                fig = px.bar(
                    df,
                    x="Category",
                    y="Success Rate",
                    title=f"Capability Success Rates for {selected_model}",
                    labels={"Category": "Capability Category", "Success Rate": "Success Rate (%)"},
                    color="Success Rate",
                    color_continuous_scale="RdYlGn",
                    range_color=[0, 100]
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed results by category
            for category in selected_categories:
                if category not in model_results:
                    continue
                
                results = model_results[category]
                if not results:
                    continue
                
                with st.expander(f"{category} Capabilities", expanded=True):
                    for result in results:
                        # Determine status icon
                        if result.get("success") is True:
                            status_icon = "✅"
                        elif result.get("success") is False:
                            status_icon = "❌"
                        else:
                            status_icon = "⚠️"
                        
                        # Create a card-like display
                        st.markdown(f"### {status_icon} {result.get('capability')}")
                        st.write(result.get("description", ""))
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Prompt:**")
                            st.markdown(f"> {result.get('prompt', '')}")
                        
                        with col2:
                            st.markdown("**Evaluation:**")
                            st.markdown(f"> {result.get('reason', '')}")
                        
                        # Show response in expandable section
                        with st.expander("See Response"):
                            st.markdown(result.get("response", ""))
                        
                        st.markdown("---")
    
    with tab2:
        st.subheader("Compare Model Capabilities")
        
        # Model selection for comparison
        selected_models = st.multiselect(
            "Select Models to Compare",
            model_names,
            key="capability_model_comparison"
        )
        
        # Category selection for comparison
        comparison_categories = st.multiselect(
            "Select Categories for Comparison",
            list(MODEL_CAPABILITIES.keys()),
            default=["Basic", "Reasoning"],
            key="capability_comparison_categories"
        )
        
        # Run comparison
        if st.button("Compare Models", key="compare_capabilities_button"):
            if not selected_models:
                st.warning("Please select at least one model to compare.")
                return
            
            if not comparison_categories:
                st.warning("Please select at least one category to compare.")
                return
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Calculate total tests
            total_tests = len(selected_models) * sum(
                len([t for t in tests if t.get("test_type") != "image"])
                for category, tests in MODEL_CAPABILITIES.items()
                if category in comparison_categories
            )
            
            if total_tests == 0:
                st.warning("No tests selected.")
                return
            
            # Initialize results in session state
            if "capability_results" not in st.session_state:
                st.session_state.capability_results = {}
            
            # Run tests for each model
            completed = 0
            
            for model in selected_models:
                # Create model entry if it doesn't exist
                if model not in st.session_state.capability_results:
                    st.session_state.capability_results[model] = {}
                
                for category, tests in MODEL_CAPABILITIES.items():
                    if category not in comparison_categories:
                        continue
                    
                    category_results = []
                    
                    for test in tests:
                        # Skip image tests for automatic testing
                        if test.get("test_type") == "image":
                            continue
                            
                        # Update progress
                        status_text.text(f"Testing: {model} - {category} - {test.get('name')}")
                        
                        # Check if we already have results
                        if (category in st.session_state.capability_results[model] and
                            any(r.get("capability") == test.get("name") for r in st.session_state.capability_results[model][category])):
                            # Use existing results
                            for r in st.session_state.capability_results[model][category]:
                                if r.get("capability") == test.get("name"):
                                    result = r
                                    break
                        else:
                            # Run test
                            result = test_model_capability(model, test)
                        
                        category_results.append(result)
                        
                        # Update progress
                        completed += 1
                        progress_bar.progress(completed / total_tests)
                    
                    # Store results
                    st.session_state.capability_results[model][category] = category_results
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()
            
            st.success("Comparison completed!")
            
            # Force re-render
            st.rerun()
        
        # Show comparison if results available
        if "capability_results" in st.session_state and any(model in st.session_state.capability_results for model in selected_models):
            # Calculate success rates for each model
            comparison_data = {}
            
            for model in selected_models:
                if model not in st.session_state.capability_results:
                    continue
                
                model_results = st.session_state.capability_results[model]
                model_data = {}
                
                for category in comparison_categories:
                    if category not in model_results:
                        continue
                    
                    results = model_results[category]
                    if not results:
                        continue
                    
                    success_count = sum(1 for r in results if r.get("success") is True)
                    total_count = len(results)
                    
                    if total_count > 0:
                        success_rate = success_count / total_count * 100
                    else:
                        success_rate = 0
                    
                    model_data[category] = success_rate
                
                comparison_data[model] = model_data
            
            # Prepare data for chart
            chart_data = {
                "Category": [],
                "Model": [],
                "Success Rate": []
            }
            
            for model, categories in comparison_data.items():
                for category, success_rate in categories.items():
                    chart_data["Category"].append(category)
                    chart_data["Model"].append(model)
                    chart_data["Success Rate"].append(success_rate)
            
            if chart_data["Category"]:
                # Create DataFrame
                df = pd.DataFrame(chart_data)
                
                # Create bar chart
                fig = px.bar(
                    df,
                    x="Category",
                    y="Success Rate",
                    color="Model",
                    barmode="group",
                    title="Model Capability Comparison",
                    labels={"Category": "Capability Category", "Success Rate": "Success Rate (%)"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a comparison table
                st.subheader("Detailed Comparison")
                
                # Prepare table data
                table_data = {"Category": [], "Capability": []}
                for model in selected_models:
                    if model in st.session_state.capability_results:
                        table_data[model] = []
                
                # Populate table data
                for category in comparison_categories:
                    tests = MODEL_CAPABILITIES.get(category, [])
                    for test in tests:
                        # Skip image tests
                        if test.get("test_type") == "image":
                            continue
                            
                        capability = test.get("name", "")
                        table_data["Category"].append(category)
                        table_data["Capability"].append(capability)
                        
                        for model in selected_models:
                            if model not in st.session_state.capability_results:
                                table_data[model].append("Not Tested")
                                continue
                                
                            model_results = st.session_state.capability_results[model]
                            if category not in model_results:
                                table_data[model].append("Not Tested")
                                continue
                                
                            results = model_results[category]
                            found = False
                            
                            for result in results:
                                if result.get("capability") == capability:
                                    if result.get("success") is True:
                                        table_data[model].append("✅ Pass")
                                    elif result.get("success") is False:
                                        table_data[model].append("❌ Fail")
                                    else:
                                        table_data[model].append("⚠️ Unknown")
                                    found = True
                                    break
                            
                            if not found:
                                table_data[model].append("Not Tested")
                
                # Create and style the DataFrame
                comparison_df = pd.DataFrame(table_data)
                
                # Convert to styled HTML for colored cells
                def color_cells(val):
                    if val == "✅ Pass":
                        return "background-color: #a8f0c6"
                    elif val == "❌ Fail":
                        return "background-color: #f7a7a3"
                    elif val == "⚠️ Unknown":
                        return "background-color: #ffd966"
                    else:
                        return ""
                
                styled_df = comparison_df.style.applymap(color_cells, subset=[model for model in selected_models])
                
                st.dataframe(styled_df, use_container_width=True)
    
    with tab3:
        st.subheader("Run Detailed Tests")
        
        # Model and test selection
        col1, col2 = st.columns(2)
        
        with col1:
            detailed_model = st.selectbox(
                "Select Model",
                model_names,
                key="detailed_model_selector"
            )
        
        with col2:
            # Flatten the capabilities for selection
            all_tests = []
            for category, tests in MODEL_CAPABILITIES.items():
                for test in tests:
                    test_name = f"{category}: {test.get('name')}"
                    all_tests.append((category, test_name, test))
            
            selected_test_name = st.selectbox(
                "Select Test",
                [t[1] for t in all_tests],
                key="detailed_test_selector"
            )
            
            # Find selected test
            selected_test = next((t[2] for t in all_tests if t[1] == selected_test_name), None)
            selected_category = next((t[0] for t in all_tests if t[1] == selected_test_name), None)
        
        # Run test
        if st.button("Run Test", key="run_detailed_test"):
            if not detailed_model or not selected_test:
                st.warning("Please select a model and test.")
                return
            
            # Skip image tests
            if selected_test.get("test_type") == "image":
                st.warning("Image tests require manual evaluation and are not supported in this interface.")
                return
            
            with st.spinner("Running test..."):
                result = test_model_capability(detailed_model, selected_test)
                
                # Store in session state
                if "detailed_test_results" not in st.session_state:
                    st.session_state.detailed_test_results = []
                
                st.session_state.detailed_test_results.append(result)
        
        # Show detailed results
        if "detailed_test_results" in st.session_state and st.session_state.detailed_test_results:
            st.subheader("Test Results")
            
            # Show each result in order (most recent first)
            for i, result in enumerate(reversed(st.session_state.detailed_test_results)):
                with st.expander(f"Test #{i+1}: {result.get('capability')} ({result.get('model')})", expanded=i == 0):
                    # Determine status icon
                    if result.get("success") is True:
                        status_icon = "✅"
                    elif result.get("success") is False:
                        status_icon = "❌"
                    else:
                        status_icon = "⚠️"
                    
                    st.markdown(f"### {status_icon} {result.get('capability')}")
                    st.write(result.get("description", ""))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Prompt:**")
                        st.markdown(f"> {result.get('prompt', '')}")
                    
                    with col2:
                        st.markdown("**Model:**")
                        st.markdown(f"> {result.get('model')}")
                        st.markdown("**Time:**")
                        st.markdown(f"> {result.get('time', 0):.2f}s")
                    
                    st.markdown("**Evaluation:**")
                    st.markdown(f"> {result.get('reason', '')}")
                    
                    # Show full response
                    st.markdown("**Response:**")
                    st.markdown(result.get("response", ""))

if __name__ == "__main__":
    model_capabilities_ui()