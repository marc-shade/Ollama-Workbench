#!/usr/bin/env python3

"""
Test script for debugging tool calling with Ollama models.
This script tests multiple models in parallel to identify which models
properly support tool calling and diagnose serialization issues.
"""

import requests
import json
import concurrent.futures
import logging
import argparse
import os
from typing import Dict, List, Any, Optional
import pytest

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tool_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Test models to try
TEST_MODELS = [
    "llama3",
    "mistral",
    "mistral:instruct",
    "mistral-small3.1:latest",
    "qwen2",
    "phi3",
    "llama3:70b",
    "llama3.1"
]

# Calculator tool definition
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform arithmetic calculations on a given expression or basic operations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (preferred), e.g. '2+2*3'"
                },
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The operation to perform (alternative to expression)"
                },
                "a": {
                    "type": "number",
                    "description": "The first number (use with operation)"
                },
                "b": {
                    "type": "number",
                    "description": "The second number (use with operation)"
                }
            },
            "required": []
        }
    }
}

def calculator_impl(operation: str = None, a: float = None, b: float = None, expression: str = None) -> str:
    """Implementation of the calculator tool

    Supports two modes:
    1. Basic operations with a and b: add, subtract, multiply, divide
    2. Expression evaluation: e.g. "2+2*3"
    """
    logger.info(f"Calculator called with: operation={operation}, a={a}, b={b}, expression={expression}")

    # Mode 1: Expression evaluation (preferred)
    if expression is not None:
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error evaluating expression: {e}"

    # Mode 2: Basic operations
    elif operation is not None and a is not None and b is not None:
        if operation == "add":
            return f"Result: {a + b}"
        elif operation == "subtract":
            return f"Result: {a - b}"
        elif operation == "multiply":
            return f"Result: {a * b}"
        elif operation == "divide":
            if b == 0:
                return "Error: Division by zero"
            return f"Result: {a / b}"
        else:
            return f"Error: Unknown operation '{operation}'"
    else:
        return "Error: Missing required parameters. Either provide 'expression' or all of 'operation', 'a', and 'b'."

def get_available_models() -> List[str]:
    """Get a list of available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            data = response.json()
            models_list = data.get("models", [])
            result = []
            for model in models_list:
                if hasattr(model, 'model'):
                    result.append(model.model)
                elif isinstance(model, dict):
                    result.append(model.get("name", str(model)))
                else:
                    result.append(str(model))
            return result
        else:
            logger.error(f"Error getting models: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Exception getting models: {e}")
        return []

# Test tool definition for MCP tools
MCP_TOOL = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files and directories within the specified directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path of the directory to list contents for"
                }
            },
            "required": [
                "path"
            ]
        }
    }
}

@pytest.mark.parametrize("model_name", TEST_MODELS)
def test_model_with_tools(model_name: str, test_prompt: str = "What is 342 multiplied by 15?") -> Dict[str, Any]:
    """Test a specific model with tool calling"""
    try:
        logger.info(f"Testing model: {model_name}")

        # Common tools list with calculator
        tools = [CALCULATOR_TOOL]

        # Initial messages with user query
        messages = [
            {"role": "user", "content": test_prompt}
        ]

        # First API call to get tool calls
        logger.info(f"Making initial API call to {model_name}")
        response1 = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model_name,
                "messages": messages,
                "tools": tools,
                "stream": False
            },
            timeout=30
        )

        # Check for HTTP errors
        if response1.status_code != 200:
            logger.error(f"HTTP error {response1.status_code} from API: {response1.text}")
            return {
                "model": model_name,
                "status": "error",
                "error": f"HTTP error {response1.status_code}",
                "details": response1.text
            }

        # Parse the JSON response
        try:
            data1 = response1.json()
            logger.info(f"Received response from {model_name}: {json.dumps(data1.get('message', {}), indent=2)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}, Response: {response1.text[:500]}")
            return {
                "model": model_name,
                "status": "error",
                "error": "JSON decode error",
                "details": str(e)
            }

        # Check if the model wants to use tools
        tool_calls = data1.get("message", {}).get("tool_calls", [])
        if not tool_calls:
            # Model didn't use tools, just returned an answer directly
            answer = data1.get("message", {}).get("content", "")
            logger.info(f"Model {model_name} didn't use tools, direct answer: {answer}")
            return {
                "model": model_name,
                "status": "direct_answer",
                "answer": answer,
                "supports_tools": False
            }

        # Model used tools - process each tool call
        logger.info(f"Model {model_name} requested tool calls: {json.dumps(tool_calls, indent=2)}")

        # New messages list including the tool calls and responses
        follow_up_messages = messages.copy()

        # Add the assistant's message with tool calls
        follow_up_messages.append({
            "role": "assistant",
            "content": data1["message"].get("content", ""),
            "tool_calls": tool_calls
        })

        # Process each tool call
        for call in tool_calls:
            func_name = call["function"]["name"]
            func_args = call["function"]["arguments"]

            # Ensure arguments is a dictionary
            if isinstance(func_args, str):
                try:
                    func_args = json.loads(func_args)
                except json.JSONDecodeError:
                    func_args = {"error": "Failed to parse arguments"}

            # Only handle calculator tool in this test
            if func_name == "calculator":
                # Call the calculator implementation
                try:
                    result = calculator_impl(**func_args)
                    logger.info(f"Calculator result: {result}")
                except Exception as e:
                    result = f"Error executing calculator: {str(e)}"
                    logger.error(f"Calculator error: {e}")

                # Add the tool response
                # Check if the call has an "id" field, if not use index or generic ID
                tool_call_id = call.get("id", f"call_{func_name}_{hash(str(func_args))}")

                follow_up_messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call_id
                })

        # Make the follow-up API call with tool results
        logger.info(f"Making follow-up API call to {model_name}")
        response2 = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model_name,
                "messages": follow_up_messages,
                "tools": tools,  # Include tools again
                "stream": False
            },
            timeout=30
        )

        # Check for HTTP errors in follow-up
        if response2.status_code != 200:
            logger.error(f"HTTP error {response2.status_code} in follow-up: {response2.text}")
            return {
                "model": model_name,
                "status": "error_in_followup",
                "error": f"HTTP error {response2.status_code}",
                "details": response2.text
            }

        # Parse the follow-up JSON response
        try:
            data2 = response2.json()
            logger.info(f"Received follow-up from {model_name}: {json.dumps(data2.get('message', {}), indent=2)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in follow-up: {e}")
            return {
                "model": model_name,
                "status": "error_in_followup",
                "error": "JSON decode error",
                "details": str(e)
            }

        # Check the final answer
        final_answer = data2.get("message", {}).get("content", "")
        logger.info(f"Final answer from {model_name}: {final_answer}")

        # Check if there are more tool calls (some models might chain multiple tools)
        more_tool_calls = data2.get("message", {}).get("tool_calls", [])

        return {
            "model": model_name,
            "status": "success",
            "initial_response": data1,
            "follow_up_response": data2,
            "tool_calls": tool_calls,
            "final_answer": final_answer,
            "more_tool_calls": more_tool_calls,
            "supports_tools": True
        }

    except Exception as e:
        logger.error(f"Error testing {model_name}: {str(e)}")
        return {
            "model": model_name,
            "status": "exception",
            "error": str(e),
            "supports_tools": False
        }

# Test tool definition for MCP tools
MCP_TOOL = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files and directories within the specified directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path of the directory to list contents for"
                }
            },
            "required": [
                "path"
            ]
        }
    }
}

@pytest.mark.parametrize("model_name", TEST_MODELS)
def test_model_with_mcp_tool(model_name: str, test_prompt: str = "List files in the current directory") -> Dict[str, Any]:
    """Test a specific model with tool calling"""
    try:
        logger.info(f"Testing model: {model_name} with MCP Tool")

        # Common tools list with calculator
        tools = [MCP_TOOL]

        # Initial messages with user query
        messages = [
            {"role": "user", "content": test_prompt, "path": "."}
        ]

        # First API call to get tool calls
        logger.info(f"Making initial API call to {model_name}")
        response1 = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model_name,
                "messages": messages,
                "tools": tools,
                "stream": False
            },
            timeout=30
        )

        # Check for HTTP errors
        if response1.status_code != 200:
            logger.error(f"HTTP error {response1.status_code} from API: {response1.text}")
            return {
                "model": model_name,
                "status": "error",
                "error": f"HTTP error {response1.status_code}",
                "details": response1.text
            }

        # Parse the JSON response
        try:
            data1 = response1.json()
            logger.info(f"Received response from {model_name}: {json.dumps(data1.get('message', {}), indent=2)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}, Response: {response1.text[:500]}")
            return {
                "model": model_name,
                "status": "error",
                "error": "JSON decode error",
                "details": str(e)
            }

        # Check if the model wants to use tools
        tool_calls = data1.get("message", {}).get("tool_calls", [])
        if not tool_calls:
            # Model didn't use tools, just returned an answer directly
            answer = data1.get("message", {}).get("content", "")
            logger.info(f"Model {model_name} didn't use tools, direct answer: {answer}")
            return {
                "model": model_name,
                "status": "direct_answer",
                "answer": answer,
                "supports_tools": False
            }

        # Model used tools - process each tool call
        logger.info(f"Model {model_name} requested tool calls: {json.dumps(tool_calls, indent=2)}")

        # New messages list including the tool calls and responses
        follow_up_messages = messages.copy()

        # Add the assistant's message with tool calls
        follow_up_messages.append({
            "role": "assistant",
            "content": data1["message"].get("content", ""),
            "tool_calls": tool_calls
        })

        # Process each tool call
        for call in tool_calls:
            func_name = call["function"]["name"]
            func_args = call["function"]["arguments"]

            # Ensure arguments is a dictionary
            if isinstance(func_args, str):
                try:
                    func_args = json.loads(func_args)
                except json.JSONDecodeError:
                    func_args = {"error": "Failed to parse arguments"}

            # Only handle calculator tool in this test
            if func_name == "list_files":
                # Call the calculator implementation
                try:
                    # Assuming the tool returns a list of files
                    # Replace this with actual MCP tool call
                    result = "MCP Tool Result: file1.txt, file2.txt"
                    logger.info(f"MCP Tool result: {result}")
                except Exception as e:
                    result = f"Error executing MCP Tool: {str(e)}"
                    logger.error(f"MCP Tool error: {e}")

                # Add the tool response
                # Check if the call has an "id" field, if not use index or generic ID
                tool_call_id = call.get("id", f"call_{func_name}_{hash(str(func_args))}")

                follow_up_messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call_id
                })

        # Make the follow-up API call with tool results
        logger.info(f"Making follow-up API call to {model_name}")
        response2 = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model_name,
                "messages": follow_up_messages,
                "tools": tools,  # Include tools again
                "stream": False
            },
            timeout=30
        )

        # Check for HTTP errors in follow-up
        if response2.status_code != 200:
            logger.error(f"HTTP error {response2.status_code} in follow-up: {response2.text}")
            return {
                "model": model_name,
                "status": "error_in_followup",
                "error": f"HTTP error {response2.status_code}",
                "details": response2.text
            }

        # Parse the follow-up JSON response
        try:
            data2 = response2.json()
            logger.info(f"Received follow-up from {model_name}: {json.dumps(data2.get('message', {}), indent=2)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in follow-up: {e}")
            return {
                "model": model_name,
                "status": "error_in_followup",
                "error": "JSON decode error",
                "details": str(e)
            }

        # Check the final answer
        final_answer = data2.get("message", {}).get("content", "")
        logger.info(f"Final answer from {model_name}: {final_answer}")

        # Check if there are more tool calls (some models might chain multiple tools)
        more_tool_calls = data2.get("message", {}).get("tool_calls", [])

        return {
            "model": model_name,
            "status": "success",
            "initial_response": data1,
            "follow_up_response": data2,
            "tool_calls": tool_calls,
            "final_answer": final_answer,
            "more_tool_calls": more_tool_calls,
            "supports_tools": True
        }

    except Exception as e:
        logger.error(f"Error testing {model_name}: {str(e)}")
        return {
            "model": model_name,
            "status": "exception",
            "error": str(e),
            "supports_tools": False
        }


def main():
    parser = argparse.ArgumentParser(description="Test tool calling with Ollama models")
    parser.add_argument("--model", type=str, help="Test only this specific model")
    parser.add_argument("--prompt", type=str, default="What is 342 multiplied by 15?",
                        help="Test prompt to use")
    parser.add_argument("--parallel", action="store_true", help="Test models in parallel")
    parser.add_argument("--all", action="store_true", help="Test all available models")
    parser.add_argument("--test_mcp", action="store_true", help="Test MCP tool calling")
    args = parser.parse_args()

    # Create a results directory
    results_dir = os.path.join(os.path.dirname(__file__), "test_results")
    os.makedirs(results_dir, exist_ok=True)

    # Determine which models to test
    models_to_test = []

    if args.model:
        # Test a single specified model
        models_to_test = [args.model]
    elif args.all:
        # Test all available models
        available_models = get_available_models()
        models_to_test = available_models
        logger.info(f"Testing all {len(available_models)} available models")
    else:
        # Test the predefined list of models
        models_to_test = TEST_MODELS
        logger.info(f"Testing {len(models_to_test)} predefined models")

    results = {}

    test_function = test_model_with_mcp_tool if args.test_mcp else test_model_with_tools

    if args.parallel and len(models_to_test) > 1:
        # Test models in parallel
        logger.info(f"Testing {len(models_to_test)} models in parallel")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(models_to_test), 4)) as executor:
            # Create a mapping of model to future
            future_to_model = {
                executor.submit(test_function, model, args.prompt): model
                for model in models_to_test
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    result = future.result()
                    results[model] = result
                except Exception as e:
                    logger.error(f"Error with model {model}: {e}")
                    results[model] = {
                        "model": model,
                        "status": "exception",
                        "error": str(e)
                    }
    else:
        # Test models sequentially
        logger.info(f"Testing {len(models_to_test)} models sequentially")
        for model in models_to_test:
            results[model] = test_function(model, args.prompt) if args.test_mcp else test_model_with_tools(model, args.prompt)

    # Save all results to a JSON file
    results_file = os.path.join(results_dir, "tool_calling_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n\nTOOL CALLING TEST RESULTS SUMMARY:")
    print("================================")

    success_models = []
    for model, result in results.items():
        status = result.get("status", "unknown")
        if status == "success":
            success_models.append(model)
            answer = result.get("final_answer", "").strip().replace("\n", " ")
            print(f"✅ {model}: SUCCESS - Answer: {answer[:60]}...")
        elif status == "direct_answer":
            answer = result.get("answer", "").strip().replace("\n", " ")
            print(f"⚠️ {model}: DIRECT ANSWER (didn't use tools) - {answer[:60]}...")
        else:
            error = result.get("error", "unknown error")
            print(f"❌ {model}: FAILED - {error}")

    print("\nSuccessful models:")
    for model in success_models:
        print(f"  - {model}")

    print(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    main()