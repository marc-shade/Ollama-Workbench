# build.py
import os
import re
import json
import sys
import subprocess
import tempfile
import uuid
import shutil
import requests
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Callable
from rich.console import Console
from rich.panel import Panel
import streamlit as st
import ollama
from ollama import Client
from duckduckgo_search import DDGS
from googleapiclient.discovery import build
from serpapi import GoogleSearch
from ollama_utils import get_available_models
from pathlib import Path
import pytest
from pytest_html import extras
from io import StringIO
import ast

API_KEYS_FILE = "api_keys.json"

def load_api_keys():
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_api_keys(api_keys):
    with open(API_KEYS_FILE, 'w') as f:
        json.dump(api_keys, f, indent=4)

# Initialize the Rich Console
console = Console()

# Define Groq models
GROQ_MODELS = [
    "llama-3.1-405b-reasoning",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "llama3-groq-8b-8192-tool-use-preview",
    "llama3-groq-70b-8192-tool-use-preview",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
    "gemma2-9b-it"
]

ADVANCED_GROQ_MODELS = [
    "llama-3.1-405b-reasoning",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant"
]

SETTINGS_FILE = "build_settings.json"

# Initialize the Ollama client
client = Client(host='http://localhost:11434')

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)

def is_groq_model(model_name):
    return model_name in GROQ_MODELS

def is_advanced_groq_model(model_name):
    return model_name in ADVANCED_GROQ_MODELS

def get_all_models(settings, include_advanced=False):
    try:
        ollama_models = get_available_models()
    except Exception as e:
        console.print(f"Error getting Ollama models: {e}", style="bold red")
        ollama_models = []
    
    groq_models = []
    if settings.get("groq_api_key"):
        groq_models = GROQ_MODELS if include_advanced else [m for m in GROQ_MODELS if m not in ADVANCED_GROQ_MODELS]
    
    return ollama_models + groq_models

def call_groq_api(model, messages, temperature, max_tokens, groq_api_key, retries=5, initial_delay=5, backoff_factor=2):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": min(max_tokens, 8192 if model != "llama-3.1-405b-reasoning" else 16384)
    }
    
    delay = initial_delay
    for attempt in range(retries):
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.HTTPError as http_err:
            error_message = f"HTTP error occurred: {http_err}"
            if response.status_code == 404:
                error_message += "\nPossible reasons for 404 error:\n"
                error_message += "1. The API endpoint URL might be incorrect\n"
                error_message += "2. The model name might be invalid or not available\n"
                error_message += "3. Your API key might not have access to this model\n"
                error_message += f"Current model: {model}\n"
                error_message += f"Current URL: {url}\n"
            elif response.status_code in (502, 429):
                console.print(Panel(f"Attempt {attempt+1} failed with {response.status_code}. Retrying in {delay} seconds...", title="[bold yellow]HTTP Error[/bold yellow]", title_align="left", border_style="yellow"))
                time.sleep(delay)
                delay *= backoff_factor
                continue
            console.print(Panel(error_message, title="[bold red]HTTP Error[/bold red]", title_align="left", border_style="red"))
            raise
        except requests.exceptions.RequestException as err:
            console.print(Panel(f"An error occurred: {err}", title="[bold red]Request Error[/bold red]", title_align="left", border_style="red"))
            raise
        except Exception as err:
            console.print(Panel(f"An unexpected error occurred: {err}", title="[bold red]Unexpected Error[/bold red]", title_align="left", border_style="red"))
            raise
    
    raise requests.exceptions.HTTPError("Failed to connect to the server after several attempts.")

def call_model(model, messages, temperature, max_tokens, groq_api_key=None):
    if is_groq_model(model):
        return call_groq_api(model, messages, temperature, max_tokens, groq_api_key)
    else:
        response = client.chat(
            model=model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )
        return response['message']['content']

def duckduckgo_search(query: str, num_results: int = 5) -> List[Dict]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))
    return [{"title": result["title"], "url": result["href"]} for result in results]

def google_search(query: str, api_key: str, cse_id: str, num_results: int = 5) -> List[Dict]:
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
    return [{"title": item["title"], "url": item["link"]} for item in res.get("items", [])]

def serpapi_search(query: str, api_key: str, num_results: int = 5) -> List[Dict]:
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": num_results
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        if not organic_results:
            print(f"SerpApi returned no results. Full response: {results}")
        return [{"title": result["title"], "url": result["link"]} for result in organic_results]
    except Exception as e:
        print(f"Error in SerpApi search: {str(e)}")
        return []

def perform_search(query: str, search_method: str, api_keys: Dict[str, str], num_results: int = 5) -> List[Dict]:
    if search_method == "duckduckgo":
        return duckduckgo_search(query, num_results)
    elif search_method == "google":
        if "google_api_key" not in api_keys or "google_cse_id" not in api_keys:
            console.print("[bold red]Error: Google API Key or CSE ID not provided.[/bold red]")
            return []
        return google_search(query, api_keys["google_api_key"], api_keys["google_cse_id"], num_results)
    elif search_method == "serpapi":
        if "serpapi_api_key" not in api_keys:
            console.print("[bold red]Error: SerpAPI Key not provided.[/bold red]")
            return []
        return serpapi_search(query, api_keys["serpapi_api_key"], num_results)
    else:
        console.print(f"[bold red]Unsupported search method: {search_method}[/bold red]")
        return []

def manage_task(objective, model, file_content=None, previous_results=None, use_search=False, temperature=0.2, max_tokens=8000, search_results=None, groq_api_key=None, api_keys=None):
    console.print(f"\n[bold]Calling Manager for your objective[/bold]")
    previous_results_text = "\n".join(previous_results) if previous_results else "None"
    if file_content:
        console.print(Panel(f"File content:\n{file_content}", title="[bold blue]File Content[/bold blue]", title_align="left", border_style="blue"))

    messages = [
        {
            "role": "user",
            "content": f"Based on the following objective{' and file content' if file_content else ''}, and the previous sub-task results (if any), please break down the objective into the next sub-task, and create a concise and detailed prompt for a subagent so it can execute that task. IMPORTANT!!! when dealing with code tasks make sure you check the code for errors and provide fixes and support as part of the next sub-task. If you find any bugs or have suggestions for better code, please include them in the next sub-task prompt. Please assess if the objective has been fully achieved. If the previous sub-task results comprehensively address all aspects of the objective, include the phrase 'The task is complete:' at the beginning of your response. If the objective is not yet fully achieved, break it down into the next sub-task and create a concise and detailed prompt for a subagent to execute that task.:\n\nObjective: {objective}" + (f'\nFile content:\n{file_content}' if file_content else '') + f"\n\nPrevious sub-task results:\n{previous_results_text}"
        }
    ]
    
    if use_search and search_results:
        messages[0]["content"] += f"\n\nSearch Results:\n{json.dumps(search_results, indent=2)}"

    # Define the search tool
    search_tool = {
        "type": "function",
        "function": {
            "name": "perform_search",
            "description": "Performs a web search using the specified search method and API keys.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    },
                    "search_method": {
                        "type": "string",
                        "description": "The search method to use (duckduckgo, google, serpapi).",
                        "enum": ["duckduckgo", "google", "serpapi"]
                    },
                    "api_keys": {
                        "type": "object",
                        "description": "API keys for the selected search methods.",
                        "properties": {
                            "google_api_key": {
                                "type": "string",
                                "description": "Google API key."
                            },
                            "google_cse_id": {
                                "type": "string",
                                "description": "Google Custom Search Engine ID."
                            },
                            "serpapi_api_key": {
                                "type": "string",
                                "description": "SerpApi API key."
                            }
                        }
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "The number of search results to return."
                    }
                },
                "required": ["query", "search_method", "api_keys", "num_results"]
            }
        }
    }

    # Add the search tool to the tools list if use_search is True
    tools = [search_tool] if use_search else []

    if is_groq_model(model):
        response_text = call_groq_api(model, messages, temperature, max_tokens, groq_api_key)
    else:
        try:
            response = client.chat(
                model=model,
                messages=messages,
                tools=tools,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            response_text = response['message']['content']
            tool_calls = response['message'].get('tool_calls')

            if tool_calls:
                # Process tool calls
                for tool_call in tool_calls:
                    function_name = tool_call['function']['name']
                    arguments = tool_call['function']['arguments']

                    if function_name == "perform_search":
                        search_query = arguments.get('query', '')
                        search_method = arguments.get('search_method', 'duckduckgo')
                        num_results = arguments.get('num_results', 5)

                        search_results = perform_search(search_query, search_method, api_keys, num_results)
                        if search_results:
                            console.print(Panel(f"Search Results: {json.dumps(search_results, indent=2)}", title="[bold green]Search Results[/bold green]", title_align="left", border_style="green"))

                            # Append search results to the messages
                            messages.append({
                                "role": "tool",
                                "content": json.dumps(search_results)
                            })

                            # Call the model again with the search results
                            response = client.chat(
                                model=model,
                                messages=messages,
                                tools=tools,
                                options={
                                    "temperature": temperature,
                                    "num_predict": max_tokens
                                }
                            )
                            response_text = response['message']['content']
                else:
                    console.print("[bold yellow]Warning: Search returned no results.[/bold yellow]")
        except Exception as e:
            if "does not support tools" in str(e):
                # If the model doesn't support tools, fall back to a regular chat without tools
                response = client.chat(
                    model=model,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                )
                response_text = response['message']['content']
            else:
                raise e

    console.print(Panel(response_text, title=f"[bold green]Manager Response[/bold green]", title_align="left", border_style="green", subtitle="Sending task to sub-agent 👇"))
    return response_text, file_content

def sub_agent_task(prompt, model, previous_tasks=None, use_search=False, continuation=False, temperature=0.2, max_tokens=8000, search_results=None, groq_api_key=None):
    if previous_tasks is None:
        previous_tasks = []

    continuation_prompt = "Continuing from the previous answer, please complete the response."
    previous_tasks_summary = "Previous Sub-agent tasks:\n" + "\n".join(f"Task: {task['task']}\nResult: {task['result']}" for task in previous_tasks)
    if continuation:
        prompt = continuation_prompt

    full_prompt = f"{previous_tasks_summary}\n\n{prompt}"
    if use_search and search_results:
        full_prompt += f"\n\nSearch Results:\n{json.dumps(search_results, indent=2)}"
    
    if not full_prompt.strip():
        raise ValueError("Prompt cannot be empty")

    messages = [{"role": "user", "content": full_prompt}]

    # Define the search tool
    search_tool = {
        "type": "function",
        "function": {
            "name": "perform_search",
            "description": "Performs a web search using the specified search method and API keys.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    },
                    "search_method": {
                        "type": "string",
                        "description": "The search method to use (duckduckgo, google, serpapi).",
                        "enum": ["duckduckgo", "google", "serpapi"]
                    },
                    "api_keys": {
                        "type": "object",
                        "description": "API keys for the selected search methods.",
                        "properties": {
                            "google_api_key": {
                                "type": "string",
                                "description": "Google API key."
                            },
"google_cse_id": {
                                "type": "string",
                                "description": "Google Custom Search Engine ID."
                            },
                            "serpapi_api_key": {
                                "type": "string",
                                "description": "SerpApi API key."
                            }
                        }
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "The number of search results to return."
                    }
                },
                "required": ["query", "search_method", "api_keys", "num_results"]
            }
        }
    }

    # Add the search tool to the tools list if use_search is True
    tools = [search_tool] if use_search else []

    if is_groq_model(model):
        response_text = call_groq_api(model, messages, temperature, max_tokens, groq_api_key)
    else:
        try:
            response = client.chat(
                model=model,
                messages=messages,
                tools=tools,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            response_text = response['message']['content']
            tool_calls = response['message'].get('tool_calls')

            if tool_calls:
                # Process tool calls
                for tool_call in tool_calls:
                    function_name = tool_call['function']['name']
                    arguments = tool_call['function']['arguments']

                    if function_name == "perform_search":
                        search_query = arguments.get('query', '')
                        search_method = arguments.get('search_method', 'duckduckgo')
                        api_keys = arguments.get('api_keys', {})
                        num_results = arguments.get('num_results', 5)

                        search_results = perform_search(search_query, search_method, api_keys, num_results)
                        if search_results:
                            console.print(Panel(f"Search Results: {json.dumps(search_results, indent=2)}", title="[bold green]Search Results[/bold green]", title_align="left", border_style="green"))

                            # Append search results to the messages
                            messages.append({
                                "role": "tool",
                                "content": json.dumps(search_results)
                            })

                            # Call the model again with the search results
                            response = client.chat(
                                model=model,
                                messages=messages,
                                tools=tools,
                                options={
                                    "temperature": temperature,
                                    "num_predict": max_tokens
                                }
                            )
                            response_text = response['message']['content']
                        else:
                            console.print("[bold yellow]Warning: Search returned no results.[/bold yellow]")
        except Exception as e:
            if "does not support tools" in str(e):
                # If the model doesn't support tools, fall back to a regular chat without tools
                response = client.chat(
                    model=model,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                )
                response_text = response['message']['content']
            else:
                raise e

    if len(response_text) >= 8000:
        console.print("[bold yellow]Warning:[/bold yellow] Output may be truncated. Attempting to continue the response.")
        continuation_response_text = sub_agent_task(continuation_prompt, model, previous_tasks, use_search, continuation=True, temperature=temperature, max_tokens=max_tokens, search_results=search_results, groq_api_key=groq_api_key)
        response_text += continuation_response_text

    console.print(Panel(response_text, title="[bold blue]Sub-agent Result[/bold blue]", title_align="left", border_style="blue", subtitle="Task completed, sending result to Manager 👇"))
    return response_text

def refine_task(objective, model, sub_task_results, filename, projectname, continuation=False, temperature=0.2, max_tokens=8000, groq_api_key=None, api_keys=None):
    print("\nCalling Refiner to provide the refined final output for your objective:")
    messages = [
        {
            "role": "user",
            "content": "Objective: " + objective + "\n\nSub-task results:\n" + "\n".join(sub_task_results) + "\n\nPlease review and refine the sub-task results into a cohesive final output. Add any missing information or details as needed. When working on code projects, ONLY AND ONLY IF THE PROJECT IS CLEARLY A CODING ONE please provide the following:\n1. Project Name: Create a concise and appropriate project name that fits the project based on what it's creating. The project name should be no more than 20 characters long.\n2. Folder Structure: Provide the folder structure as a valid JSON object, where each key represents a folder or file, and nested keys represent subfolders. Use null values for files. Ensure the JSON is properly formatted without any syntax errors. Please make sure all keys are enclosed in double quotes, and ensure objects are correctly encapsulated with braces, separating items with commas as necessary.\nWrap the JSON object in <folder_structure> tags.\n3. Code Files: For each code file, include ONLY the file name NEVER EVER USE THE FILE PATH OR ANY OTHER FORMATTING YOU ONLY USE THE FOLLOWING format 'Filename: <filename>' followed by the code block enclosed in triple backticks, with the language identifier after the opening backticks, like this:\n\nFilename: <filename>\n```python\n<code>\n```"
        }
    ]

    response_text = call_model(model, messages, temperature, max_tokens, groq_api_key)

    if len(response_text) >= 8000 and not continuation:
        console.print("[bold yellow]Warning:[/bold yellow] Output may be truncated. Attempting to continue the response.")
        continuation_response_text = refine_task(objective, model, sub_task_results + [response_text], filename, projectname, continuation=True, temperature=temperature, max_tokens=max_tokens, groq_api_key=groq_api_key, api_keys=api_keys)
        response_text += "\n" + continuation_response_text

    console.print(Panel(response_text, title="[bold green]Final Output[/bold green]", title_align="left", border_style="green"))
    return response_text

def parse_folder_structure(structure_string):
    structure_string = re.sub(r'\s+', ' ', structure_string)
    match = re.search(r'<folder_structure>(.*?)</folder_structure>', structure_string)
    if not match:
        return None
    
    json_string = match.group(1)
    
    try:
        structure = json.loads(json_string)
        return structure
    except json.JSONDecodeError as e:
        console.print(Panel(f"Error parsing JSON: {e}", title="[bold red]JSON Parsing Error[/bold red]", title_align="left", border_style="red"))
        console.print(Panel(f"Invalid JSON string: [bold]{json_string}[/bold]", title="[bold red]Invalid JSON String[/bold red]", title_align="left", border_style="red"))
        return None

def extract_code_blocks(refined_output):
    code_blocks = {}
    pattern = r'Filename: ([\w.-]+)\n```[\w]*\n(.*?)\n```'
    matches = re.finditer(pattern, refined_output, re.DOTALL)
    for match in matches:
        filename = match.group(1)
        code = match.group(2).strip()
        code_blocks[filename] = code
    return code_blocks

def save_file(content: str, filename: str, project_dir: str) -> None:
    file_path = Path(project_dir) / "code" / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding='utf-8')
    console.print(Panel(f"[yellow]Saved file: [bold]{file_path}[/bold]", title="[bold green]File Saved[/bold green]", title_align="left", border_style="yellow"))

def dump_repository(repo_path: str) -> Dict[str, str]:
    repo_contents = {}
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(('.py', '.json', '.md', '.txt', '.yml', '.yaml')):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        content = f.read()
                        relative_path = os.path.relpath(file_path, repo_path)
                        repo_contents[relative_path] = content
                    except UnicodeDecodeError:
                        console.print(f"Skipping binary file: {file_path}", style="yellow")
    return repo_contents

def execute_code(code: str, project_type: str) -> str:
    if project_type == "Streamlit App":
        return (
            "This is a Streamlit app. To run it, save the following code to a .py file "
            "and use the command: streamlit run your_file.py\n\n"
            f"```python\n{code}\n```"
        )
    else:
        try:
            if not code.strip():
                return "Error: No code to execute."

            pip_commands = [line.strip('# ') for line in code.split('\n') if line.strip().startswith('# !pip install')]
            for cmd in pip_commands:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + cmd.split()[2:])

            code_lines = [line for line in code.split('\n') if not line.strip().startswith('#')]
            code_to_run = '\n'.join(code_lines)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code_to_run)
                temp_file_path = temp_file.name

            result = subprocess.run([sys.executable, temp_file_path], capture_output=True, text=True, timeout=30)
            os.unlink(temp_file_path)

            if result.returncode == 0:
                return f"Execution output:\n\n{result.stdout}"
            else:
                return f"Error during execution:\n\n{result.stderr}"
        except subprocess.TimeoutExpired:
            return "Execution timed out after 30 seconds."
        except Exception as e:
            return f"Error during execution: {str(e)}"

def analyze_code(code: str) -> Dict[str, List[str]]:
    tree = ast.parse(code)
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    return {"functions": functions, "classes": classes}

def generate_test_cases(code_analysis: Dict[str, List[str]]) -> str:
    test_cases = []
    for func in code_analysis["functions"]:
        test_cases.append(f"""
def test_{func}():
    # TODO: Implement test for {func}
    assert True
""")
    for cls in code_analysis["classes"]:
        test_cases.append(f"""
class Test{cls}:
    def test_init(self):
        # TODO: Implement test for {cls} initialization
        assert True

    def test_methods(self):
        # TODO: Implement tests for {cls} methods
        assert True
""")
    return "\n".join(test_cases)

def run_tests(project_dir: str) -> Dict[str, Any]:
    test_dir = os.path.join(project_dir, "tests")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Generate test files for each Python file in the project
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".py") and not file.startswith("test_"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    code = f.read()
                code_analysis = analyze_code(code)
                test_cases = generate_test_cases(code_analysis)
                test_file_path = os.path.join(test_dir, f"test_{file}")
                with open(test_file_path, "w") as f:
                    f.write(f"import sys\nimport os\nsys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n")
                    f.write(f"from {file[:-3]} import *\n\n")
                    f.write(test_cases)

    # Capture stdout to get pytest output
    captured_output = StringIO()
    sys.stdout = captured_output

    # Run pytest
    pytest.main(["-v", test_dir])

    # Restore stdout
    sys.stdout = sys.__stdout__

    # Read the captured output
    test_output = captured_output.getvalue()

    # Parse the test results
    test_results = {
        "passed": test_output.count("PASSED"),
        "failed": test_output.count("FAILED"),
        "errors": test_output.count("ERROR"),
        "warnings": test_output.count("WARN"),
        "output": test_output,
    }

    return test_results

def generate_readme(project_name, user_request, project_type, refined_output, refiner_model):
    readme_prompt = f"""
    Create a comprehensive README.md file for the following project:

    Project Name: {project_name}
    Project Type: {project_type}
    User Request: {user_request}

    Include the following sections:
    1. Project Title and Description
    2. Installation Instructions
    3. Usage Guide
    4. Features
    5. Dependencies
    6. Contributing Guidelines
    7. License Information (Assume MIT license)
    8. Contact/Support Information (Use generic placeholder information)

    Use appropriate Markdown formatting to make the README visually appealing and easy to read.
    Base the content on the following refined output from the project generation:

    {refined_output}
    """

    messages = [{"role": "user", "content": readme_prompt}]
    readme_content = call_model(refiner_model, messages, temperature=0.7, max_tokens=2000)
    return readme_content

def build_interface():
    os.environ['USER_AGENT'] = 'BuildAgent/1.0'

    st.title("🔨 Build: Autonomous Multi-Agent Software Development System")

    settings = load_settings()
    api_keys = load_api_keys()  # Load API keys

    if 'project_state' not in st.session_state:
        st.session_state.project_state = {
            'status': 'Not Started',
            'current_step': '',
            'code': '',
            'documentation': '',
            'test_results': '',
            'quality_review': '',
            'project_dir': '',
            'errors': [],
            'warnings': [],
            'agent_logs': [],
            'progress': 0,
            'iterations': 0,
            'max_iterations': 3,
        }

    progress_bar = st.progress(0)
    status_text = st.empty()

    groq_api_key = st.text_input("Groq API Key", value=settings.get("groq_api_key", ""), type="password")
    if groq_api_key != settings.get("groq_api_key"):
        settings["groq_api_key"] = groq_api_key
        save_settings(settings)

    has_advanced_access = st.checkbox("I have access to advanced Groq models (Early API access)", value=settings.get("has_advanced_access", False))
    if has_advanced_access != settings.get("has_advanced_access"):
        settings["has_advanced_access"] = has_advanced_access
        save_settings(settings)

    all_models = get_all_models(settings, include_advanced=has_advanced_access)

    def get_valid_model(model_key, default_index=0):
        saved_model = settings.get(model_key)
        try:
            index = all_models.index(saved_model)
        except ValueError:
            index = default_index
        return st.selectbox(f"{model_key.replace('_', ' ').title()}", all_models, index=index)

    st.subheader("Model Selection")
    manager_model = get_valid_model("manager_model")
    subagent_model = get_valid_model("subagent_model")
    refiner_model = get_valid_model("refiner_model")

    settings.update({
        "manager_model": manager_model,
        "subagent_model": subagent_model,
        "refiner_model": refiner_model
    })
    save_settings(settings)

    st.info("Note: Some Groq models (Llama 3.1 405B, 70B, and 8B) are currently only available to paying customers with early API access. All 3.1 models are limited to max_tokens of 8k, and 405b is limited to 16k input tokens during the preview launch.")

    input_method = st.radio("Choose input method:", ["Enter Project Request", "Provide Repository Directory"])

    if input_method == "Enter Project Request":
        user_request = st.text_area("Enter your project request:", height=100, key="user_request")
        project_type = st.selectbox("Select Project Type", ["Command-line Tool", "Streamlit App", "API", "Data Analysis Script"], key="project_type")
        repo_contents = None
    else:
        repo_dir = st.text_input("Enter the path to your repository directory:")
        if repo_dir and os.path.isdir(repo_dir):
            repo_contents = dump_repository(repo_dir)
            st.success(f"Repository loaded successfully. Found {len(repo_contents)} files.")
        else:
            repo_contents = None
            if repo_dir:
                st.error("Invalid directory path. Please enter a valid directory.")
        user_request = None
        project_type = None

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    with col2:
        max_tokens = st.slider("Max Tokens", min_value=1000, max_value=128000, value=8000, step=1000)

    use_search = st.checkbox("Use search functionality")
    if use_search:
        search_method = st.selectbox("Select search method", ["duckduckgo", "google", "serpapi"])
        
        # Display current API keys and allow editing
        if search_method == "google":
            api_keys["google_api_key"] = st.text_input("Google API Key", value=api_keys.get("google_api_key", ""), type="password")
            api_keys["google_cse_id"] = st.text_input("Google Custom Search Engine ID", value=api_keys.get("google_cse_id", ""), type="password")
        elif search_method == "serpapi":
            api_keys["serpapi_api_key"] = st.text_input("SerpApi API Key", value=api_keys.get("serpapi_api_key", ""), type="password")
        
        # Add num_results input
        num_results = st.number_input("Number of search results", min_value=1, max_value=20, value=5)
        
        # Save API keys if they have changed
        save_api_keys(api_keys)
    else:
        search_method = None
        num_results = None

    agent_logs = st.empty()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Code", "Documentation", "Test Results", "Quality Review", "Execution"])

    if st.button("🚀 Build Project"):
        if user_request or repo_contents:
            console.print(f"Starting new project build", style="bold white on blue")
            if user_request:
                console.print(f"User Request: {user_request}", style="cyan")
                console.print(f"Project Type: {project_type}", style="cyan")
            else:
                console.print(f"Repository loaded with {len(repo_contents)} files", style="cyan")

            st.session_state.project_state['status'] = 'In Progress'
            st.session_state.project_state['agent_logs'] = []
            st.session_state.project_state['progress'] = 0
            st.session_state.project_state['iterations'] = 0

            search_results = None
            if use_search:
                search_query = user_request or "Repository improvement techniques"
                search_results = perform_search(search_query, search_method, api_keys, num_results)
                console.print(Panel(f"Search Results: {json.dumps(search_results, indent=2)}", title="[bold green]Search Results[/bold green]", title_align="left", border_style="green"))

            task_exchanges = []
            worker_tasks = []

            while st.session_state.project_state['iterations'] < st.session_state.project_state['max_iterations']:
                previous_results = [result for _, result in task_exchanges]
                if not task_exchanges:
                    agent_result, file_content_for_worker = manage_task(
                        user_request or "Analyze and improve the provided repository",
                        model=manager_model,
                        file_content=repo_contents,
                        previous_results=previous_results,
                        use_search=use_search,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        search_results=search_results,
                        groq_api_key=settings.get("groq_api_key"),
                        api_keys=api_keys
                    )
                else:
                    agent_result, _ = manage_task(
                        user_request or "Analyze and improve the provided repository",
                        model=manager_model,
                        previous_results=previous_results,
                        use_search=use_search,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        search_results=search_results,
                        groq_api_key=settings.get("groq_api_key"),
                        api_keys=api_keys
                    )

                if "The task is complete:" in agent_result:
                    final_output = agent_result.replace("The task is complete:", "").strip()
                    break
                else:
                    sub_task_prompt = agent_result
                    if file_content_for_worker and not worker_tasks:
                        sub_task_prompt = f"{sub_task_prompt}\n\nFile content:\n{file_content_for_worker}"
                    sub_task_result = sub_agent_task(
                        sub_task_prompt,
                        model=subagent_model,
                        previous_tasks=worker_tasks,
                        use_search=use_search,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        search_results=search_results,
                        groq_api_key=settings.get("groq_api_key"),
                        api_keys=api_keys
                    )
                    worker_tasks.append({"task": sub_task_prompt, "result": sub_task_result})
                    task_exchanges.append((sub_task_prompt, sub_task_result))
                    file_content_for_worker = None

                st.session_state.project_state['iterations'] += 1
                st.session_state.project_state['progress'] = st.session_state.project_state['iterations'] / st.session_state.project_state['max_iterations']
                progress_bar.progress(st.session_state.project_state['progress'])
                status_text.text(f"Iteration {st.session_state.project_state['iterations']} of {st.session_state.project_state['max_iterations']}")

            sanitized_objective = re.sub(r'\W+', '_', user_request if user_request else "repository_improvement")
            timestamp = datetime.now().strftime("%H-%M-%S")
            refined_output = refine_task(
                user_request or "Analyze and improve the provided repository",
                model=refiner_model,
                sub_task_results=[result for _, result in task_exchanges],
                filename=timestamp,
                projectname=sanitized_objective,
                temperature=temperature,
                max_tokens=max_tokens,
                groq_api_key=settings.get("groq_api_key"),
                api_keys=api_keys
            )

            project_name_match = re.search(r'Project Name: (.*)', refined_output)
            project_name = project_name_match.group(1).strip() if project_name_match else sanitized_objective

            folder_structure = parse_folder_structure(refined_output)
            code_blocks = extract_code_blocks(refined_output)

            project_dir = Path("generated_projects") / f"{project_name}_{timestamp}"
            project_dir.mkdir(parents=True, exist_ok=True)
            console.print(Panel(f"Created project folder: [bold]{project_dir}[/bold]", title="[bold green]Project Folder[/bold green]", title_align="left", border_style="green"))

            if code_blocks:
                for filename, code in code_blocks.items():
                    save_file(code, filename, str(project_dir))
                    console.print(Panel(f"Created file: [bold]{filename}[/bold]", title="[bold green]Project Files[/bold green]", title_align="left", border_style="yellow"))

            # Generate README.md
            readme_content = generate_readme(project_name, user_request, project_type, refined_output, refiner_model)
            readme_path = project_dir / "README.md"
            readme_path.write_text(readme_content, encoding='utf-8')
            console.print(Panel(f"Created README.md: [bold]{readme_path}[/bold]", title="[bold green]README.md Created[/bold green]", title_align="left", border_style="green"))

            st.session_state.project_state['documentation'] = readme_content

            # Run tests
            test_results = run_tests(project_dir)
            st.session_state.project_state['test_results'] = test_results

            # Check if there are any failed tests or errors
            if test_results['failed'] > 0 or test_results['errors'] > 0:
                console.print(Panel(f"[bold red]Tests failed or encountered errors. Asking user for additional iterations.[/bold red]", title="[bold red]Test Results[/bold red]", title_align="left", border_style="red"))
                st.warning("Tests failed or encountered errors. Click the button below to run additional iterations.")
                if st.button("Run Additional Iterations to Fix Issues"):
                    st.session_state.project_state['max_iterations'] += 3
                    st.session_state.project_state['iterations'] = 0  # Reset iterations
                    st.rerun()  # This will restart the Streamlit app with the updated max_iterations

            # Continue with the rest of the process if no additional iterations are needed
            exchange_log = f"Objective: {user_request}\n\n" if user_request else f"Objective: Analyze and improve the provided repository\n\n"
            exchange_log += "=" * 40 + " Task Breakdown " + "=" * 40 + "\n\n"
            for i, (prompt, result) in enumerate(task_exchanges, start=1):
                exchange_log += f"Task {i}:\n"
                exchange_log += f"Prompt: {prompt}\n"
                exchange_log += f"Result: {result}\n\n"
            exchange_log += "=" * 40 + " Refined Final Output " + "=" * 40 + "\n\n"
            exchange_log += refined_output

            exchange_log_filename = f"{timestamp}_{project_name}_log.md"
            exchange_log_path = os.path.join(project_dir, exchange_log_filename)
            try:
                with open(exchange_log_path, 'w', encoding='utf-8') as f:
                    f.write(exchange_log)
                console.print(Panel(f"Saved exchange log: [bold]{exchange_log_path}[/bold]", title="[bold green]Exchange Log[/bold green]", title_align="left", border_style="green"))
            except IOError as e:
                console.print(Panel(f"Error saving exchange log: {str(e)}", title="[bold red]Exchange Log Error[/bold red]", title_align="left", border_style="red"))

            st.session_state.project_state['status'] = 'Completed'
            st.session_state.project_state['code'] = refined_output
            st.session_state.project_state['project_dir'] = project_dir

            st.success(f"Project built successfully! Files saved in: {project_dir}")
        else:
            console.print("Please enter a project request or provide a valid repository directory.", style="red")
            st.error("Please enter a project request or provide a valid repository directory.")

    if st.session_state.project_state['status'] != 'Not Started':
        agent_logs.text("\n".join(st.session_state.project_state['agent_logs']))

        with tab1:
            st.code(st.session_state.project_state.get('code', ''))

        with tab2:
            st.markdown(st.session_state.project_state.get('documentation', ''))

        with tab3:
            test_results = st.session_state.project_state.get('test_results', {})
            st.subheader("Test Results")
            st.write(f"Passed: {test_results.get('passed', 0)}")
            st.write(f"Failed: {test_results.get('failed', 0)}")
            st.write(f"Errors: {test_results.get('errors', 0)}")
            st.write(f"Warnings: {test_results.get('warnings', 0)}")
            st.subheader("Test Output")
            st.code(test_results.get('output', ''), language='plaintext')

        with tab4:
            st.markdown(st.session_state.project_state.get('quality_review', ''))

        with tab5:
            if st.session_state.project_state['status'] == 'Completed':
                execution_result = execute_code(st.session_state.project_state['code'], project_type)
                st.code(execution_result, language='python')

if __name__ == "__main__":
    build_interface()