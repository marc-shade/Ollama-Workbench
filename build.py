# build.py
import streamlit as st
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List

import ollama
import pytest
import requests
from duckduckgo_search import DDGS
from googleapiclient.discovery import build
from ollama import Client
from rich.console import Console
from rich.panel import Panel
from serpapi import GoogleSearch
from streamlit import session_state as st_ss

from ollama_utils import *
from groq_utils import *
from openai_utils import *
from external_providers import get_available_groq_models

API_KEYS_FILE = "api_keys.json"
SETTINGS_FILE = "build_settings.json"

# Initialize the Rich Console
console = Console()

# Initialize the Ollama client
client = Client(host="http://localhost:11434")

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)

def is_groq_model(model_name):
    api_keys = load_api_keys()
    available_groq_models = get_available_groq_models(api_keys)
    return model_name in available_groq_models


def ensure_ruff_installed():
    try:
        subprocess.run(
            ["ruff", "--version"], check=True, capture_output=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Ruff is not installed. Installing...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "ruff"], check=True
        )
    print("Ruff is ready.")


# Call this at the beginning of build_interface()
ensure_ruff_installed()


def duckduckgo_search(query: str, num_results: int = 5) -> List[Dict]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))
    return [{"title": result["title"], "url": result["href"]} for result in results]


def google_search(
    query: str, api_key: str, cse_id: str, num_results: int = 5
) -> List[Dict]:
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
    return [
        {"title": item["title"], "url": item["link"]}
        for item in res.get("items", [])
    ]


def serpapi_search(query: str, api_key: str, num_results: int = 5) -> List[Dict]:
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": num_results,
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        if not organic_results:
            print(f"SerpApi returned no results. Full response: {results}")
        return [
            {"title": result["title"], "url": result["link"]}
            for result in organic_results
        ]
    except Exception as e:
        print(f"Error in SerpApi search: {str(e)}")
        return []


def perform_search(
    query: str, search_method: str, api_keys: Dict[str, str], num_results: int = 5
) -> List[Dict]:
    if search_method == "duckduckgo":
        return duckduckgo_search(query, num_results)
    elif search_method == "google":
        if "google_api_key" not in api_keys or "google_cse_id" not in api_keys:
            console.print(
                "[bold red]Error: Google API Key or CSE ID not provided.[/bold red]"
            )
            return []
        return google_search(
            query, api_keys["google_api_key"], api_keys["google_cse_id"], num_results
        )
    elif search_method == "serpapi":
        if "serpapi_api_key" not in api_keys:
            console.print("[bold red]Error: SerpAPI Key not provided.[/bold red]")
            return []
        return serpapi_search(query, api_keys["serpapi_api_key"], num_results)
    else:
        console.print(
            f"[bold red]Unsupported search method: {search_method}[/bold red]"
        )
        return []


def create_agent_context(project_state, test_results, current_task):
    return {
        "project_state": {
            "status": project_state["status"],
            "current_step": project_state["current_step"],
            "iterations": project_state["iterations"],
            "max_iterations": project_state["max_iterations"],
        },
        "test_results": {
            "pytest_summary": {
                "passed": test_results.get("passed", 0),
                "failed": test_results.get("failed", 0),
                "errors": test_results.get("errors", 0),
                "warnings": test_results.get("warnings", 0),
            },
            "ruff_summary": {
                "total_violations": test_results.get("ruff_violations", 0),
            },
            "detailed_reports": {
                "pytest_output": test_results.get("pytest_output", ""),
                "ruff_report": test_results.get("ruff_report", ""),
            },
        },
        "current_task": current_task,
        "agile_process": {
            "sprint": project_state["iterations"],
            "total_sprints": project_state["max_iterations"],
            "phase": "Development"
            if project_state["iterations"] < project_state["max_iterations"]
            else "Final Review",
        },
        "previous_tasks": project_state.get("previous_tasks", []),
    }


def manager_agent_task(
    context: Dict[str, Any], model: str, temperature: float, max_tokens: int, groq_api_key=None, openai_api_key=None
) -> Dict[str, Any]:
    prompt = f"""
    You are the manager agent in an Agile software development process. 
    Current project state: {json.dumps(context['project_state'], indent=2)}
    Latest test results: {json.dumps(context['test_results'], indent=2)}
    Current task: {context['current_task']}
    Agile process: {json.dumps(context['agile_process'], indent=2)}

    Based on the current state and test results, please:
    1. Analyze the test results and code quality issues.
    2. Update the work plan for the next sprint.
    3. Prioritize tasks to address failed tests and code quality issues.
    4. Provide clear instructions for the coding agents.

    Respond with a JSON object containing:
    1. "analysis": Your analysis of the current state and test results.
    2. "work_plan": The updated work plan for the next sprint.
    3. "priorities": A list of prioritized tasks.
    4. "instructions": Clear instructions for the coding agents.
    """

    try:
        if model.startswith("gpt-"):
            response = call_openai_api(model, [{"role": "user", "content": prompt}], temperature, max_tokens, openai_api_key)
        elif is_groq_model(model):
            response = call_groq_api(model, prompt, temperature, max_tokens, groq_api_key)
        else:
            response = call_model(model, [{"role": "user", "content": prompt}], temperature, max_tokens, groq_api_key)
        return json.loads(response)
    except json.JSONDecodeError as e:
        st.session_state.project_state["errors"].append(
            f"Error parsing JSON response from Manager Agent: {e}"
        )
        st.session_state.project_state["errors"].append(f"Raw response: {response}")
        return {
            "analysis": "Error parsing response. Please review the raw output.",
            "work_plan": "Unable to generate work plan due to parsing error.",
            "priorities": ["Review and fix JSON parsing issues"],
            "instructions": "Please review the raw output and manually extract relevant information.",
        }
    except Exception as e:
        st.session_state.project_state["errors"].append(
            f"An error occurred during the manager agent task: {str(e)}"
        )
        return {}


def sub_agent_task(
    context, manager_response, model, temperature, max_tokens, groq_api_key=None, openai_api_key=None
):
    previous_tasks = context.get("previous_tasks", [])

    # Convert previous tasks to a string representation
    previous_tasks_str = "\n".join(
        [
            f"Task: {task.get('task', '')}\nResult: {task.get('result', '')}"
            for task in previous_tasks
            if isinstance(task, dict)
        ]
    )

    prompt = f"""
    You are a coding agent in an Agile software development process.
    Current project state: {json.dumps(context['project_state'], indent=2)}
    Latest test results: {json.dumps(context['test_results'], indent=2)}
    Current task: {context['current_task']}
    Agile process: {json.dumps(context['agile_process'], indent=2)}

    Manager's analysis: {manager_response['analysis']}
    Work plan: {manager_response['work_plan']}
    Priorities: {json.dumps(manager_response['priorities'], indent=2)}
    Instructions: {manager_response['instructions']}

    Previous tasks:
    {previous_tasks_str}

    Based on the provided information and instructions, please:
    1. Implement the necessary code changes to address the current task and priorities.
    2. Focus on fixing failed tests and improving code quality.
    3. Provide a brief explanation of your changes.

    Respond with a JSON object containing:
    1. "code_changes": The implemented code changes.
    2. "explanation": A brief explanation of your changes.
    """

    try:
        if model.startswith("gpt-"):
            response = call_openai_api(model, [{"role": "user", "content": prompt}], temperature, max_tokens, openai_api_key)
        elif is_groq_model(model):
            response = call_groq_api(model, prompt, temperature, max_tokens, groq_api_key)
        else:
            response = client.chat(
                model=str(model),  # Ensure model is always a string
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            response = response['message']['content']
        return json.loads(response)
    except json.JSONDecodeError as e:
        st.session_state.project_state["errors"].append(
            f"Error parsing JSON response from Sub Agent: {e}"
        )
        st.session_state.project_state["errors"].append(f"Raw response: {response}")
        return {}
    except Exception as e:
        st.session_state.project_state["errors"].append(
            f"An error occurred during the sub-agent task: {str(e)}"
        )
        return {}


def manage_task(
    objective,
    model,
    file_content=None,
    previous_results=None,
    use_search=False,
    temperature=0.2,
    max_tokens=8000,
    search_results=None,
    groq_api_key=None,
    openai_api_key=None,
    api_keys=None,
):
    console.print(f"\n[bold]Calling Manager for your objective[/bold]")
    previous_results_text = (
        "\n".join(previous_results) if previous_results else "None"
    )
    if file_content:
        console.print(
            Panel(
                f"File content:\n{file_content}",
                title="[bold blue]File Content[/bold blue]",
                title_align="left",
                border_style="blue",
            )
        )

    messages = [
        {
            "role": "user",
            "content": f"Based on the following objective{' and file content' if file_content else ''}, and the previous sub-task results (if any), please break down the objective into the next sub-task, and create a concise and detailed prompt for a subagent so it can execute that task. IMPORTANT!!! when dealing with code tasks make sure you check the code for errors and provide fixes and support as part of the next sub-task. If you find any bugs or have suggestions for better code, please include them in the next sub-task prompt. Please assess if the objective has been fully achieved. If the previous sub-task results comprehensively address all aspects of the objective, include the phrase 'The task is complete:' at the beginning of your response. If the objective is not yet fully achieved, break it down into the next sub-task and create a concise and detailed prompt for a subagent to execute that task.:\n\nObjective: {objective}"
            + (f"\nFile content:\n{file_content}" if file_content else "")
            + f"\n\nPrevious sub-task results:\n{previous_results_text}",
        }
    ]

    if use_search and search_results:
        messages[0][
            "content"
        ] += f"\n\nSearch Results:\n{json.dumps(search_results, indent=2)}"

    # Define the search tool
    search_tool = {
        "type": "function",
        "function": {
            "name": "perform_search",
            "description": "Performs a web search using the specified search method and API keys.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "search_method": {
                        "type": "string",
                        "description": "The search method to use (duckduckgo, google, serpapi).",
                        "enum": ["duckduckgo", "google", "serpapi"],
                    },
                    "api_keys": {
                        "type": "object",
                        "description": "API keys for the selected search methods.",
                        "properties": {
                            "google_api_key": {
                                "type": "string",
                                "description": "Google API key.",
                            },
                            "google_cse_id": {
                                "type": "string",
                                "description": "Google Custom Search Engine ID.",
                            },
                            "serpapi_api_key": {
                                "type": "string",
                                "description": "SerpApi API key.",
                            },
                        },
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "The number of search results to return.",
                    },
                },
                "required": ["query", "search_method", "api_keys", "num_results"],
            },
        },
    }

    # Add the search tool to the tools list if use_search is True
    tools = [search_tool] if use_search else []

    if model.startswith("gpt-"):
        response_text = call_openai_api(model, messages, temperature, max_tokens, openai_api_key)
    elif is_groq_model(model):
        response_text = call_groq_api(
            model, messages, temperature, max_tokens, groq_api_key
        )
    else:
        try:
            response = client.chat(
                model=model,
                messages=messages,
                tools=tools,
                options={"temperature": temperature, "num_predict": max_tokens},
            )
            response_text = response["message"]["content"]
            tool_calls = response["message"].get("tool_calls")

            if tool_calls:
                # Process tool calls
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    arguments = tool_call["function"]["arguments"]

                    if function_name == "perform_search":
                        search_query = arguments.get("query", "")
                        search_method = arguments.get(
                            "search_method", "duckduckgo"
                        )
                        num_results = arguments.get("num_results", 5)

                        search_results = perform_search(
                            search_query, search_method, st_ss.api_keys, num_results
                        )
                        if search_results:
                            console.print(
                                Panel(
                                    f"Search Results: {json.dumps(search_results, indent=2)}",
                                    title="[bold green]Search Results[/bold green]",
                                    title_align="left",
                                    border_style="green",
                                )
                            )

                            # Append search results to the messages
                            messages.append(
                                {"role": "tool", "content": json.dumps(search_results)}
                            )

                            # Call the model again with the search results
                            response = client.chat(
                                model=model,
                                messages=messages,
                                tools=tools,
                                options={
                                    "temperature": temperature,
                                    "num_predict": max_tokens,
                                },
                            )
                            response_text = response["message"]["content"]
                        else:
                            console.print(
                                "[bold yellow]Warning: Search returned no results.[/bold yellow]"
                            )
        except Exception as e:
            if "does not support tools" in str(e):
                # If the model doesn't support tools, fall back to a regular chat without tools
                response = client.chat(
                    model=model,
                    messages=messages,
                    options={"temperature": temperature, "num_predict": max_tokens},
                )
                response_text = response["message"]["content"]
            else:
                st.session_state.project_state["errors"].append(
                    f"An error occurred during the manager task: {str(e)}"
                )
                response_text = ""

    console.print(
        Panel(
            response_text,
            title=f"[bold green]Manager Response[/bold green]",
            title_align="left",
            border_style="green",
            subtitle="Sending task to sub-agent 👇",
        )
    )
    return response_text, file_content


def sub_agent_task(
    prompt,
    model,
    previous_tasks=None,
    use_search=False,
    continuation=False,
    temperature=0.2,
    max_tokens=8000,
    search_results=None,
    groq_api_key=None,
    openai_api_key=None,
):
    if previous_tasks is None:
        previous_tasks = []

    # Convert previous tasks to a string representation
    previous_tasks_str = "\n".join(
        [
            f"Task: {task.get('task', '')}\nResult: {task.get('result', '')}"
            for task in previous_tasks
            if isinstance(task, dict)
        ]
    )

    continuation_prompt = (
        "Continuing from the previous answer, please complete the response."
    )
    previous_tasks_summary = (
        f"Previous Sub-agent tasks:\n{previous_tasks_str}"
        if previous_tasks
        else "No previous sub-agent tasks."
    )
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
                    "query": {"type": "string", "description": "The search query."},
                    "search_method": {
                        "type": "string",
                        "description": "The search method to use (duckduckgo, google, serpapi).",
                        "enum": ["duckduckgo", "google", "serpapi"],
                    },
                    "api_keys": {
                        "type": "object",
                        "description": "API keys for the selected search methods.",
                        "properties": {
                            "google_api_key": {
                                "type": "string",
                                "description": "Google API key.",
                            },
                            "google_cse_id": {
                                "type": "string",
                                "description": "Google Custom Search Engine ID.",
                            },
                            "serpapi_api_key": {
                                "type": "string",
                                "description": "SerpApi API key.",
                            },
                        },
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "The number of search results to return.",
                    },
                },
                "required": ["query", "search_method", "api_keys", "num_results"],
            },
        },
    }

    # Add the search tool to the tools list if use_search is True
    tools = [search_tool] if use_search else []

    if model.startswith("gpt-"):
        response_text = call_openai_api(model, messages, temperature, max_tokens, openai_api_key)
    elif is_groq_model(model):
        response_text = call_groq_api(
            model, messages, temperature, max_tokens, groq_api_key
        )
    else:
        try:
            response = client.chat(
                model=str(model),  # Ensure model is always a string
                messages=messages,
                tools=tools,
                options={"temperature": temperature, "num_predict": max_tokens},
            )
            response_text = response["message"]["content"]
            tool_calls = response["message"].get("tool_calls")

            if tool_calls:
                # Process tool calls
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])

                    if function_name == "perform_search":
                        search_query = arguments.get("query", "")
                        search_method = arguments.get(
                            "search_method", "duckduckgo"
                        )
                        api_keys = arguments.get("api_keys", {})
                        num_results = arguments.get("num_results", 5)

                        search_results = perform_search(
                            search_query, search_method, api_keys, num_results
                        )
                        if search_results:
                            console.print(
                                Panel(
                                    f"Search Results: {json.dumps(search_results, indent=2)}",
                                    title="[bold green]Search Results[/bold green]",
                                    title_align="left",
                                    border_style="green",
                                )
                            )

                            # Append search results to the messages
                            messages.append(
                                {"role": "tool", "content": json.dumps(search_results)}
                            )

                            # Call the model again with the search results
                            response = client.chat(
                                model=str(model),  # Ensure model is always a string
                                messages=messages,
                                tools=tools,
                                options={
                                    "temperature": temperature,
                                    "num_predict": max_tokens,
                                },
                            )
                            response_text = response["message"]["content"]
                        else:
                            console.print(
                                "[bold yellow]Warning: Search returned no results.[/bold yellow]"
                            )
        except Exception as e:
            if "does not support tools" in str(e):
                # If the model doesn't support tools, fall back to a regular chat without tools
                response = client.chat(
                    model=str(model),  # Ensure model is always a string
                    messages=messages,
                    options={"temperature": temperature, "num_predict": max_tokens},
                )
                response_text = response["message"]["content"]
            else:
                st.session_state.project_state["errors"].append(
                    f"An error occurred during the sub-agent task: {str(e)}"
                )
                response_text = ""

    if len(response_text) >= 8000:
        console.print(
            "[bold yellow]Warning:[/bold yellow] Output may be truncated. Attempting to continue the response."
        )
        continuation_response_text = sub_agent_task(
            continuation_prompt,
            model,
            previous_tasks,
            use_search,
            continuation=True,
            temperature=temperature,
            max_tokens=max_tokens,
            search_results=search_results,
            groq_api_key=groq_api_key,
            openai_api_key=openai_api_key,
        )
        response_text += continuation_response_text

    console.print(
        Panel(
            response_text,
            title="[bold blue]Sub-agent Result[/bold blue]",
            title_align="left",
            border_style="blue",
            subtitle="Task completed, sending result to Manager 👇",
        )
    )
    return response_text


def refine_task(
    objective,
    model,
    sub_task_results,
    filename,
    projectname,
    continuation=False,
    temperature=0.2,
    max_tokens=8000,
    groq_api_key=None,
    openai_api_key=None,
):
    print("\nCalling Refiner to provide the refined final output for your objective:")
    messages = [
        {
            "role": "user",
            "content": "Objective: "
            + objective
            + "\n\nSub-task results:\n"
            + "\n".join(sub_task_results)
            + "\n\nPlease review and refine the sub-task results into a cohesive final output. Add any missing information or details as needed. When working on code projects, ONLY AND ONLY IF THE PROJECT IS CLEARLY A CODING ONE please provide the following:\n1. Project Name: Create a concise and appropriate project name that fits the project based on what it's creating. The project name should be no more than 20 characters long.\n2. Folder Structure: Provide the folder structure as a valid JSON object, where each key represents a folder or file, and nested keys represent subfolders. Use null values for files. Ensure the JSON is properly formatted without any syntax errors. Please make sure all keys are enclosed in double quotes, and ensure objects are correctly encapsulated with braces, separating items with commas as necessary.\nWrap the JSON object in <folder_structure> tags.\n3. Code Files: For each code file, include ONLY the file name NEVER EVER USE THE FILE PATH OR ANY OTHER FORMATTING YOU ONLY USE THE FOLLOWING format 'Filename: <filename>' followed by the code block enclosed in triple backticks, with the language identifier after the opening backticks, like this:\n\nFilename: <filename>\n```python\n<code>\n```",
        }
    ]

    try:
        if model.startswith("gpt-"):
            response_text = call_openai_api(model, messages, temperature, max_tokens, openai_api_key)
        elif is_groq_model(model):
            response_text = call_groq_api(model, messages, temperature, max_tokens, groq_api_key)
        else:
            response_text = call_model(model, messages, temperature, max_tokens, groq_api_key)
    except Exception as e:
        st.session_state.project_state["errors"].append(
            f"An error occurred during the refine task: {str(e)}"
        )
        response_text = ""

    if len(response_text) >= 8000 and not continuation:
        console.print(
            "[bold yellow]Warning:[/bold yellow] Output may be truncated. Attempting to continue the response."
        )
        continuation_response_text = refine_task(
            objective,
            model,
            sub_task_results + [response_text],
            filename,
            projectname,
            continuation=True,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=groq_api_key,
            openai_api_key=openai_api_key,
        )
        response_text += "\n" + continuation_response_text

    console.print(
        Panel(
            response_text,
            title="[bold green]Final Output[/bold green]",
            title_align="left",
            border_style="green",
        )
    )
    return response_text


def parse_folder_structure(structure_string):
    structure_string = re.sub(r"\s+", " ", structure_string)
    match = re.search(
        r"<folder_structure>(.*?)</folder_structure>", structure_string
    )
    if not match:
        return None

    json_string = match.group(1)

    try: 
        structure = json.loads(json_string)
        return structure
    except json.JSONDecodeError as e:
        console.print(
            Panel(
                f"Error parsing JSON: {e}",
                title="[bold red]JSON Parsing Error[/bold red]",
                title_align="left",
                border_style="red",
            )
        )
        console.print(
            Panel(
                f"Invalid JSON string: [bold]{json_string}[/bold]",
                title="[bold red]Invalid JSON String[/bold red]",
                title_align="left",
                border_style="red",
            )
        )
        return None


def extract_code_blocks(refined_output):
    code_blocks = {}
    pattern = r"Filename: ([\w.-]+)\n```[\w]*\n(.*?)\n```"
    matches = re.finditer(pattern, refined_output, re.DOTALL)
    for match in matches:
        filename = match.group(1)
        code = match.group(2).strip()
        code_blocks[filename] = code
    return code_blocks


def save_file(content: str, filename: str, project_dir: str) -> None:
    try:
        file_path = Path(project_dir) / "code" / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        console.print(
            Panel(
                f"[yellow]Saved file: [bold]{file_path}[/bold]",
                title="[bold green]File Saved[/bold green]",
                title_align="left",
                border_style="yellow",
            )
        )
    except Exception as e:
        st.session_state.project_state["errors"].append(
            f"Error saving file '{filename}': {str(e)}"
        )


def dump_repository(repo_path: str) -> Dict[str, str]:
    repo_contents = {}
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith((".py", ".json", ".md", ".txt", ".yml", ".yaml")):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        content = f.read()
                        relative_path = os.path.relpath(file_path, repo_path)
                        repo_contents[relative_path] = content
                    except UnicodeDecodeError:
                        console.print(
                            f"Skipping binary file: {file_path}", style="yellow"
                        )
    return repo_contents


def analyze_code(code: str) -> Dict[str, List[str]]:
    try:
        tree = ast.parse(code)
        functions = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        classes = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ]
        return {"functions": functions, "classes": classes}
    except SyntaxError as e:
        st.session_state.project_state["errors"].append(
            f"Syntax error in code analysis: {e}"
        )
        return {"functions": [], "classes": []}


def analyze_code_with_ruff(file_path: str) -> List[Dict]:
    if shutil.which("ruff") is None:
        print(
            "Ruff is not installed or not in PATH. Skipping Ruff analysis."
        )
        return []

    try:
        result = subprocess.run(
            ["ruff", "check", file_path, "--format=json"],
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)
    except FileNotFoundError:
        print("Ruff command not found. Skipping Ruff analysis.")
        return []
    except json.JSONDecodeError:
        print("Error parsing Ruff output. Skipping Ruff analysis.")
        return []
    except Exception as e:
        print(f"An error occurred while running Ruff: {str(e)}")
        return []


def generate_ruff_report(violations: List[Dict]) -> str:
    report = "Code Analysis Report:\n\n"
    for violation in violations:
        report += f"File: {violation['filename']}\n"
        report += f"Line: {violation['location']['row']}\n"
        report += f"Column: {violation['location']['column']}\n"
        report += f"Error Code: {violation['code']}\n"
        report += f"Description: {violation['message']}\n"
        report += f"Suggested Fix: {generate_fix_suggestion(violation)}\n\n"
    return report


def generate_fix_suggestion(violation: Dict) -> str:
    # This function would contain logic to suggest fixes based on the error code
    # For simplicity, we'll just return a placeholder message
    return "Review the code and address the issue according to the error description."


def generate_test_cases(code_analysis: Dict[str, List[str]]) -> str:
    test_cases = []
    for func in code_analysis["functions"]:
        test_cases.append(
            f"""
def test_{func}():
    # TODO: Implement test for {func}
    assert True
"""
        )
    for cls in code_analysis["classes"]:
        test_cases.append(
            f"""
class Test{cls}:
    def test_init(self):
        # TODO: Implement test for {cls} initialization
        assert True

    def test_methods(self):
        # TODO: Implement tests for {cls} methods
        assert True
"""
        )
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
                    f.write(
                        f"import sys\nimport os\nsys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
                    )
                    f.write(f"from {file[:-3]} import *\n\n")
                    f.write(test_cases)

    try:
        # Run pytest
        pytest_result = subprocess.run(
            ["pytest", "-v", test_dir], capture_output=True, text=True
        )

        # Run Ruff on all Python files
        ruff_violations = []
        for root, _, files in os.walk(project_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    result = subprocess.run(
                        ["ruff", "check", file_path, "--format=json"],
                        capture_output=True,
                        text=True,
                    )
                    try:
                        violations = json.loads(result.stdout)
                        ruff_violations.extend(violations)
                    except json.JSONDecodeError:
                        print(f"Error parsing Ruff output for {file_path}")

        ruff_report = generate_ruff_report(ruff_violations)

        return {
            "pytest_output": pytest_result.stdout,
            "ruff_report": ruff_report,
            "passed": pytest_result.stdout.count("PASSED"),
            "failed": pytest_result.stdout.count("FAILED"),
            "errors": pytest_result.stdout.count("ERROR"),
            "warnings": pytest_result.stdout.count("WARN"),
            "ruff_violations": len(ruff_violations),
        }
    except Exception as e:
        st.session_state.project_state["errors"].append(
            f"An error occurred during testing: {str(e)}"
        )
        return {}


def generate_readme(
    project_name, user_request, project_type, refined_output, refiner_model
):
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
    try:
        readme_content = call_model(
            refiner_model, messages, temperature=0.7, max_tokens=2000
        )
        return readme_content
    except Exception as e:
        st.session_state.project_state["errors"].append(
            f"An error occurred during README generation: {str(e)}"
        )
        return ""


def execute_code(code: str, project_type: str) -> str:
    """Executes the generated code based on the project type."""
    if project_type == "Command-line Tool":
        # For command-line tools, execute the code in a subprocess and capture the output
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".py"
        ) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        try:
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
            )
            return result.stdout
        except Exception as e:
            return f"Error executing command-line tool: {str(e)}"
        finally:
            os.remove(temp_file_path)
    elif project_type == "Streamlit App":
        # For Streamlit apps, save the code to a file and run it with Streamlit
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".py"
        ) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        try:
            subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", temp_file_path]
            )
            return "Streamlit app launched in a new window."
        except Exception as e:
            return f"Error launching Streamlit app: {str(e)}"
        finally:
            os.remove(temp_file_path)
    elif project_type == "API":
        # For APIs, you'll need to determine how to execute them based on the framework used
        # For example, if using Flask, you could save the code to a file and run it with Flask
        return "API execution not yet implemented."
    elif project_type == "Data Analysis Script":
        # For data analysis scripts, execute the code in a subprocess and capture the output
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".py"
        ) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        try:
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
            )
            return result.stdout
        except Exception as e:
            return f"Error executing data analysis script: {str(e)}"
        finally:
            os.remove(temp_file_path)
    else:
        return "Unsupported project type for execution."


def build_interface():
    os.environ["USER_AGENT"] = "BuildAgent/1.0"

    st.title("🔨 Build: Autonomous Multi-Agent Software Development System")

    # Initialize session state variables
    if "project_state" not in st.session_state:
        st.session_state.project_state = {
            "status": "Not Started",
            "current_step": "",
            "iterations": 0,
            "max_iterations": 3,
            "code": "",
            "documentation": "",
            "test_results": {},
            "quality_review": "",
            "project_dir": "",
            "errors": [],
            "warnings": [],
            "agent_logs": [],
            "progress": 0,
        }

    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()

    if "api_keys" not in st.session_state:
        st.session_state.api_keys = load_api_keys()

    # Get all models
    all_models = get_all_models()

    # Get available Groq models
    available_groq_models = get_available_groq_models(st.session_state.api_keys)

    # Update all_models to include only available Groq models
    all_models = [model for model in all_models if model not in GROQ_MODELS or model in available_groq_models]

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Sidebar
    with st.sidebar:

        with st.expander("🤖 Model Selection"):
            
            def get_valid_model(model_key, default_index=0):
                saved_model = st.session_state.settings.get(model_key)
                try:
                    index = all_models.index(saved_model)
                except ValueError:
                    index = default_index
                return st.selectbox(
                    f"{model_key.replace('_', ' ').title()}",
                    all_models,
                    index=index,
                )

            # Manager settings
            st.session_state.settings["manager_model"] = get_valid_model("manager_model")
            st.session_state.settings["manager_temperature"] = st.slider(
                "Manager Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.settings.get("manager_temperature", 0.2),
                step=0.1,
            )
            st.session_state.settings["manager_max_tokens"] = st.slider(
                "Manager Max Tokens",
                min_value=1000,
                max_value=128000,
                value=st.session_state.settings.get("manager_max_tokens", 8000),
                step=1000,
            )

            # Subagent settings
            st.session_state.settings["subagent_model"] = get_valid_model("subagent_model")
            st.session_state.settings["subagent_temperature"] = st.slider(
                "Subagent Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.settings.get("subagent_temperature", 0.2),
                step=0.1,
            )
            st.session_state.settings["subagent_max_tokens"] = st.slider(
                "Subagent Max Tokens",
                min_value=1000,
                max_value=128000,
                value=st.session_state.settings.get("subagent_max_tokens", 8000),
                step=1000,
            )

            # Refiner settings
            st.session_state.settings["refiner_model"] = get_valid_model("refiner_model")
            st.session_state.settings["refiner_temperature"] = st.slider(
                "Refiner Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.settings.get("refiner_temperature", 0.2),
                step=0.1,
            )
            st.session_state.settings["refiner_max_tokens"] = st.slider(
                "Refiner Max Tokens",
                min_value=1000,
                max_value=128000,
                value=st.session_state.settings.get("refiner_max_tokens", 8000),
                step=1000,
            )

        if st.button("💾 Save Settings"):
            save_settings(st.session_state.settings)
            st.success("Settings saved successfully!")

    input_method = st.radio(
        "Choose input method:", ["Enter Project Request", "Provide Repository Directory"]
    )

    if input_method == "Enter Project Request":
        user_request = st.text_area(
            "Enter your project request:", height=100, key="user_request"
        )
        project_type = st.selectbox(
            "Select Project Type",
            ["Command-line Tool", "Streamlit App", "API", "Data Analysis Script"],
            key="project_type",
        )
        repo_contents = None
    else:
        repo_dir = st.text_input(
            "Enter the path to your repository directory:"
        )
        if repo_dir and os.path.isdir(repo_dir):
            repo_contents = dump_repository(repo_dir)
            st.success(
                f"Repository loaded successfully. Found {len(repo_contents)} files."
            )
        else:
            repo_contents = None
            if repo_dir:
                st.error(
                    "Invalid directory path. Please enter a valid directory."
                )
        user_request = None
        project_type = None

    use_search = st.checkbox("Use search functionality")
    if use_search:
        search_method = st.selectbox(
            "Select search method", ["duckduckgo", "google", "serpapi"]
        )

        # Display current API keys and allow editing
        if search_method == "google":
            st_ss.api_keys["google_api_key"] = st.text_input(
                "Google API Key",
                value=st_ss.api_keys.get("google_api_key", ""),
                type="password",
            )
            st_ss.api_keys["google_cse_id"] = st.text_input(
                "Google Custom Search Engine ID",
                value=st_ss.api_keys.get("google_cse_id", ""),
                type="password",
            )
        elif search_method == "serpapi":
            st_ss.api_keys["serpapi_api_key"] = st.text_input(
                "SerpApi API Key",
                value=st_ss.api_keys.get("serpapi_api_key", ""),
                type="password",
            )

        # Add num_results input
        num_results = st.number_input(
            "Number of search results", min_value=1, max_value=20, value=5
        )

        # Save API keys if they have changed
        save_api_keys(st_ss.api_keys)
    else:
        search_method = None
        num_results = None

    agent_logs = st.empty()

    (
        tab1,
        tab2,
        tab3,
        tab4,
        tab5,
    ) = st.tabs(
        [
            "Code",
            "Documentation",
            "Test Results",
            "Quality Review",
            "Execution",
        ]
    )

    if st.button("🚀 Build Project"):
        if user_request or repo_contents:
            console.print(
                f"Starting new project build", style="bold white on blue"
            )
            if user_request:
                console.print(f"User Request: {user_request}", style="cyan")
                console.print(f"Project Type: {project_type}", style="cyan")
            else:
                console.print(
                    f"Repository loaded with {len(repo_contents)} files",
                    style="cyan",
                )

            st_ss.project_state["status"] = "In Progress"
            st_ss.project_state["agent_logs"] = []
            st_ss.project_state["progress"] = 0
            st_ss.project_state["iterations"] = 0
            st_ss.project_state["errors"].clear()

            search_results = None
            if use_search:
                search_query = user_request or "Repository improvement techniques"
                search_results = perform_search(
                    search_query, search_method, st_ss.api_keys, num_results
                )
                console.print(
                    Panel(
                        f"Search Results: {json.dumps(search_results, indent=2)}",
                        title="[bold green]Search Results[/bold green]",
                        title_align="left",
                        border_style="green",
                    )
                )

            task_exchanges = []
            worker_tasks = []

            while (
                st_ss.project_state["iterations"]
                < st_ss.project_state["max_iterations"]
            ):
                previous_results = [result for _, result in task_exchanges]
                if not task_exchanges:
                    agent_result, file_content_for_worker = manage_task(
                        user_request or "Analyze and improve the provided repository",
                        model=st.session_state.settings["manager_model"],
                        file_content=repo_contents,
                        previous_results=previous_results,
                        use_search=use_search,
                        temperature=st.session_state.settings["manager_temperature"],
                        max_tokens=st.session_state.settings["manager_max_tokens"],
                        search_results=search_results,
                        groq_api_key=st.session_state.settings.get("groq_api_key"),
                        openai_api_key=st.session_state.api_keys.get("openai_api_key"),
                        api_keys=st.session_state.api_keys,
                    )
                else:
                    agent_result, _ = manage_task(
                        user_request
                        or "Analyze and improve the provided repository",
                        model=st_ss.settings["manager_model"],
                        previous_results=previous_results,
                        use_search=use_search,
                        temperature=st.session_state.settings["manager_temperature"],
                        max_tokens=st.session_state.settings["manager_max_tokens"],
                        search_results=search_results,
                        groq_api_key=st.session_state.settings.get("groq_api_key"),
                        openai_api_key=st.session_state.api_keys.get("openai_api_key"),
                        api_keys=st.session_state.api_keys,
                    )

                if "The task is complete:" in agent_result:
                    final_output = agent_result.replace(
                        "The task is complete:", ""
                    ).strip()
                    break
                else:
                    sub_task_prompt = agent_result
                    if file_content_for_worker and not worker_tasks:
                        sub_task_prompt = f"{sub_task_prompt}\n\nFile content:\n{file_content_for_worker}"
                    sub_task_result = sub_agent_task(
                        sub_task_prompt,
                        model=st.session_state.settings["subagent_model"],
                        previous_tasks=worker_tasks,
                        use_search=use_search,
                        temperature=st.session_state.settings["subagent_temperature"],
                        max_tokens=st.session_state.settings["subagent_max_tokens"],
                        search_results=search_results,
                        groq_api_key=st.session_state.settings.get("groq_api_key"),
                        openai_api_key=st.session_state.api_keys.get("openai_api_key"),
                    )
                    worker_tasks.append(
                        {"task": sub_task_prompt, "result": sub_task_result}
                    )
                    task_exchanges.append(
                        (sub_task_prompt, sub_task_result)
                    )
                    file_content_for_worker = None

                st_ss.project_state["iterations"] += 1
                st_ss.project_state["progress"] = (
                    st_ss.project_state["iterations"]
                    / st_ss.project_state["max_iterations"]
                )
                progress_bar.progress(st_ss.project_state["progress"])
                status_text.text(
                    f"Iteration {st_ss.project_state['iterations']} of {st_ss.project_state['max_iterations']}"
                )

            sanitized_objective = re.sub(
                r"\W+",
                "_",
                user_request if user_request else "repository_improvement",
            )
            timestamp = datetime.now().strftime("%H-%M-%S")
            refined_output = refine_task(
                user_request or "Analyze and improve the provided repository",
                model=st.session_state.settings["refiner_model"],
                sub_task_results=[result for _, result in task_exchanges],
                filename=timestamp,
                projectname=sanitized_objective,
                temperature=st.session_state.settings["refiner_temperature"],
                max_tokens=st.session_state.settings["refiner_max_tokens"],
                groq_api_key=st.session_state.settings.get("groq_api_key"),
                openai_api_key=st.session_state.api_keys.get("openai_api_key"),
            )

            project_name_match = re.search(
                r"Project Name: (.*)", refined_output
            )
            project_name = (
                project_name_match.group(1).strip()
                if project_name_match
                else sanitized_objective
            )

            folder_structure = parse_folder_structure(refined_output)
            code_blocks = extract_code_blocks(refined_output)

            project_dir = (
                Path("generated_projects") / f"{project_name}_{timestamp}"
            )
            project_dir.mkdir(parents=True, exist_ok=True)
            console.print(
                Panel(
                    f"Created project folder: [bold]{project_dir}[/bold]",
                    title="[bold green]Project Folder[/bold green]",
                    title_align="left",
                    border_style="green",
                )
            )

            if code_blocks:
                for filename, code in code_blocks.items():
                    save_file(code, filename, str(project_dir))
                    console.print(
                        Panel(
                            f"Created file: [bold]{filename}[/bold]",
                            title="[bold green]Project Files[/bold green]",
                            title_align="left",
                            border_style="yellow",
                        )
                    )

            # Generate README.md
            readme_content = generate_readme(
                project_name,
                user_request,
                project_type,
                refined_output,
                st_ss.settings["refiner_model"],
            )
            readme_path = project_dir / "README.md"
            readme_path.write_text(readme_content, encoding="utf-8")
            console.print(
                Panel(
                    f"Created README.md: [bold]{readme_path}[/bold]",
                    title="[bold green]README.md Created[/bold green]",
                    title_align="left",
                    border_style="green",
                )
            )

            st_ss.project_state["documentation"] = readme_content

            # Run tests
            test_results = run_tests(project_dir)
            st_ss.project_state["test_results"] = test_results

            # Update current task (you might want to define this based on the current state of the project)
            current_task = "Implement initial project structure"

            # Call manager agent
            manager_response = manager_agent_task(
                create_agent_context(
                    st.session_state.project_state,
                    st.session_state.project_state["test_results"],
                    current_task,
                ),
                st.session_state.settings["manager_model"],
                st.session_state.settings["manager_temperature"],
                st.session_state.settings["manager_max_tokens"],
                st.session_state.settings.get("groq_api_key"),
                st.session_state.api_keys.get("openai_api_key"),
            )

            if "analysis" not in manager_response:
                st.error(
                    "Manager response is incomplete. Please check the logs for details."
                )
                st.json(manager_response)
            else:
                # Proceed with sub_agent_task
                sub_agent_response = sub_agent_task(
                    create_agent_context(
                        st.session_state.project_state,
                        st.session_state.project_state["test_results"],
                        current_task
                    ),
                    manager_response,
                    st.session_state.settings["subagent_model"],
                    st.session_state.settings["subagent_temperature"],
                    st.session_state.settings["subagent_max_tokens"],
                    st.session_state.settings.get("groq_api_key"),
                    st.session_state.api_keys.get("openai_api_key"),
                )

                # Process sub_agent_response here
                st.write("Sub-agent Response:")
                st.json(sub_agent_response)

                # Update project state with sub-agent results
                if sub_agent_response:  # Check if sub_agent_response is not empty
                    st.session_state.project_state["code"] = sub_agent_response.get(
                        "code_changes", ""
                    )
                    st.session_state.project_state["quality_review"] = (
                        sub_agent_response.get("explanation", "")
                    )

            # Check if there are any failed tests, errors, or Ruff violations
            if (
                test_results.get("failed", 0) > 0
                or test_results.get("errors", 0) > 0
                or test_results.get("ruff_violations", 0) > 0
            ):
                console.print(
                    Panel(
                        f"[bold red]Tests failed, encountered errors, or Ruff violations detected. Asking user for additional iterations.[/bold red]",
                        title="[bold red]Test Results[/bold red]",
                        title_align="left",
                        border_style="red",
                    )
                )
                st.warning(
                    "Tests failed, encountered errors, or Ruff violations detected. Click the button below to run additional iterations."
                )
                if st.button("Run Additional Iterations to Fix Issues"):
                    st_ss.project_state["max_iterations"] += 3
            else:
                st_ss.project_state["status"] = "Completed"
                st.session_state.project_state["progress"] = 1.0
                progress_bar.progress(1.0)
                st.success("Project build completed successfully!")
        else:
            st.error("Please enter a valid project request or repository directory.")

    with tab1:
        st.subheader("Code")
        if st.session_state.project_state["errors"]:
            st.error("Errors encountered during code generation:")
            for error in st.session_state.project_state["errors"]:
                st.write(error)
        st.text_area("Generated Code:", st.session_state.project_state["code"], height=400)

    with tab2:
        st.subheader("Documentation")
        st.text_area("Generated Documentation:", st.session_state.project_state["documentation"], height=400)

    with tab3:
        st.subheader("Test Results")
        st.text_area("Test Results:", st.session_state.project_state["test_results"].get("pytest_output", ""), height=400)

    with tab4:
        st.subheader("Quality Review")
        st.text_area("Quality Review:", st.session_state.project_state["quality_review"], height=400)

    with tab5:
        st.subheader("Execution")
        st.text_area("Execution Output:", "", height=400)


if __name__ == "__main__":
    build_interface()