import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import ollama
import pytest
from duckduckgo_search import DDGS
from googleapiclient.discovery import build
from ollama import Client
from openai import ChatCompletion, OpenAI
from rich.console import Console
from streamlit import session_state as st_ss

from openai_utils import *
from groq_utils import *
from ollama_utils import *
from external_providers import *

API_KEYS_FILE = "api_keys.json"
SETTINGS_FILE = "build_settings.json"

# Initialize the Rich Console
console = Console()

# Initialize the Ollama client
client = Client(host="http://localhost:11434")

def load_json_file(filepath: str) -> dict:
    """Loads JSON data from a file if it exists."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}

def save_json_file(filepath: str, data: dict) -> None:
    """Saves JSON data to a file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def load_api_keys() -> dict:
    """Loads API keys from the JSON file."""
    return load_json_file(API_KEYS_FILE)

def save_api_keys(api_keys: dict) -> None:
    """Saves API keys to the JSON file."""
    save_json_file(API_KEYS_FILE, api_keys)

def load_settings() -> dict:
    """Loads settings from the JSON file."""
    return load_json_file(SETTINGS_FILE)

def save_settings(settings: dict) -> None:
    """Saves settings to the JSON file."""
    save_json_file(SETTINGS_FILE, settings)

def set_openai_api_key(api_key: str) -> None:
    """Sets the OpenAI API key."""
    api_keys = load_api_keys()
    api_keys['openai_api_key'] = api_key
    save_api_keys(api_keys)
    console.print("[bold green]OpenAI API key has been set.[/bold green]")

def call_openai_api(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.7,
    max_tokens: int = 1000,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stream: bool = False,
    openai_api_key: str = None
) -> Any:
    """Wrapper function to call the OpenAI Chat API with a unified interface."""
    try:
        temperature = float(temperature)
        max_tokens = int(max_tokens)
        frequency_penalty = float(frequency_penalty)
        presence_penalty = float(presence_penalty)
    except ValueError as ve:
        raise ValueError(f"Invalid value for one of the numerical parameters: {ve}")

    if not openai_api_key or not isinstance(openai_api_key, str):
        api_keys = load_api_keys()
        openai_api_key = api_keys.get('openai_api_key')
        if not openai_api_key or not isinstance(openai_api_key, str):
            raise ValueError("Invalid or missing OpenAI API key.")

    openai.api_key = openai_api_key

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=stream
        )

        if stream:
            return response  # Return the stream object
        else:
            return response.choices[0].message['content'].strip()

    except json.JSONDecodeError as json_err:
        raise ValueError(f"Failed to parse response as JSON: {json_err}")
    except Exception as e:
        console.print(f"[bold red]Error calling OpenAI API:[/bold red] {e}")
        return "Error occurred while calling OpenAI API"

def is_groq_model(model_name: str) -> bool:
    """Determines if a given model name is a Groq model."""
    api_keys = load_api_keys()
    available_groq_models = get_available_groq_models(api_keys)
    return model_name in available_groq_models

def perform_search(
    query: str,
    search_method: str,
    api_keys: Dict[str, str],
    num_results: int = 5
) -> List[Dict[str, str]]:
    """Performs a web search using the specified method."""
    if search_method == "duckduckgo":
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        return [{"title": result["title"], "url": result["href"]} for result in results]

    elif search_method == "google":
        if "google_api_key" not in api_keys or "google_cse_id" not in api_keys:
            console.print("[bold red]Error: Google API Key or CSE ID not provided.[/bold red]")
            return []
        service = build("customsearch", "v1", developerKey=api_keys["google_api_key"])
        res = service.cse().list(q=query, cx=api_keys["google_cse_id"], num=num_results).execute()
        return [{"title": item["title"], "url": item["link"]} for item in res.get("items", [])]

    elif search_method == "serpapi":
        if "serpapi_api_key" not in api_keys:
            console.print("[bold red]Error: SerpAPI Key not provided.[/bold red]")
            return []
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": api_keys["serpapi_api_key"],
                "num": num_results,
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            organic_results = results.get("organic_results", [])
            return [{"title": result["title"], "url": result["link"]} for result in organic_results]
        except Exception as e:
            console.print(f"Error in SerpApi search: {str(e)}")
            return []

    else:
        console.print(f"[bold red]Unsupported search method: {search_method}[/bold red]")
        return []

def create_agent_context(
    project_state: dict,
    current_task: str
) -> dict:
    """Creates the context for agents based on the project state."""
    return {
        "project_state": {
            "status": project_state["status"],
            "current_step": project_state["current_step"],
            "iterations": project_state["iterations"],
            "max_iterations": project_state["max_iterations"],
        },
        "current_task": current_task,
        "previous_tasks": project_state.get("previous_tasks", []),
    }

def manager_agent_task(
    context: Dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int,
    openai_api_key=None
) -> Tuple[Dict[str, Any], Any]:
    """Handles the manager agent task based on the context."""
    prompt = f"""
    You are the manager agent in an Agile software development process. 
    Current project state: {json.dumps(context['project_state'], indent=2)}
    Current task: {context['current_task']}

    Based on the current state, please:
    1. Analyze the current project step.
    2. Update the work plan for the next sprint.
    3. Prioritize tasks for the coding agents.
    4. If repository files haven't been created yet, make this the top priority.

    Respond with a JSON object containing:
    1. "analysis": Your analysis of the current state.
    2. "work_plan": The updated work plan for the next sprint.
    3. "priorities": A list of prioritized tasks.
    4. "instructions": Clear instructions for the coding agents.
    5. "create_files": true if repository files should be created, false otherwise.

    Your response must be a valid JSON object.
    """

    try:
        temperature = float(temperature)
        max_tokens = int(max_tokens)

        if model.startswith("gpt-"):
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            response_content = response.choices[0].message.content
        elif model in GROQ_MODELS:
            response_content = call_groq_api(model, prompt, temperature, max_tokens, groq_api_key=openai_api_key)
        else:
            # Assume it's an Ollama model
            response_content, _, _, _ = call_ollama_endpoint(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

        parsed_response = json.loads(response_content)
        return parsed_response, None

    except json.JSONDecodeError as e:
        error_msg = f"Error parsing JSON response from Manager Agent: {e}"
        console.print(error_msg)
        console.print(f"Raw response that caused the error: {response_content}")
        return {
            "analysis": "Error parsing response. Please review the raw output.",
            "work_plan": "Unable to generate work plan due to parsing error.",
            "priorities": ["Review and fix JSON parsing issues"],
            "instructions": "Please review the raw output and manually extract relevant information.",
        }, None
    except Exception as e:
        error_msg = f"An error occurred during the manager agent task: {str(e)}"
        console.print(error_msg)
        return {}, None

def create_repository_files(project_dir: Path, refined_output: str) -> Dict[str, str]:
    """
    Creates repository files based on the refined output.
    Returns a dictionary of filenames and their contents.
    """
    created_files = {}
    
    # Extract folder structure
    folder_structure = parse_folder_structure(refined_output)
    if folder_structure:
        create_folder_structure(project_dir, folder_structure)
    
    # Extract and create code files
    code_blocks = extract_code_blocks(refined_output)
    for filename, code in code_blocks.items():
        file_path = project_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(code, encoding="utf-8")
        created_files[str(file_path.relative_to(project_dir))] = code
    
    return created_files

def coding_agent_task(
    prompt: str,
    model: str,
    previous_tasks=None,
    use_search=False,
    continuation=False,
    temperature: float = 0.2,
    max_tokens: int = 8000,
    search_results=None,
    groq_api_key=None,
    openai_api_key=None
) -> Dict[str, Any]:
    """Handles the coding agent task based on the provided prompt and context."""
    
    if previous_tasks is None:
        previous_tasks = []

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
        if previous_tasks_str
        else "No previous sub-agent tasks."
    )
    if continuation:
        prompt = continuation_prompt

    full_prompt = f"{previous_tasks_summary}\n\n{prompt}"
    if use_search and search_results:
        full_prompt += f"\n\nSearch Results:\n{json.dumps(search_results, indent=2)}"

    if not full_prompt.strip():
        raise ValueError("Prompt cannot be empty")

    full_prompt += "\n\nRespond with a JSON object containing your implementation details and any necessary explanations. Your response must be a valid JSON object."

    messages = [{"role": "user", "content": full_prompt}]

    try:
        temperature = float(temperature)
        max_tokens = int(max_tokens)

        if model.startswith("gpt-"):
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            response_content = response.choices[0].message.content
        elif model in GROQ_MODELS:
            response_content = call_groq_api(model, full_prompt, temperature, max_tokens, groq_api_key=groq_api_key)
        else:
            # Assume it's an Ollama model
            response_content, _, _, _ = call_ollama_endpoint(
                model=model,
                prompt=full_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

        parsed_response = json.loads(response_content)
        return parsed_response

    except json.JSONDecodeError as e:
        error_msg = f"Error parsing JSON response from Coding Agent: {e}"
        console.print(error_msg)
        console.print(f"Raw response that caused the error: {response_content}")
        return {"error": error_msg, "raw_response": response_content}
    except Exception as e:
        error_msg = f"An error occurred during the coding agent task: {str(e)}"
        console.print(error_msg)
        return {"error": error_msg}

def refine_task(
    objective: str,
    model: str,
    sub_task_results: List[str],
    filename: str,
    projectname: str,
    continuation=False,
    temperature: float = 0.2,
    max_tokens: int = 8000,
    groq_api_key=None,
    openai_api_key=None
) -> str:
    """Handles the task refinement process for the final output."""
    console.print("\nCalling Refiner to provide the refined final output for your objective:")
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
        if isinstance(model, str) and model.startswith("gpt-"):
            response_text = call_openai_api(model, messages, temperature, max_tokens, openai_api_key=openai_api_key)
        elif isinstance(model, str) and is_groq_model(model):
            response_text = call_groq_api(model, messages[0]["content"], temperature, max_tokens, groq_api_key=groq_api_key)
        else:
            response_text = call_ollama_endpoint(
                model=model,
                prompt=messages[0]["content"],
                temperature=temperature,
                max_tokens=max_tokens
            )

        # Ensure response_text is a string
        if isinstance(response_text, tuple):
            response_text = response_text[0]
        
        if not isinstance(response_text, str):
            response_text = str(response_text)

        if len(response_text) >= 8000 and not continuation:
            console.print("Warning: Output may be truncated. Attempting to continue the response.")
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

        return response_text

    except Exception as e:
        error_msg = f"An error occurred during the refine task: {str(e)}"
        console.print(error_msg)
        return error_msg

def parse_folder_structure(structure_string: str) -> dict:
    """Parses the folder structure from the response string."""
    structure_string = re.sub(r"\s+", " ", structure_string)
    match = re.search(r"<folder_structure>(.*?)</folder_structure>", structure_string)
    if not match:
        return None

    json_string = match.group(1)

    try:
        structure = json.loads(json_string)
        return structure
    except json.JSONDecodeError as e:
        console.print(f"[bold red]Error parsing JSON:[/bold red] {e}")
        console.print(f"[bold red]Invalid JSON string:[/bold red] {json_string}")
        return None

def extract_code_blocks(refined_output: str) -> Dict[str, str]:
    """Extracts code blocks from the refined output."""
    code_blocks = {}
    pattern = r"Filename: (.*?)\n```(.*?)\n```"
    matches = re.finditer(pattern, refined_output, re.DOTALL)
    for match in matches:
        filename = match.group(1).strip()
        code = match.group(2).strip()
        code_blocks[filename] = code
    return code_blocks

def save_file(content: str, filename: str, project_dir: str) -> None:
    """Saves content to a file within the specified project directory."""
    try:
        file_path = Path(project_dir) / "code" / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        console.print(f"[bold green]Saved file: {file_path}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error saving file '{filename}': {str(e)}[/bold red]")

def dump_repository(repo_path: str) -> Dict[str, str]:
    """Dumps the contents of a repository to a dictionary."""
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
                        console.print(f"[bold yellow]Skipping binary file: {file_path}[/bold yellow]")
    return repo_contents

def generate_test_cases(code_analysis: Dict[str, List[str]]) -> str:
    """Generates test cases based on the analyzed code."""
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
    """Runs tests in the project directory and returns the results."""
    test_dir = os.path.join(project_dir, "tests")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

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
        pytest_result = subprocess.run(
            ["pytest", "-v", test_dir], capture_output=True, text=True
        )

        return {
            "pytest_output": pytest_result.stdout,
            "passed": pytest_result.stdout.count("PASSED"),
            "failed": pytest_result.stdout.count("FAILED"),
            "errors": pytest_result.stdout.count("ERROR"),
            "warnings": pytest_result.stdout.count("WARN"),
        }
    except Exception as e:
        console.print(f"[bold red]An error occurred during testing: {str(e)}[/bold red]")
        return {}

def generate_readme(
    project_name: str,
    user_request: str,
    project_type: str,
    refined_output: str,
    refiner_model: str
) -> str:
    """Generates a README.md file for the project."""
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
        api_keys = load_api_keys()
        if refiner_model.startswith("gpt-"):
            readme_content = call_openai_api(
                model=refiner_model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                openai_api_key=api_keys.get("openai_api_key")
            )
        elif is_groq_model(refiner_model):
            readme_content = call_groq_api(
                model=refiner_model,
                prompt=messages[0]["content"],
                temperature=0.7,
                max_tokens=2000,
                groq_api_key=api_keys.get("groq_api_key")
            )
        else:
            readme_content = call_ollama_endpoint(
                model=refiner_model,
                prompt=messages[0]["content"],
                temperature=0.7,
                max_tokens=2000
            )

        # Ensure readme_content is a string
        if isinstance(readme_content, tuple):
            readme_content = readme_content[0]
        
        if not isinstance(readme_content, str):
            readme_content = str(readme_content)

        return readme_content
    except Exception as e:
        error_msg = f"An error occurred during README generation: {str(e)}"
        console.print(error_msg)
        return f"# README\n\nError generating README: {error_msg}\n\nPlease check the API connection and try again."
    
def execute_code(code: str, project_type: str) -> str:
    """Executes the generated code based on the project type."""
    if project_type == "Command-line Tool":
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".py") as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        try:
            result = subprocess.run(
                [sys.executable, temp_file_path], capture_output=True, text=True
            )
            return result.stdout
        except Exception as e:
            return f"Error executing command-line tool: {str(e)}"
        finally:
            os.remove(temp_file_path)

    elif project_type == "Streamlit App":
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".py") as temp_file:
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
        return "API execution not yet implemented."

    elif project_type == "Data Analysis Script":
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".py") as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        try:
            result = subprocess.run(
                [sys.executable, temp_file_path], capture_output=True, text=True
            )
            return result.stdout
        except Exception as e:
            return f"Error executing data analysis script: {str(e)}"
        finally:
            os.remove(temp_file_path)

    else:
        return "Unsupported project type for execution."

def build_project(user_request: str, repo_contents: dict, project_type: str) -> Dict[str, Any]:
    """
    Implements the core build logic without running tests.

    Args:
        user_request (str): The user's project request.
        repo_contents (dict): The contents of the repository.
        project_type (str): The type of the project.

    Returns:
        dict: A dictionary containing the build results.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_dir = Path("generated_projects") / f"project_{timestamp}"
        project_dir.mkdir(parents=True, exist_ok=True)

        return {"success": True, "project_dir": project_dir}
    except Exception as e:
        return {"success": False, "error": str(e)}

def build_interface() -> None:
    """The main interface for the build system."""
    os.environ["USER_AGENT"] = "BuildAgent/1.0"

    st.title("🔨 Build: Autonomous Multi-Agent Software Development System")

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
            "previous_tasks": [],
        }

    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()

    if "api_keys" not in st.session_state:
        st.session_state.api_keys = load_api_keys()

    all_models = get_all_models()
    available_groq_models = get_available_groq_models(st.session_state.api_keys)
    all_models = [
        model for model in all_models if model not in GROQ_MODELS or model in available_groq_models
    ]

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
                    f"{model_key.replace('_', ' ').title()}", all_models, index=index
                )

            # Manager settings
            st.session_state.settings["manager_model"] = get_valid_model("manager_model")
            st.session_state.settings["manager_temperature"] = st.slider(
                "Manager Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.settings.get("manager_temperature", 0.2)),
                step=0.1,
            )
            st.session_state.settings["manager_max_tokens"] = st.slider(
                "Manager Max Tokens",
                min_value=1000,
                max_value=128000,
                value=int(st.session_state.settings.get("manager_max_tokens", 8000)),
                step=1000,
            )

            # Subagent settings
            st.session_state.settings["subagent_model"] = get_valid_model("subagent_model")
            st.session_state.settings["subagent_temperature"] = st.slider(
                "Subagent Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.settings.get("subagent_temperature", 0.2)),
                step=0.1,
            )
            st.session_state.settings["subagent_max_tokens"] = st.slider(
                "Subagent Max Tokens",
                min_value=1000,
                max_value=128000,
                value=int(st.session_state.settings.get("subagent_max_tokens", 8000)),
                step=1000,
            )

            # Refiner settings
            st.session_state.settings["refiner_model"] = get_valid_model("refiner_model")
            st.session_state.settings["refiner_temperature"] = st.slider(
                "Refiner Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.settings.get("refiner_temperature", 0.2)),
                step=0.1,
            )
            st.session_state.settings["refiner_max_tokens"] = st.slider(
                "Refiner Max Tokens",
                min_value=1000,
                max_value=128000,
                value=int(st.session_state.settings.get("refiner_max_tokens", 8000)),
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
        repo_dir = st.text_input("Enter the path to your repository directory:")
        if repo_dir and os.path.isdir(repo_dir):
            repo_contents = dump_repository(repo_dir)
            st.success(
                f"Repository loaded successfully. Found {len(repo_contents)} files."
            )
        else:
            repo_contents = None
            if repo_dir:
                st.error("Invalid directory path. Please enter a valid directory.")
        user_request = None
        project_type = None

    use_search = st.checkbox("Use search functionality")
    if use_search:
        search_method = st.selectbox(
            "Select search method", ["duckduckgo", "google", "serpapi"]
        )

        if search_method == "google":
            st.session_state.api_keys["google_api_key"] = st.text_input(
                "Google API Key",
                value=st.session_state.api_keys.get("google_api_key", ""),
                type="password",
            )
            st.session_state.api_keys["google_cse_id"] = st.text_input(
                "Google Custom Search Engine ID",
                value=st.session_state.api_keys.get("google_cse_id", ""),
                type="password",
            )
        elif search_method == "serpapi":
            st.session_state.api_keys["serpapi_api_key"] = st.text_input(
                "SerpApi API Key",
                value=st.session_state.api_keys.get("serpapi_api_key", ""),
                type="password",
            )

        num_results = st.number_input(
            "Number of search results", min_value=1, max_value=20, value=5
        )

        save_api_keys(st.session_state.api_keys)
    else:
        search_method = None
        num_results = None

    (tab1, tab2, tab3) = st.tabs(
        ["Workflow", "Documentation", "Test Results"]
    )

    if st.button("🚀 Build Project"):
        if user_request or repo_contents:
            st.info(f"Starting new project build")
            if user_request:
                st.info(f"User Request: {user_request}")
                st.info(f"Project Type: {project_type}")
            else:
                st.info(f"Repository loaded with {len(repo_contents)} files")

            st.session_state.project_state["status"] = "In Progress"
            st.session_state.project_state["agent_logs"] = []
            st.session_state.project_state["progress"] = 0
            st.session_state.project_state["iterations"] = 0
            st.session_state.project_state["errors"].clear()

            # Create project directory
            sanitized_objective = re.sub(r"\W+", "_", user_request if user_request else "repository_improvement")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_dir = Path("generated_projects") / f"{sanitized_objective}_{timestamp}"
            project_dir.mkdir(parents=True, exist_ok=True)
            st.success(f"Created project folder: {project_dir}")

            search_results = None
            if use_search:
                search_query = user_request or "Repository improvement techniques"
                search_results = perform_search(
                    search_query, search_method, st.session_state.api_keys, num_results
                )
                st.json(search_results)

            task_exchanges = []
            worker_tasks = []
            refined_output = ""

            while (
                st.session_state.project_state["iterations"] < st.session_state.project_state["max_iterations"]
            ):
                manager_response, _ = manager_agent_task(
                    create_agent_context(
                        st.session_state.project_state,
                        user_request or "Analyze and improve the provided repository",
                    ),
                    model=st.session_state.settings["manager_model"],
                    temperature=float(st.session_state.settings["manager_temperature"]),
                    max_tokens=int(st.session_state.settings["manager_max_tokens"]),
                    openai_api_key=st.session_state.api_keys.get("openai_api_key"),
                )

                st.subheader(f"Manager Response - Iteration {st.session_state.project_state['iterations'] + 1}")
                st.json(manager_response)

                if manager_response.get("create_files", False):
                    st.subheader("Creating Repository Files")
                    created_files = create_repository_files(project_dir, refined_output)
                    st.success(f"Created {len(created_files)} repository files")
                    for filename, content in created_files.items():
                        st.text(f"Created file: {filename}")
                        st.code(content, language="python")
                else:
                    sub_agent_response = coding_agent_task(
                        manager_response["instructions"],
                        model=st.session_state.settings["subagent_model"],
                        previous_tasks=worker_tasks,
                        use_search=use_search,
                        temperature=float(st.session_state.settings["subagent_temperature"]),
                        max_tokens=int(st.session_state.settings["subagent_max_tokens"]),
                        search_results=search_results,
                        openai_api_key=st.session_state.api_keys.get("openai_api_key"),
                    )

                    st.subheader(f"Sub-agent Response - Iteration {st.session_state.project_state['iterations'] + 1}")
                    st.json(sub_agent_response)

                    worker_tasks.append({"task": manager_response["instructions"], "result": sub_agent_response})
                    task_exchanges.append((manager_response["instructions"], sub_agent_response))

                st.session_state.project_state["iterations"] += 1
                st.session_state.project_state["progress"] = min(
                    st.session_state.project_state["iterations"]
                    / st.session_state.project_state["max_iterations"],
                    1.0
                )
                progress_bar.progress(st.session_state.project_state["progress"])
                status_text.text(
                    f"Iteration {st.session_state.project_state['iterations']} of {st.session_state.project_state['max_iterations']}"
                )

            # Refiner task
            st.subheader("Refiner Task")
            refined_output = refine_task(
                user_request or "Analyze and improve the provided repository",
                model=st.session_state.settings["refiner_model"],
                sub_task_results=[json.dumps(result) if isinstance(result, dict) else str(result) for _, result in task_exchanges],
                filename=timestamp,
                projectname=sanitized_objective,
                temperature=float(st.session_state.settings["refiner_temperature"]),
                max_tokens=int(st.session_state.settings["refiner_max_tokens"]),
                openai_api_key=st.session_state.api_keys.get("openai_api_key"),
            )

            if isinstance(refined_output, tuple):
                refined_output = refined_output[0]

            st.text_area("Refined Output:", refined_output, height=400)

            project_name_match = re.search(r"Project Name: (.*)", refined_output)
            project_name = (
                project_name_match.group(1).strip()
                if project_name_match
                else sanitized_objective
            )

            folder_structure = parse_folder_structure(refined_output)
            code_blocks = extract_code_blocks(refined_output)

            project_dir = Path("generated_projects") / f"{project_name}_{timestamp}"
            project_dir.mkdir(parents=True, exist_ok=True)
            st.success(f"Created project folder: {project_dir}")

            # Create folder structure
            if folder_structure:
                create_folder_structure(project_dir, folder_structure)
                st.success("Created folder structure")

            # Create code files
            if code_blocks:
                for filename, code in code_blocks.items():
                    file_path = project_dir / filename
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(code, encoding="utf-8")
                    st.success(f"Created file: {file_path}")

            # Generate README.md
            readme_content = generate_readme(
                project_name,
                user_request,
                project_type,
                refined_output,
                st.session_state.settings["refiner_model"],
            )

            # Ensure readme_content is a string
            if isinstance(readme_content, tuple):
                readme_content = readme_content[0]

            if not isinstance(readme_content, str):
                readme_content = str(readme_content)

            readme_path = project_dir / "README.md"
            readme_path.write_text(readme_content, encoding="utf-8")
            st.success(f"Created README.md: {readme_path}")

            st.session_state.project_state["documentation"] = readme_content

            # Option to run tests separately
            if st.button("🔍 Run Tests"):
                test_results = run_tests(project_dir)
                st.session_state.project_state["test_results"] = test_results

                if (
                    test_results.get("failed", 0) > 0
                    or test_results.get("errors", 0) > 0
                ):
                    st.warning("Tests failed or errors encountered. Review the results and consider additional iterations.")
                else:
                    st.success("All tests passed successfully!")

            st.session_state.project_state["status"] = "Completed"
            st.session_state.project_state["progress"] = 1.0
            progress_bar.progress(1.0)
            st.success("Project build completed successfully!")

            # Display list of generated files
            st.subheader("Generated Files")
            for root, dirs, files in os.walk(project_dir):
                for file in files:
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(project_dir)
                    st.markdown(f"[{relative_path}]({file_path})")

        else:
            st.error("Please enter a valid project request or repository directory.")

    with tab1:
        st.subheader("Workflow")
        for i, (task, response) in enumerate(st.session_state.project_state.get("agent_logs", [])):
            st.subheader(f"Iteration {i+1}")
            st.text_area(f"Task {i+1}", task, height=100)
            st.text_area(f"Response {i+1}", json.dumps(response, indent=2), height=200)

    with tab2:
        st.subheader("Documentation")
        st.text_area(
            "Generated Documentation:", st.session_state.project_state["documentation"], height=400
        )

    with tab3:
        st.subheader("Test Results")
        st.text_area(
            "Test Results:",
            st.session_state.project_state["test_results"].get("pytest_output", ""),
            height=400,
        )

    # Display any errors
    if st.session_state.project_state["errors"]:
        st.header("Errors")
        for error in st.session_state.project_state["errors"]:
            st.error(error)

# Add this function to create the folder structure
def create_folder_structure(base_path: Path, structure: dict) -> None:
    for name, content in structure.items():
        path = base_path / name
        if content is None:
            path.touch()
        elif isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            create_folder_structure(path, content)

def main():
    try:
        build_interface()
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {str(e)}[/bold red]")
        raise e

if __name__ == "__main__":
    main()
