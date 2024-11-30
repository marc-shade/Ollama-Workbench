# repo_docs.py
import os
import requests
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from fpdf import FPDF
import tempfile
import queue
import re
from datetime import datetime
from radon.complexity import cc_visit, cc_rank
from radon.metrics import mi_visit, h_visit
from flake8.api import legacy as flake8
from ollama_utils import get_available_models as get_ollama_models
from ollama_utils import load_api_keys
from openai_utils import OPENAI_MODELS
from groq_utils import GROQ_MODELS

# Settings file for model settings
MODEL_SETTINGS_FILE = "model_settings.json"

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Repository Analysis', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title.encode('latin1', 'replace').decode('latin1'), 0, 1, 'L')
        self.ln(10)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        body = body.encode('latin1', 'replace').decode('latin1')
        self.multi_cell(0, 10, body)
        self.ln()

    def add_chapter(self, title, body):
        self.add_page()
        self.chapter_title(title)
        self.chapter_body(body)

def call_ollama_endpoint(model, prompt, temperature=0.5, max_tokens=150, presence_penalty=0.0, frequency_penalty=0.0, context=None):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "context": context,
        "stream": True,
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)
        response.raise_for_status()

        full_response = ""
        eval_count = 0
        eval_duration = 0
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                try:
                    json_response = json.loads(decoded_line)
                    if 'response' in json_response:
                        full_response += json_response['response']
                    if 'eval_count' in json_response:
                        eval_count = json_response['eval_count']
                    if 'eval_duration' in json_response:
                        eval_duration = json_response['eval_duration']
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {decoded_line}")

        return full_response.strip(), None, eval_count, eval_duration

    except requests.exceptions.RequestException as e:
        print(f"Error making request to {url}: {e}")
        return f"Error calling Ollama endpoint: {str(e)}", None, 0, 0

@st.cache_data
def get_available_models():
    ollama_models = get_ollama_models()
    all_models = ollama_models + OPENAI_MODELS + GROQ_MODELS
    return all_models

def generate_documentation_stream(file_content, task_type, model, temperature, max_tokens, repo_info=None):
    if task_type == "documentation":
        prompt = f"""
You are an expert in programming and technical writing. Your task is to generate comprehensive documentation and insightful commentary for the provided code. 

Follow these steps:

1. **Overview**: Provide a high-level summary of what the code does, its purpose, its main components, and how it connects to the other scripts in this repository.
2. **Code Walkthrough**: Go through the code section by section, explaining the functionality of each part. Highlight key functions, classes, and methods.
3. **Best Practices**: Identify any non-idiomatic practices or areas where the code could be improved. Suggest best practices and optimizations.
4. **Examples**: Where applicable, include example usages or scenarios that demonstrate how the code should be used.
5. **Formatting**: Ensure that the documentation is well-organized, clearly formatted, and easy to read. Use bullet points, headers, and code blocks where necessary.

Generate a detailed and neatly formatted documentation for the following code:

{file_content}
"""
    elif task_type == "debug":
        prompt = f"""
You are a highly experienced code debugger with a deep understanding of best practices and coding standards. Your task is to thoroughly analyze the provided code, identify any issues, and offer recommendations for improvements. Follow these steps:

1. **Identify Issues**: Go through the code line by line and identify syntax errors, logical errors, performance issues, and any non-idiomatic practices.
2. **Provide Recommendations**: For each issue identified, provide a detailed explanation of why it is an issue and how it can be fixed or improved.
3. **Code Examples**: Whenever possible, include code snippets to illustrate the recommended changes or improvements.
4. **Formatting**: Ensure that your debug report is well-organized, clearly formatted, and easy to read. Use bullet points, headers, and code blocks where necessary.

Generate a comprehensive and neatly formatted debug report for the following code:

{file_content}
"""
    elif task_type == "readme":
        existing_readme = repo_info.get('existing_readme', '') if repo_info else ''
        requirements = repo_info.get('requirements', '') if repo_info else ''
        file_structure = '\n'.join(repo_info.get('file_structure', [])) if repo_info else ''
        
        # Create a summary of key files content
        key_files_summary = ""
        for filename, content in (repo_info.get('key_files', {}) if repo_info else {}).items():
            key_files_summary += f"\n### {filename}:\n```python\n{content[:500]}...\n```\n"

        if existing_readme:
            prompt = f"""
You are an expert technical writer and programmer, tasked with updating an existing README.md file for a GitHub repository.
The repository has evolved, and the README needs to be updated to reflect the current state of the project.

Here is the current state of the repository:

1. Existing README.md:
```markdown
{existing_readme}
```

2. Current Repository Structure:
```
{file_structure}
```

3. Key Files:
{key_files_summary}

4. Requirements (if available):
```
{requirements}
```

Task: Update the README.md while maintaining its current structure and style. Make the following improvements:
1. Update any outdated information based on the current repository structure
2. Add any new features or components that are visible in the current codebase
3. Update installation instructions if new dependencies are present
4. Maintain any custom sections or formatting from the original README
5. Ensure all links and references are still valid
6. Add any missing but important sections that a good README should have

Keep what works from the existing README, but enhance it based on the current state of the repository.
The output should be a complete, updated README.md file.
"""
        else:
            prompt = f"""
You are an expert technical writer and programmer, tasked with creating a comprehensive README.md file for a GitHub repository.
Based on the repository structure and contents, create a clear and informative README that will help users understand and use this project.

Repository Analysis:

1. Repository Structure:
```
{file_structure}
```

2. Key Files:
{key_files_summary}

3. Requirements (if available):
```
{requirements}
```

Create a comprehensive README.md that includes:

1. Project Title and Description
   - Analyze the code to determine the project's main purpose
   - Provide a clear, concise description of what the project does
   - Highlight key features and capabilities

2. Installation
   - List all prerequisites
   - Step-by-step installation instructions
   - Environment setup requirements

3. Usage
   - Getting started guide
   - Common use cases and examples
   - Configuration options
   - Command-line arguments (if applicable)

4. Project Structure
   - Explain the main components
   - Describe how different parts work together
   - Document key files and their purposes

5. Dependencies
   - List major dependencies with versions
   - Explain why each major dependency is needed

6. Contributing
   - Guidelines for contributing
   - Development setup
   - Testing instructions

7. License
   - Include MIT license information

8. Contact/Support
   - How to get help
   - Where to report issues

Make the README clear, professional, and well-formatted using Markdown.
Focus on helping users understand and use the project effectively.
"""
    elif task_type == "project_summary":
        prompt = f"""
You are an expert technical writer, skilled at creating project summary documentation in Markdown format.
Create a comprehensive project_summary.md file for a repository containing the following files and folders.
The project_summary.md file will include:

1. Table of Contents
    - List all files and folders in the repository, excluding the 'files' folder.
2. File Details
    - For each file, provide the following information:
        - Full Path
        - Extension
        - Language (if applicable)
        - Size (in bytes)
        - Created Date and Time
        - Modified Date and Time
        - Code Snippet (if applicable)

INSTRUCTION: Use appropriate Markdown formatting to make the project summary visually appealing and easy to read. Here's the file and folder structure to base the project summary on:

{file_content}
"""
    elif task_type == "requirements":
        return None
    api_keys = load_api_keys()
    
    if model in OPENAI_MODELS:
        from openai_utils import call_openai_api
        response = call_openai_api(model, [{"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens, openai_api_key=api_keys.get('openai_api_key'))
        yield response
    elif model in GROQ_MODELS:
        from groq_utils import call_groq_api
        response = call_groq_api(model, prompt, temperature=temperature, max_tokens=max_tokens, groq_api_key=api_keys.get('groq_api_key'))
        yield response
    else:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        headers = {"Content-Type": "application/json"}

        with requests.post(url, json=payload, headers=headers, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        json_response = json.loads(decoded_line)
                        if 'response' in json_response:
                            yield json_response['response']
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line: {decoded_line}")

def run_pylint(file_path):
    result = subprocess.run(['pylint', file_path], capture_output=True, text=True)
    return result.stdout

def run_phpstan(file_path):
    result = subprocess.run(['phpstan', 'analyse', file_path], capture_output=True, text=True)
    return result.stdout

def run_eslint(file_path):
    result = subprocess.run(['eslint', file_path], capture_output=True, text=True)
    return result.stdout

def get_all_code_files(root_dir, exclude_patterns):
    code_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            if file.endswith(('.py', '.php', '.js', '.css', '.html')):
                try:
                    if not any(re.search(bad_pattern, file_path) for bad_pattern in exclude_patterns):
                        code_files.append(file_path)
                except re.error as e:
                    problematic_patterns = [bad_pattern for bad_pattern in exclude_patterns if re.search(bad_pattern, file_path)]
                    st.warning(f"Invalid exclude pattern(s): {problematic_patterns}. Error: {e}. Skipping these pattern(s).")
    return code_files

def get_file_info(file_path):
    try:
        file_stats = os.stat(file_path)
        created_time = datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
        modified_time = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

        extension = os.path.splitext(file_path)[1].lower()
        language = {
            '.py': "python",
            '.txt': "plaintext",
            '.md': "markdown",
            '.json': "json",
            '.sh': "bash",
            '.csv': "csv",
            '.php': "php",
            '.js': "javascript",
            '.css': "css",
            '.html': "html",
            '.xml': "xml",
            '.yaml': "yaml",
            '.yml': "yaml",
            '.sql': "sql",
            '.java': "java",
            '.cpp': "cpp",
            '.c': "c",
            '.h': "c",
            '.rb': "ruby",
            '.go': "go",
            '.rs': "rust",
            '.ts': "typescript",
            '.swift': "swift",
            '.kt': "kotlin",
            '.scala': "scala",
            '.pl': "perl",
            '.lua': "lua",
            '.r': "r",
            '.m': "matlab",
            '.vb': "vbnet",
            '.fs': "fsharp",
            '.hs': "haskell",
            '.ex': "elixir",
            '.erl': "erlang",
            '.clj': "clojure",
            '.groovy': "groovy",
            '.dart': "dart",
            '.f': "fortran",
            '.cob': "cobol",
            '.asm': "assembly",
        }.get(extension, "unknown")

        file_info = {
            "Full Path": file_path,
            "Extension": extension,
            "Language": language,
            "Size": file_stats.st_size,
            "Created": created_time,
            "Modified": modified_time
        }

        # Read file content for all supported languages
        if language != "unknown":
            try:
                with open(file_path, 'r', encoding='utf-8') as code_file:
                    file_info['Code'] = code_file.read()
            except UnicodeDecodeError:
                file_info['Code'] = f"Error reading file: UnicodeDecodeError"

        # Calculate code complexity metrics for Python files
        if language == "python":
            complexity = cc_visit(file_info['Code'])
            maintainability = mi_visit(file_info['Code'], multi=False)
            halstead = h_visit(file_info['Code'])

            file_info['Complexity'] = {
                'Cyclomatic Complexity': [f"{func.name}: {func.complexity} ({cc_rank(func.complexity)})" for func in complexity],
                'Maintainability Index': maintainability,
                'Halstead Metrics': halstead
            }

            # Analyze code style using flake8
            style_guide = flake8.get_style_guide()
            report = style_guide.check_files([file_path])

            style_violations = []
            for error in report.get_statistics(''):
                try:
                    if isinstance(error, str):
                        style_violations.append(error)
                    else:
                        style_violations.append(f"{error.code} ({error.text}): line {error.line}")
                except AttributeError:
                    style_violations.append(f"Unexpected error format: {error}")

            file_info['Style Violations'] = style_violations

        elif language == "php":
            file_info['PHPStan Report'] = run_phpstan(file_path)

        elif language in ["javascript", "css"]:
            file_info['ESLint Report'] = run_eslint(file_path)

        return file_info
    except FileNotFoundError:
        return None

def process_file_with_updates(file_path, task_type, model, temperature, max_tokens, api_key, update_queue, progress_bar, status_text, output_area, repo_info=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

        # Update status
        update_queue.put(("status", f"Processing: {file_path}"))
        
        # Generate documentation with real-time updates
        documentation = ""
        for chunk in generate_documentation_stream(file_content, task_type, model, temperature, max_tokens, repo_info):
            documentation += chunk
            update_queue.put(("output", documentation))

        pylint_report = run_pylint(file_path) if task_type == "debug" and file_path.endswith('.py') else ""
        phpstan_report = run_phpstan(file_path) if task_type == "debug" and file_path.endswith('.php') else ""
        eslint_report = run_eslint(file_path) if task_type == "debug" and (file_path.endswith('.js') or file_path.endswith('.css')) else ""

        return file_path, documentation, pylint_report + phpstan_report + eslint_report, file_content
    except UnicodeDecodeError:
        print(f"Error reading file {file_path}: UnicodeDecodeError")
        return file_path, f"Error reading file: UnicodeDecodeError", "", ""

def analyze_repository_structure(repo_path, code_files):
    """Analyze the repository structure and return key information."""
    repo_info = {
        'main_files': [],
        'key_files': {},
        'file_structure': [],
        'existing_readme': None,
        'existing_readme_path': None,
        'requirements': None,
        'setup_file': None,
        'entry_points': []
    }
    
    # First, explicitly check for README.md in the repository root
    root_readme_path = os.path.join(repo_path, 'README.md')
    if os.path.exists(root_readme_path):
        try:
            with open(root_readme_path, 'r', encoding='utf-8') as f:
                repo_info['existing_readme'] = f.read()
                repo_info['existing_readme_path'] = root_readme_path
        except Exception as e:
            print(f"Error reading root README.md: {str(e)}")
    
    # Look for key files
    for file_path in code_files:
        rel_path = os.path.relpath(file_path, repo_path)
        repo_info['file_structure'].append(rel_path)
        
        filename = os.path.basename(file_path)
        # Only look for README in other locations if we haven't found it in root
        if filename.lower() == 'readme.md' and not repo_info['existing_readme']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    repo_info['existing_readme'] = f.read()
                    repo_info['existing_readme_path'] = file_path
            except Exception as e:
                print(f"Error reading README.md at {file_path}: {str(e)}")
        elif filename in ['setup.py', 'pyproject.toml']:
            repo_info['setup_file'] = file_path
        elif filename.endswith('requirements.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    repo_info['requirements'] = f.read()
            except Exception:
                pass
        elif filename in ['main.py', 'app.py', 'index.py']:
            repo_info['entry_points'].append(file_path)
            repo_info['main_files'].append(file_path)
    
    # Read content of main files
    for file_path in repo_info['main_files']:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                repo_info['key_files'][os.path.basename(file_path)] = f.read()
        except Exception:
            continue
    
    return repo_info

def generate_pdf(results, output_path, task_type):
    pdf = PDF()
    pdf.set_left_margin(10)
    pdf.set_right_margin(10)
    pdf.add_page()

    for file_path, documentation, pylint_report, file_content in results:
        chapter_title = f"File: {file_path}"
        if task_type == "debug":
            chapter_body = f"Pylint Report:\n{pylint_report}\n\nDebug Report:\n{documentation}\n\nCode:\n{file_content}"
        elif task_type == "documentation":
            chapter_body = f"Documentation:\n{documentation}\n\nCode:\n{file_content}"
        else:  # README
            chapter_body = documentation
        pdf.add_chapter(chapter_title, chapter_body)

    # Create 'files' directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path, 'F')

def generate_requirements_file(repo_path, exclude_patterns):
    import importlib.util
    import pkg_resources
    from importlib.metadata import distribution, PackageNotFoundError
    import sys
    from pathlib import Path
    import subprocess
    import re
    from collections import defaultdict
    
    # Categories with their common packages
    PACKAGE_CATEGORIES = {
        'Core Dependencies': {
            'streamlit', 'streamlit-option-menu', 'streamlit-extras', 'streamlit-javascript',
            'ollama', 'openai', 'streamlit-flow'
        },
        'Machine Learning & Data Science': {
            'numpy', 'scipy', 'pandas', 'scikit-learn', 'torch', 'transformers',
            'sentence-transformers', 'spacy', 'tiktoken', 'plotly', 'pydantic'
        },
        'Language Models & AI': {
            'langchain', 'langchain-community', 'groq', 'autogen', 'pyautogen',
            'mistralai'
        },
        'Web & API': {
            'requests', 'httpx', 'beautifulsoup4', 'bs4', 'fake-useragent', 'flask',
            'Flask-Cors', 'duckduckgo-search', 'google-api-python-client',
            'serpapi', 'selenium', 'webdriver-manager', 'playwright', 'gTTS'
        },
        'System & Utilities': {
            'psutil', 'GPUtil', 'rich', 'tqdm', 'humanize', 'schedule', 'cursor',
            'pydub', 'networkx', 'bleach'
        },
        'Document Processing': {
            'PyPDF2', 'fpdf', 'pdfkit', 'reportlab', 'mdutils', 'Markdown'
        },
        'Development & Testing': {
            'pytest', 'pytest-html', 'flake8', 'radon', 'ruff', 'Pygments', 'PyYAML'
        }
    }
    
    def get_installed_version(package_name):
        """Get the installed version of a package."""
        try:
            return pkg_resources.get_distribution(package_name).version
        except pkg_resources.DistributionNotFound:
            return None
    
    def normalize_package_name(name):
        """Normalize package names to handle different formats."""
        return re.sub(r'[-_.]+', '-', name).lower()
    
    def get_package_category(package_name):
        """Determine the category of a package."""
        normalized_name = normalize_package_name(package_name)
        for category, packages in PACKAGE_CATEGORIES.items():
            if any(normalize_package_name(pkg) == normalized_name for pkg in packages):
                return category
        return 'Utilities'  # Default category
    
    def get_imports_from_pipreqs():
        """Use pipreqs to get imports from the codebase."""
        try:
            # Create a temporary requirements file
            temp_req_file = os.path.join(repo_path, 'temp_requirements.txt')
            
            # Run pipreqs
            subprocess.run([
                'pipreqs',
                '--savepath', temp_req_file,
                '--force',
                '--ignore', ','.join(exclude_patterns),
                repo_path
            ], capture_output=True)
            
            # Read the requirements
            with open(temp_req_file, 'r') as f:
                requirements = [line.strip() for line in f if line.strip()]
            
            # Clean up
            os.remove(temp_req_file)
            return requirements
        except Exception as e:
            print(f"Error running pipreqs: {str(e)}")
            return []
    
    def get_installed_packages():
        """Get all installed packages in the current environment."""
        return {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    # Get dependencies using multiple methods
    requirements_dict = defaultdict(set)
    
    # Method 1: Use pipreqs to find imports
    pipreqs_requirements = get_imports_from_pipreqs()
    for req in pipreqs_requirements:
        # Parse package name and version
        match = re.match(r'([^>=<]+).*', req)
        if match:
            package_name = match.group(1).strip()
            version = get_installed_version(package_name)
            if version:
                category = get_package_category(package_name)
                requirements_dict[category].add(f'{package_name}=={version}')
    
    # Method 2: Check installed packages against our known packages
    installed_packages = get_installed_packages()
    for category, packages in PACKAGE_CATEGORIES.items():
        for package in packages:
            normalized_name = normalize_package_name(package)
            if normalized_name in installed_packages:
                requirements_dict[category].add(
                    f'{package}=={installed_packages[normalized_name]}'
                )
    
    # Write requirements.txt with categorized sections
    requirements_path = os.path.join(repo_path, 'requirements.txt')
    with open(requirements_path, 'w', encoding='utf-8') as req_file:
        req_file.write('# Generated by Ollama Workbench Repository Analyzer\n')
        req_file.write('# This file contains verified package dependencies\n\n')
        
        # Write requirements in category order
        for category in PACKAGE_CATEGORIES.keys():
            packages = requirements_dict[category]
            if packages:
                req_file.write(f'# {category}\n')
                for package in sorted(packages):
                    req_file.write(f'{package}\n')
                req_file.write('\n')
        
        # Write any remaining packages in Utilities
        other_packages = requirements_dict['Utilities']
        if other_packages:
            req_file.write('# Utilities\n')
            for package in sorted(other_packages):
                req_file.write(f'{package}\n')
            req_file.write('\n')
    
    return requirements_path

def generate_project_summary(repo_path, exclude_patterns):
    # Create 'files' directory if it doesn't exist (for other outputs)
    files_dir = os.path.join(repo_path, 'files')
    os.makedirs(files_dir, exist_ok=True)

    # Summary file path in the repository root
    summary_path = os.path.join(repo_path, 'project_summary.md')
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        summary_file.write("--- START OF FILE project_summary.md ---\n\n")
        summary_file.write("# Table of Contents\n")

        # Get all files and directories, excluding those matching the exclude patterns
        all_files = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d != 'files' and not any(re.search(pattern, os.path.join(root, d)) for pattern in exclude_patterns)]
            files[:] = [f for f in files if not any(re.search(pattern, os.path.join(root, f)) for pattern in exclude_patterns)]
            for file in files:
                all_files.append(os.path.join(root, file))

        # 1. Write README.md first
        readme_path = os.path.join(repo_path, "README.md")
        if readme_path in all_files:
            summary_file.write(f"- {readme_path}\n")
            all_files.remove(readme_path)

        # 2. Write requirements.txt second
        requirements_path = os.path.join(repo_path, "requirements.txt")
        if requirements_path in all_files:
            summary_file.write(f"- {requirements_path}\n")
            all_files.remove(requirements_path)

        # 3. Write the rest of the files in alphabetical order
        all_files.sort()
        for file_path in all_files:
            summary_file.write(f"- {file_path}\n")

        summary_file.write("\n")

        # Write file details (including README.md and requirements.txt)
        for file_path in [readme_path, requirements_path] + all_files:
            file_info = get_file_info(file_path)
            if file_info:
                write_file_details(summary_file, file_info)

        summary_file.write("--- END OF FILE project_summary.md ---")
    return summary_path

def write_file_details(summary_file, file_info):
    """Writes the file details to the summary file."""
    summary_file.write(f"## File: {file_info['Full Path']}\n\n")
    for key, value in file_info.items():
        if key != "Full Path":
            if key == "Size":
                summary_file.write(f"- {key}: {value} bytes\n")
            elif key == "Language":
                summary_file.write(f"- {key}: {value}\n")
                # Include content for all code and text-based file types
                if value in ("python", "php", "javascript", "css", "plaintext", "markdown", "bash", "csv", "json", "html", "xml", "yaml", "sql"):
                    summary_file.write(f"### Code\n\n")
                    summary_file.write(f"```{value}\n{file_info.get('Code', 'Code not available')}\n```\n\n")
            elif key == "Complexity":  # Write complexity metrics for Python files
                summary_file.write(f"- {key}:\n")
                for metric_name, metric_value in value.items():
                    if isinstance(metric_value, list):  # For Cyclomatic Complexity
                        summary_file.write(f"    - {metric_name}:\n")
                        for item in metric_value:
                            summary_file.write(f"        - {item}\n")
                    elif isinstance(metric_value, dict):  # For Halstead Metrics
                        summary_file.write(f"    - {metric_name}:\n")
                        for sub_metric, sub_value in metric_value.items():
                            summary_file.write(f"        - {sub_metric}: {sub_value}\n")
                    else:  # For Maintainability Index
                        summary_file.write(f"    - {metric_name}: {metric_value}\n")
            elif key == "Style Violations":  # Write style violations for Python files
                summary_file.write(f"- {key}:\n")
                for violation in value:
                    summary_file.write(f"    - {violation}\n")
            elif key == "PHPStan Report":  # Write PHPStan report for PHP files
                summary_file.write(f"- {key}:\n```\n{value}\n```\n")
            elif key == "ESLint Report":  # Write ESLint report for JS and CSS files
                summary_file.write(f"- {key}:\n```\n{value}\n```\n")
            else:
                summary_file.write(f"- {key}: {value}\n")
    summary_file.write("\n")

# Function to load model settings
def load_model_settings():
    if os.path.exists(MODEL_SETTINGS_FILE):
        try:
            with open(MODEL_SETTINGS_FILE, "r") as f:
                settings = json.load(f)
        except json.JSONDecodeError:
            st.warning("Invalid JSON in model settings file. Loading default settings.")
            settings = {}
    else:
        settings = {}

    # Set default values if keys are missing
    default_settings = {
        "model": "mistral:instruct",
        "temperature": 0.7,
        "max_tokens": 4000,
        "api_key": ""
    }

    for key, value in default_settings.items():
        settings.setdefault(key, value)

    return settings

# Function to save model settings
def save_model_settings(settings):
    with open(MODEL_SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)

def main():
    # Load model settings
    model_settings = load_model_settings()
    api_keys = load_api_keys()
    
    st.title("🔍 Repository Analyzer")
    st.write("Enter the path to your repository in the box below. Choose the task type (documentation, debug, readme, requirements, or project_summary) from the dropdown menu. Select the desired Ollama model for the task. Adjust the temperature and max tokens using the sliders. Click 'Analyze Repository' to begin. Once complete, a PDF report will be saved in the repository's 'files' folder. If you chose the 'readme' task type, a README.md file will also be created in the repository's 'files' folder. If you chose the 'project_summary' task type, a project_summary.md file will be created in the repository's 'files' folder.")
    
    col1, col2 = st.columns(2)
    with col1:
        task_type = st.selectbox("Select task type", ["project_summary", "documentation", "debug", "readme", "requirements"])
    with col2:
        repo_path = st.text_input("Enter the path to your repository:")

    # Input for exclude patterns
    exclude_patterns_str = st.text_input("Enter file/folder patterns to exclude (comma-separated, use regex):", 
                                        value=".pythonlibs,.cache,.git,node_modules,__pycache__,cli,.*\.pkl,tmp,.*\.bin,.*\.sqlite3,.*\.db,.DS_Store,files,venv,.*\.ipynb,notebooks,ragtest,LICENSE,checkpoints,.*\.pdf,.*\.png,.*\.jpg,.*\.jpeg,.*\.gif,.*\.csv,.*\.docx,.*\.zip,.*\.eml,.*\.json,.*\.svg,.*\.vue,.*\.ogg,.*\.eot,.*\.ttf,.*\.ico,.*\.otf,.*\.woff,.*\.woff2,chroma_db,.pytest_cache,project_summary.md,agent_prompts,docs,__init__.py")
    exclude_patterns = [pattern.strip() for pattern in exclude_patterns_str.split(",")]

    # Model Settings in a collapsed section in the sidebar
    with st.sidebar:
        with st.expander("🤖 Model Settings", expanded=False):
            # Model selection
            available_models = get_available_models()
            model_settings["model"] = st.selectbox("Select Model", available_models, index=available_models.index(model_settings["model"]) if model_settings["model"] in available_models else 0)

            # API Key input (for OpenAI and Groq models)
            if model_settings["model"] in OPENAI_MODELS or model_settings["model"] in GROQ_MODELS:
                model_settings["api_key"] = st.text_input("API Key", value=model_settings.get("api_key", ""), type="password")

            # Temperature slider
            model_settings["temperature"] = st.slider("Temperature", min_value=0.0, max_value=1.0, value=model_settings["temperature"], step=0.1)

            # Max Tokens slider
            model_settings["max_tokens"] = st.slider("Max Tokens", min_value=1000, max_value=128000, value=model_settings["max_tokens"], step=1000)

            # Save Settings button
            if st.button("💾 Save Settings"):
                save_model_settings(model_settings)
                st.success("Model settings saved!")

    if st.button("🔍 Analyze Repository"):
        if not repo_path or not os.path.isdir(repo_path):
            st.error("Please enter a valid repository path.")
            return

        # Create 'files' directory if it doesn't exist
        files_dir = os.path.join(repo_path, 'files')
        os.makedirs(files_dir, exist_ok=True)

        if task_type == "requirements":
            requirements_path = generate_requirements_file(repo_path, exclude_patterns)
            st.success(f"requirements.txt file has been created at {requirements_path}")
            return
        elif task_type == "project_summary":
            summary_path = generate_project_summary(repo_path, exclude_patterns)
            st.success(f"project_summary.md file has been created at {summary_path}")
            return

        code_files = get_all_code_files(repo_path, exclude_patterns)

        if not code_files:
            st.warning("No code files found in the specified directory.")
            return

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        output_area = st.empty()

        update_queue = queue.Queue()

        def update_ui():
            while True:
                try:
                    update_type, content = update_queue.get(block=False)
                    if update_type == "status":
                        status_text.text(content)
                    elif update_type == "output":
                        output_area.text(content)
                    update_queue.task_done()
                except queue.Empty:
                    break

        with ThreadPoolExecutor(max_workers=1) as executor:
            api_keys = load_api_keys()
            futures = {executor.submit(
                process_file_with_updates, 
                file_path, 
                task_type, 
                model_settings["model"], 
                model_settings["temperature"], 
                model_settings["max_tokens"], 
                model_settings.get("api_key"),  
                update_queue, 
                progress_bar, 
                status_text, 
                output_area
            ): file_path for file_path in code_files}
            
            for i, future in enumerate(as_completed(futures)):               
                file_path, documentation, pylint_report, file_content = future.result()
                results.append((file_path, documentation, pylint_report, file_content))
                progress = (i + 1) / len(code_files)
                progress_bar.progress(progress)
                update_ui()
        progress_bar.empty()
        status_text.empty()

        # Generate and save PDF report in the 'files' folder
        pdf_filename = f"repository_{task_type}_report.pdf"
        pdf_path = os.path.join(files_dir, pdf_filename)
        generate_pdf(results, pdf_path, task_type)
        
        st.success(f"Analysis complete! PDF report saved as {pdf_filename} in the repository's 'files' folder.")

        if task_type == "readme":
            # Analyze repository structure
            repo_info = analyze_repository_structure(repo_path, code_files)
            
            # Report README status
            if repo_info['existing_readme']:
                st.info(f"Found existing README.md at: {os.path.relpath(repo_info['existing_readme_path'], repo_path)}")
                st.write("Will update the existing README while preserving its structure.")
            else:
                st.warning("No existing README.md found. Will create a new one based on repository analysis.")
            
            # Process the entire repository as one
            future = executor.submit(
                process_file_with_updates,
                "README.md",  # Just a placeholder filename
                task_type,
                model_settings["model"],
                model_settings["temperature"],
                model_settings["max_tokens"],
                model_settings.get("api_key"),
                update_queue,
                progress_bar,
                status_text,
                output_area,
                repo_info  # Pass repository information to the generation function
            )
            
            _, readme_content, _, _ = future.result()
            
            # Save the README
            if repo_info['existing_readme_path']:
                # Create backup of existing README
                backup_path = repo_info['existing_readme_path'] + '.backup'
                try:
                    with open(repo_info['existing_readme_path'], 'r', encoding='utf-8') as src:
                        with open(backup_path, 'w', encoding='utf-8') as dst:
                            dst.write(src.read())
                    st.info(f"Created backup of existing README at: {os.path.relpath(backup_path, repo_path)}")
                except Exception as e:
                    st.warning(f"Failed to create backup of existing README: {str(e)}")
                
                # Update existing README
                readme_path = repo_info['existing_readme_path']
            else:
                # Create new README in repository root
                readme_path = os.path.join(repo_path, 'README.md')
            
            with open(readme_path, 'w', encoding='utf-8') as readme_file:
                readme_file.write(readme_content)
            
            st.success(f"README.md has been {'updated' if repo_info['existing_readme'] else 'created'} at: {os.path.relpath(readme_path, repo_path)}")
            st.markdown("## Generated README.md")
            st.markdown(readme_content)
            return

if __name__ == "__main__":
    main()