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
    url = "http://localhost:11434/api/tags"
    try:
        response = requests.get(url)
        response.raise_for_status()
        models = response.json()['models']
        return [model['name'] for model in models]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching available models: {e}")
        return []

def generate_documentation_stream(file_content, task_type, model, temperature, max_tokens):
    if task_type == "documentation":
        prompt = f"""
You are an expert in Python programming and technical writing. Your task is to generate comprehensive documentation and insightful commentary for the provided Python code. 

Follow these steps:

1. **Overview**: Provide a high-level summary of what the code does, its purpose, its main components, and how it connects to the other scripts in this repository.
2. **Code Walkthrough**: Go through the code section by section, explaining the functionality of each part. Highlight key functions, classes, and methods.
3. **Best Practices**: Identify any non-Pythonic practices or areas where the code could be improved. Suggest best practices and optimizations.
4. **Examples**: Where applicable, include example usages or scenarios that demonstrate how the code should be used.
5. **Formatting**: Ensure that the documentation is well-organized, clearly formatted, and easy to read. Use bullet points, headers, and code blocks where necessary.

Generate a detailed and neatly formatted documentation for the following code:

{file_content}
"""
    elif task_type == "debug":
        prompt = f"""
You are a highly experienced Python code debugger with a deep understanding of Pythonic best practices and coding standards. Your task is to thoroughly analyze the provided code, identify any issues, and offer recommendations for improvements. Follow these steps:

1. **Identify Issues**: Go through the code line by line and identify syntax errors, logical errors, performance issues, and any non-Pythonic practices.
2. **Provide Recommendations**: For each issue identified, provide a detailed explanation of why it is an issue and how it can be fixed or improved.
3. **Code Examples**: Whenever possible, include code snippets to illustrate the recommended changes or improvements.
4. **Formatting**: Ensure that your debug report is well-organized, clearly formatted, and easy to read. Use bullet points, headers, and code blocks where necessary.

Generate a comprehensive and neatly formatted debug report for the following code:

{file_content}
"""
    elif task_type == "readme":
        prompt = f"""
You are an expert writer and programmer, skilled at writing README.md content for GitHub. 
Create a comprehensive README.md file for a GitHub repository containing the following code. 
Pay close attention to how the UI is executed in the code and write the documentation accordingly. 
As a prime example, if the app uses Streamlit, assume the user will interface with it via a browser and give directions on how to use based on How Streamlit works, not how direct access via Python works to interface with the app.

The README will include:

1. Project Title and Description
    - The repository name should be on the first line of the main.py file. If not, then create a funny name that will make people laugh.
2. Installation Instructions
    - Give standard GitHub command line instructions.
3. Usage Guide
4. Features
5. Dependencies
    - List in code block
6. Contributing Guidelines
7. License Information
    - Assume MIT license
8. Contact/Support Information
    - Create a generic template using me@mydomain.com, etc.

INSTRUCTION: Use appropriate Markdown formatting to make the README visually appealing and easy to read. Here's the code to base the README on:

{file_content}
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

def get_all_code_files(root_dir, exclude_patterns):
    code_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            if file.endswith('.py'):
                try:
                    if not any(re.search(pattern, file_path) for pattern in exclude_patterns):
                        code_files.append(file_path)
                except re.error as e:
                    st.warning(f"Invalid exclude pattern: '{pattern}'. Error: {e}. Skipping this pattern.")
    return code_files

def get_file_info(file_path):
    try:
        file_stats = os.stat(file_path)
        created_time = datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
        modified_time = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

        extension = os.path.splitext(file_path)[1].lower()  # Get extension and make it lowercase
        language = {
            '.py': "python",
            '.txt': "plaintext",
            '.md': "markdown",
            '.json': "json",
            '.sh': "bash",
            '.csv': "csv"
        }.get(extension, "unknown")  # Determine language based on extension

        return {
            "Full Path": file_path,
            "Extension": extension,
            "Language": language,
            "Size": file_stats.st_size,
            "Created": created_time,
            "Modified": modified_time
        }
    except FileNotFoundError:
        return None

def process_file_with_updates(file_path, task_type, model, temperature, max_tokens, update_queue, progress_bar, status_text, output_area):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

        # Update status
        update_queue.put(("status", f"Processing: {file_path}"))
        
        # Generate documentation with real-time updates
        documentation = ""
        for chunk in generate_documentation_stream(file_content, task_type, model, temperature, max_tokens):
            documentation += chunk
            update_queue.put(("output", documentation))

        pylint_report = run_pylint(file_path) if task_type == "debug" else ""
        return file_path, documentation, pylint_report, file_content
    except UnicodeDecodeError:
        print(f"Error reading file {file_path}: UnicodeDecodeError")
        return file_path, f"Error reading file: UnicodeDecodeError", "", ""

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
    requirements = set()
    code_files = get_all_code_files(repo_path, exclude_patterns)
    for code_file in code_files:
        with open(code_file, 'r') as file:
            for line in file:
                if line.startswith('import ') or line.startswith('from '):
                    parts = line.split()
                    if parts[0] == 'import':
                        module = parts[1].split('.')[0]
                        if '_' not in module:
                            requirements.add(module)
                    elif parts[0] == 'from':
                        module = parts[1].split('.')[0]
                        if '_' not in module:
                            requirements.add(module)

    # Create 'files' directory if it doesn't exist
    files_dir = os.path.join(repo_path, 'files')
    os.makedirs(files_dir, exist_ok=True)
    
    requirements_path = os.path.join(files_dir, 'requirements.txt')
    with open(requirements_path, 'w') as req_file:
        for requirement in sorted(requirements):
            req_file.write(requirement + '\n')
    return requirements_path

def generate_project_summary(repo_path, exclude_patterns):
    # Create 'files' directory if it doesn't exist
    files_dir = os.path.join(repo_path, 'files')
    os.makedirs(files_dir, exist_ok=True)

    summary_path = os.path.join(files_dir, 'project_summary.md')
    with open(summary_path, 'w') as summary_file:
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
                # Include content for specific file types
                if value in ("python", "plaintext", "markdown", "bash", "csv", "json"): 
                    summary_file.write(f"### Code\n\n")
                    try:
                        with open(file_info['Full Path'], 'r', encoding='utf-8') as code_file:
                            code_content = code_file.read()
                            summary_file.write(f"```{value}\n{code_content}\n```\n\n")
                    except UnicodeDecodeError:
                        summary_file.write(f"- Unable to display code snippet. File may be binary or encoded differently.\n")
            else:
                summary_file.write(f"- {key}: {value}\n")
    summary_file.write("\n")

def main():
    st.title("✔️ Repository Analyzer")
    st.write("Enter the path to your repository in the box below. Choose the task type (documentation, debug, readme, requirements, or project_summary) from the dropdown menu. Select the desired Ollama model for the task. Adjust the temperature and max tokens using the sliders. Click 'Analyze Repository' to begin. Once complete, a PDF report will be saved in the repository's 'files' folder. If you chose the 'readme' task type, a README.md file will also be created in the repository's 'files' folder. If you chose the 'project_summary' task type, a project_summary.md file will be created in the repository's 'files' folder.")
    repo_path = st.text_input("Enter the path to your repository:")

    # Input for exclude patterns
    exclude_patterns_str = st.text_input("Enter file/folder patterns to exclude (comma-separated, use regex):", 
                                        value=".git,__pycache__,cli,.*\.pkl,tmp,.*\.bin,.*\.sqlite3,.*\.db,.DS_Store,.*\.log,files,venv,.*\.ipynb,notebooks,checkpoints,.*\.pdf,.*\.png,.*\.jpg,.*\.jpeg,.*\.gif,.*\.eml")
    exclude_patterns = [pattern.strip() for pattern in exclude_patterns_str.split(",")]

    # Four-column layout for task type, model, temperature, and max tokens
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        task_type = st.selectbox("Select task type", ["documentation", "debug", "readme", "requirements", "project_summary"])
    with col2:
        available_models = get_available_models()
        model = st.selectbox(f"Select model for {task_type} task", available_models)
    with col3:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    with col4:
        max_tokens = st.slider("Max Tokens", min_value=100, max_value=32000, value=4000, step=100)

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
            st.warning("No Python files found in the specified directory.")
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
            futures = {executor.submit(process_file_with_updates, file_path, task_type, model, temperature, max_tokens, update_queue, progress_bar, status_text, output_area): file_path for file_path in code_files}
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
            readme_content = results[0][1] if results else "No content generated"
            readme_path = os.path.join(files_dir, "README.md")
            with open(readme_path, "w") as readme_file:
                readme_file.write(readme_content)
            st.success(f"README.md file has been created in the repository's 'files' folder.")
            st.markdown("## Generated README.md")
            st.markdown(readme_content)

if __name__ == "__main__":
    main()