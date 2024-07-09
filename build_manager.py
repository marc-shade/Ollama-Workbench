# build_manager.py
import json
import os
import queue
import csv  # For CSV handling
import threading  # Import the threading module
import logging  # Import the logging module
from agents import Agent
from ollama_utils import call_ollama_endpoint
from projects import Task, save_tasks, load_tasks  # Import necessary functions
import ollama  # Import the ollama module
from concurrent.futures import ThreadPoolExecutor
import re
from functools import reduce
import streamlit as st  # Import Streamlit
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(filename='build_manager.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class BuildManager(Agent):
    def __init__(self, message_queue: queue.Queue, update_task_callback, agents_dir: str = "agents", workflows_dir: str = "workflows"):
        super().__init__(name="Build Manager", capabilities=["workflow_management"], prompts={}, model="mistral:7b-instruct-v0.2-q8_0")
        self.message_queue = message_queue
        self.agents_dir = agents_dir
        self.workflows_dir = workflows_dir
        self.update_task_callback = update_task_callback
        self.cancel_flag = False  # Flag for cancellation

        # Create the agents and workflows directories if they don't exist
        os.makedirs(self.agents_dir, exist_ok=True)
        os.makedirs(self.workflows_dir, exist_ok=True)

        try:
            self.load_agents()
            self.load_workflows()
        except Exception as e:
            print(f"Critical error during initialization: {e}")
            logging.error(f"Critical error during initialization: {e}")
            self.cancel_workflow()
            self.update_task_callback(0, "Failed", f"Critical Error: {e}")  # Assuming index 0 is the first task
            return  # Terminate initialization

    def load_agents(self):
        self.agents = {}
        for filename in os.listdir(self.agents_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.agents_dir, filename)
                try:
                    agent = Agent.from_json(filepath)
                    self.agents[agent.name] = agent
                except Exception as e:
                    print(f"Error loading agent from {filepath}: {e}")
                    logging.error(f"Error loading agent from {filepath}: {e}")
                    raise  # Re-raise the exception to trigger workflow termination

    def load_workflows(self):
        self.workflows = {}
        for filename in os.listdir(self.workflows_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.workflows_dir, filename)
                with open(filepath, 'r') as f:
                    try:
                        workflow_data = json.load(f)
                        self.workflows[workflow_data["name"]] = workflow_data
                    except Exception as e:
                        print(f"Error loading workflow from {filepath}: {e}")
                        logging.error(f"Error loading workflow from {filepath}: {e}")
                        raise  # Re-raise the exception to trigger workflow termination

    def process_request(self, user_request: str):
        try:
            # 1. Interpret User Request (Using an LLM for classification)
            classification_prompt = f"""
            Classify the following user request into one of these categories:
            - data_analysis
            - code_generation
            - content_creation
            - other

            User Request: {user_request}

            Classification: 
            """
            # Use ollama.generate for non-streaming response
            response = ollama.generate(model="mistral:7b-instruct-v0.2-q8_0", prompt=classification_prompt)
            task_category = response['response'].strip().lower()

            if task_category == "data_analysis":
                self.execute_workflow("Data Analysis Workflow", user_request)
            elif task_category == "code_generation":
                self.execute_workflow("Code Generation Workflow", user_request)
            # ... handle other categories ...
            else:
                print(f"Error: Unable to classify user request: {user_request}")
                logging.error(f"Error: Unable to classify user request: {user_request}")
                self.cancel_workflow()
                self.update_task_callback(0, "Failed", f"Error: Unable to classify user request")  # Assuming index 0 is the first task

        except Exception as e:
            print(f"Critical error in process_request: {e}")
            logging.error(f"Critical error in process_request: {e}")
            self.cancel_workflow()
            self.update_task_callback(0, "Failed", f"Critical Error: {e}")  # Assuming index 0 is the first task

    def execute_workflow(self, workflow_name: str, user_request: str, project_name="My Project"):
        workflow = self.workflows.get(workflow_name)
        if not workflow:
            print(f"Error: Workflow '{workflow_name}' not found.")
            return

        # Create initial task data from user request
        task_data = {"user_request": user_request}

        # Create agents and assign them to tasks
        for step_index, step in enumerate(workflow["steps"]):
            agent_name = step["agent"]

            # Check if agent already exists
            agent = self.agents.get(agent_name)
            if not agent:
                # Create a new agent
                agent = Agent(name=agent_name, capabilities=[], prompts={}, model="mistral:7b-instruct-v0.2-q8_0")
                self.agents[agent_name] = agent

            # Handle user input before creating the Task object
            user_input_config = step.get("user_input")
            if user_input_config:
                input_type = user_input_config["type"]
                prompt = user_input_config["prompt"]

                if input_type == "file_path":
                    file_path = st.text_input(prompt, key=f"user_input_{step_index}")
                    if file_path:
                        task_data["file_path"] = file_path  # Add file_path to task_data
                    else:
                        st.warning("Please provide a file path.")
                        return  # Stop workflow execution if no file path is provided

                elif input_type == "options":
                    options = user_input_config.get("options", [])
                    selected_option = st.selectbox(prompt, options, key=f"user_input_{step_index}")
                    if selected_option:
                        task_data["selected_option"] = selected_option  # Add selected_option to task_data
                    else:
                        st.warning("Please select an option.")
                        return  # Stop workflow execution if no option is selected

                elif input_type == "confirmation":
                    if not st.button(prompt, key=f"user_input_{step_index}"):
                        st.warning("Task skipped due to unconfirmed user input.")
                        continue  # Skip to the next step if not confirmed

            # Prepare task description and inputs (now with user input in task_data)
            task_description = step["task_description"].format(**task_data)
            task_inputs = {input_key: task_data.get(input_key) for input_key in step["inputs"]}

            # Create Task object
            task = Task(name=agent.name, description=task_description, **task_inputs)

            # Create a threading event for this task
            task_event = threading.Event()
            task_event.set()  # Start in running state

            # Send task to agent
            task_message = {
                "task": task,
                "agent": agent,
                "step_index": step_index,
                "project_name": project_name,
                "task_event": task_event
            }
            self.message_queue.put(task_message)
            print(f"Task {task.name} added to message queue.")

        # Create initial task entries in session state
        st.session_state.bm_tasks = [
            {"task_id": task.task_id, "name": step["agent"], "status": "Pending", "result": None, "project_name": workflow_name}
            for step, task in zip(workflow["steps"], self.message_queue.queue)
        ]

        # Get workflow start time
        start_time = st.session_state.get("workflow_start_time")
        if not start_time:
            start_time = datetime.now()
            st.session_state.workflow_start_time = start_time

    def run(self):
        # Initialize session state for tasks
        if "bm_tasks" not in st.session_state:
            st.session_state.bm_tasks = []

        # Process all tasks in the queue
        while not self.message_queue.empty() and not self.cancel_flag:
            task_message = self.message_queue.get()
            print(f"Task {task_message['task'].name} retrieved from message queue.")
            self.execute_task(task_message)

        if not self.cancel_flag:
            print("Workflow completed!")
            self.reset_workflow()

    def execute_task(self, task_message):
        """Executes a task in the main thread."""
        agent = task_message["agent"]
        task = task_message["task"]
        step_index = task_message["step_index"]
        project_name = task_message["project_name"]
        task_event = task_message["task_event"]

        # Task cancellation flag
        task_canceled = False

        try:
            # Update task status to "In Progress"
            self.update_task_callback(step_index, "In Progress", None)  # Use the callback to update UI

            # Generate prompt using the task description
            prompt_template = agent.prompts.get("generate_code")
            if prompt_template:
                prompt = prompt_template.format(task_description=task.description)
            else:
                prompt = task.description

            # Execute the task using the agent's model
            response, _, _, _ = call_ollama_endpoint(agent.model, prompt=prompt)

            # Process agent response and update Task object
            if agent.name == "Data Extractor":
                try:
                    # Assuming the response is a CSV string
                    reader = csv.reader(response.splitlines())
                    extracted_data = list(reader)
                    task.result = extracted_data
                    task.completed = True
                except Exception as e:
                    print(f"Error extracting data: {e}")
                    task.result = f"Error: {e}"
                    task.completed = False

            elif agent.name == "Data Analyst":
                try:
                    analysis_results = json.loads(response)
                    task.result = analysis_results
                    task.completed = True
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON response from Data Analyst: {response}")
                    task.result = f"Error: Invalid JSON response"
                    task.completed = False

            elif agent.name == "Report Writer":
                task.result = response
                task.completed = True

            # Wait for task event (for pause/resume)
            task_event.wait()

            # Check if task was canceled
            if task_canceled:
                print(f"Task {task.name} canceled.")
                self.update_task_callback(step_index, "Canceled", None)
                return

            # Update task status to "Completed"
            self.update_task_callback(step_index, "Completed", task.result)  # Use the callback to update UI

        except Exception as e:
            print(f"Error executing task: {e}")
            logging.error(f"Error executing task: {task.name} - {e}", exc_info=True)
            self.update_task_callback(step_index, "Failed", f"Error: {e}")  # Use the callback to update UI

        # Handle pause/resume/cancel commands
        command = st.session_state.get("command")
        task_name = st.session_state.get("task_name")
        if task_name and task.name == task_name:
            if command == "pauseTask":
                task_event.clear()  # Pause the task
                self.update_task_callback(step_index, "Paused", None)
            elif command == "resumeTask":
                task_event.set()  # Resume the task
                self.update_task_callback(step_index, "In Progress", None)
            elif command == "cancelTask":
                task_canceled = True  # Set the cancellation flag
                task_event.set()  # Release the task if it's waiting

    def cancel_workflow(self):
        self.cancel_flag = True
        # Clear the message queue
        while not self.message_queue.empty():
            self.message_queue.get()

        # Reset parallel task counts
        self.parallel_task_counts = {}

    def reset_workflow(self):
        """Resets the workflow state."""
        self.cancel_flag = False
        # Clear the message queue
        while not self.message_queue.empty():
            self.message_queue.get()

        # Reset parallel task counts
        self.parallel_task_counts = {}

        # Reset session state
        st.session_state.bm_tasks = []
        st.session_state.workflow_start_time = None

    def cancel_task(self, task_id: str):
        """
        Cancels a task and all its dependent tasks.

        Args:
            task_id: The ID of the task to cancel.
        """
        # Find the task index in the session state
        task_index = next((i for i, t in enumerate(st.session_state.bm_tasks) if t["task_id"] == task_id), None)
        if task_index is None:
            print(f"Error: Task with ID {task_id} not found.")
            return

        # Cancel the task
        self.cancel_task_by_index(task_index)

        # Recursively cancel dependent tasks
        workflow_name = st.session_state.bm_tasks[0].get("project_name", "Data Analysis Workflow")  # Get workflow name from task data
        workflow = self.workflows.get(workflow_name)
        if workflow:
            for i, step in enumerate(workflow["steps"]):
                if step.get("depends_on") == task_index:
                    self.cancel_task_by_index(i)

    def cancel_task_by_index(self, task_index: int):
        """
        Cancels a task by its index in the session state.

        Args:
            task_index: The index of the task to cancel.
        """
        task = st.session_state.bm_tasks[task_index]
        task["status"] = "Canceled"

        # Get the task_event from the task_message queue
        for message in self.message_queue.queue:
            if message["task"].task_id == task["task_id"]:
                task_event = message.get("task_event")
                if task_event:
                    task_event.set()  # Signal the task to stop
                break

        # Call the agent's cancel_task method
        agent = self.agents.get(task["name"])
        if agent:
            agent.cancel_task(task)

    def pause_task(self, task_id: str):
        """Pauses a specific task."""
        task_index = next((i for i, t in enumerate(st.session_state.bm_tasks) if t["task_id"] == task_id), None)
        if task_index is not None:
            for message in self.message_queue.queue:
                if message["task"].task_id == task_id:
                    task_event = message.get("task_event")
                    if task_event:
                        task_event.clear()  # Pause the task
                        self.update_task_callback(task_index, "Paused", None)
                        print(f"Task {task_id} paused.")
                    break

    def resume_task(self, task_id: str):
        """Resumes a specific task."""
        task_index = next((i for i, t in enumerate(st.session_state.bm_tasks) if t["task_id"] == task_id), None)
        if task_index is not None:
            for message in self.message_queue.queue:
                if message["task"].task_id == task_id:
                    task_event = message.get("task_event")
                    if task_event:
                        task_event.set()  # Resume the task
                        self.update_task_callback(task_index, "In Progress", None)
                        print(f"Task {task_id} resumed.")
                    break

    def generate_workflow(self, user_request: str):
        """
        Generates a workflow template based on a user request.

        Args:
            user_request: The user's request for the workflow.

        Returns:
            The generated workflow template as a JSON string, or None if generation or validation failed.
        """
        # 1. Generate Workflow Template using LLM
        generation_prompt = f"""
        Generate a valid JSON workflow template to fulfill the following user request:

        User Request: {user_request}

        Available Agents:
        - Data Extractor: Extracts data from CSV files.
        - Data Analyst: Analyzes data and calculates statistics.
        - Report Writer: Generates reports from analysis results.
        - Data Visualizer: Creates visualizations from data.

        Workflow Template Structure:
        {{
          "name": "Workflow Name",
          "description": "Workflow Description",
          "steps": [
            {{
              "agent": "Agent Name",
              "task_description": "Task Description",
              "inputs": ["Input Variables"],
              "outputs": ["Output Variables"],
              "depends_on": null or Step Index,
              "parallel_group": null or Group Name,
              "user_input": {{
                "type": "Input Type",
                "prompt": "User Input Prompt",
                "options": ["Option 1", "Option 2"]  // If applicable
              }},
              "condition": "Condition Expression"  // Evaluate to True or False
            }}
          ]
        }}

        Example:
        {{
          "name": "Data Analysis Workflow",
          "description": "A simple workflow to extract data from a CSV file, analyze it, and generate a report.",
          "steps": [
            {{
              "agent": "Data Extractor",
              "task_description": "Extract data from the CSV file.",
              "inputs": ["csv_file_path"],
              "outputs": ["extracted_data"],
              "depends_on": null,
              "parallel_group": null,
              "user_input": {{
                "type": "file_path",
                "prompt": "Please provide the path to the CSV file:"
              }},
              "condition": null
            }},
            {{
              "agent": "Data Analyst",
              "task_description": "Analyze the extracted data.",
              "inputs": ["extracted_data"],
              "outputs": ["analysis_results"],
              "depends_on": 0,
              "parallel_group": null,
              "user_input": null,
              "condition": null
            }},
            {{
              "agent": "Report Writer",
              "task_description": "Generate a report from the analysis results.",
              "inputs": ["analysis_results"],
              "outputs": ["report"],
              "depends_on": 1,
              "parallel_group": null,
              "user_input": null,
              "condition": null
            }}
          ]
        }}
        """
        response = ollama.generate(model="mistral:7b-instruct-v0.2-q8_0", prompt=generation_prompt)
        generated_workflow = response['response']

        # 2. Preprocess generated workflow (remove extra characters and whitespace)
        generated_workflow = generated_workflow.strip()  # Remove leading/trailing whitespace
        generated_workflow = generated_workflow.replace("'", '"')  # Replace single quotes with double quotes
        generated_workflow = re.sub(r"[^\w\s{}\[\]\":,.-]+", "", generated_workflow)  # Remove invalid characters

        # 3. Ensure Enclosing Braces
        if not generated_workflow.startswith("{"):
            generated_workflow = "{" + generated_workflow
        if not generated_workflow.endswith("}"):
            generated_workflow = generated_workflow + "}"

        # 4. Ensure property names are enclosed in double quotes
        generated_workflow = re.sub(r"(\s*)'(\w+)'(\s*):", r'\1"\2"\3:', generated_workflow)

        # 5. Add Missing Commas (More Robust)
        generated_workflow = re.sub(r"([}\]])(\s*)(?=[{\[\"a-zA-Z])", r"\1,\2", generated_workflow)

        # 6. Load JSON and Post-process to ensure 'condition' is an array and 'depends_on' is valid
        try:
            workflow_data = json.loads(generated_workflow)  # Load JSON here
            num_steps = len(workflow_data["steps"])
            for step_index, step in enumerate(workflow_data["steps"]):  # Include step_index in the loop
                if "condition" in step and not isinstance(step["condition"], list):
                    if step["condition"] is None:
                        step["condition"] = []
                    else:
                        # Create a condition object with "expression" and a default "next_step"
                        step["condition"] = [{"expression": step["condition"], "next_step": step_index + 1}]

                # Validate depends_on field
                depends_on = step.get("depends_on")
                if depends_on is not None and (not isinstance(depends_on, int) or depends_on < 0 or depends_on >= num_steps):
                    step["depends_on"] = None  # Reset depends_on to None if invalid

                # Validate condition field (ensure each condition has a valid next_step)
                if "condition" in step and isinstance(step["condition"], list) and len(step["condition"]) > 0:  # Check if condition is a non-empty list
                    for condition in step["condition"]:
                        if "next_step" not in condition or (not isinstance(condition["next_step"], int) or condition["next_step"] < 0 or condition["next_step"] >= num_steps):
                            condition["next_step"] = step_index + 1  # Set a default next_step if invalid

            generated_workflow = json.dumps(workflow_data)  # Update generated_workflow after post-processing
        except json.JSONDecodeError as e:
            print(f"Error loading or post-processing JSON: {e}")
            return None

        # 7. Validate Workflow Template
        is_valid, error_message = self.validate_workflow_template(generated_workflow)
        if is_valid:
            print("Generated workflow template is valid.")
            return generated_workflow
        else:
            print(f"Generated workflow template is invalid: {error_message}")
            return None

    def validate_workflow_template(self, workflow_json: str) -> (bool, str):
        """
        Validates a workflow template in JSON format.

        Args:
            workflow_json: The workflow template as a JSON string.

        Returns:
            A tuple containing:
                - True if the template is valid, False otherwise.
                - An error message if the template is invalid, an empty string otherwise.
        """
        try:
            workflow_data = json.loads(workflow_json)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"

        # Check for required fields
        required_fields = ["name", "description", "steps"]
        for field in required_fields:
            if field not in workflow_data:
                return False, f"Missing required field: '{field}'"

        # Check steps array
        steps = workflow_data["steps"]
        if not isinstance(steps, list):
            return False, "The 'steps' field must be an array."

        # Check individual steps
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                return False, f"Step {i+1} is not a valid object."

            # Check required step fields
            required_step_fields = ["agent", "task_description", "inputs", "outputs"]
            for field in required_step_fields:
                if field not in step:
                    return False, f"Step {i+1} is missing required field: '{field}'"

            # Check data types of step fields
            if not isinstance(step["agent"], str):
                return False, f"Step {i+1}: 'agent' must be a string."
            if not isinstance(step["task_description"], str):
                return False, f"Step {i+1}: 'task_description' must be a string."
            if not isinstance(step["inputs"], list):
                return False, f"Step {i+1}: 'inputs' must be an array."
            if not isinstance(step["outputs"], list):
                return False, f"Step {i+1}: 'outputs' must be an array."

            # Check depends_on field
            depends_on = step.get("depends_on")
            if depends_on is not None and (not isinstance(depends_on, int) or depends_on < 0 or depends_on >= len(steps)):
                return False, f"Step {i+1}: 'depends_on' must be a valid step index."

            # Check parallel_group field (optional)
            parallel_group = step.get("parallel_group")
            if parallel_group is not None and not isinstance(parallel_group, str):
                return False, f"Step {i+1}: 'parallel_group' must be a string."

            # Check user_input field (optional)
            user_input = step.get("user_input")
            if user_input is not None:
                if not isinstance(user_input, dict):
                    return False, f"Step {i+1}: 'user_input' must be an object."
                if "type" not in user_input or not isinstance(user_input["type"], str):
                    return False, f"Step {i+1}: 'user_input' must have a 'type' field (string)."

            # Check condition field (ensure it's an array)
            conditions = step.get("condition")
            if conditions is not None and not isinstance(conditions, list):
                return False, f"Step {i+1}: 'condition' must be an array."

            # Check individual conditions
            if isinstance(conditions, list):
                for j, cond in enumerate(conditions):
                    if not isinstance(cond, dict):
                        return False, f"Step {i+1}, Condition {j+1}: Each condition must be an object."
                    if "expression" not in cond or not isinstance(cond["expression"], str):
                        return False, f"Step {i+1}, Condition {j+1}: Each condition must have an 'expression' field (string)."
                    if "next_step" not in cond or (not isinstance(cond["next_step"], int) or cond["next_step"] < 0 or cond["next_step"] >= len(steps)):
                        return False, f"Step {i+1}, Condition {j+1}: Each condition must have a 'next_step' field (valid step index)."

        return True, ""