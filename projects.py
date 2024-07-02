import streamlit as st
import pandas as pd
import json
import os
import uuid  # For generating unique task IDs
from datetime import datetime
import base64
from ollama_utils import get_available_models, ollama, call_ollama_endpoint
from prompts import get_agent_prompt, get_metacognitive_prompt, get_voice_prompt
import re  # Import the re module

class Task:
    def __init__(self, name, description, deadline=None, priority="Medium", completed=False, agent="None", result=None):
        self.task_id = str(uuid.uuid4())  # Generate a unique ID for each task
        self.name = name
        self.description = description
        self.deadline = deadline
        self.priority = priority
        self.completed = completed
        self.agent = agent
        self.result = result

# Ensure the 'projects' directory exists
if not os.path.exists('projects'):
    os.makedirs('projects')

# Custom JSON encoder to handle Timestamp objects and NaT values
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            if pd.isna(obj):
                return None
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, Task):  # Add support for encoding Task objects
            return obj.__dict__
        return super().default(obj)

# Function to load projects
def load_projects():
    try:
        with open('projects/projects.json', 'r') as f:
            content = f.read()
            if not content:
                return []
            return json.loads(content)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        st.error(f"Error loading projects: {str(e)}. Starting with an empty project list.")
        return []

# Function to save projects
def save_projects(projects):
    with open('projects/projects.json', 'w') as f:
        json.dump(projects, f)

# Function to load tasks for a specific project
def load_tasks(project_name):
    try:
        with open(f'projects/{project_name}_tasks.json', 'r') as f:
            content = f.read()
            if not content:
                return []
            task_data = json.loads(content)
        tasks = []
        for data in task_data:
            task = Task(
                name=data.get("name"),
                description=data.get("description"),
                deadline=pd.to_datetime(data.get("deadline"), errors='coerce'),
                priority=data.get("priority", "Medium"),
                completed=data.get("completed", False),
                agent=data.get("agent"),
                result=data.get("result")
            )
            tasks.append(task)
        return tasks
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        st.error(f"Error loading tasks for {project_name}: {str(e)}. Starting with an empty task list.")
        return []

# Function to save tasks for a specific project
def save_tasks(project_name, tasks):
    # Remove tasks with NaT deadlines
    tasks = [task for task in tasks if pd.notna(task.deadline)]
    with open(f'projects/{project_name}_tasks.json', 'w') as f:
        json.dump(tasks, f, cls=DateTimeEncoder)

# Function to get corpus options
def get_corpus_options():
    corpus_folder = "corpus"
    if not os.path.exists(corpus_folder):
        os.makedirs(corpus_folder)
    return ["None"] + [f for f in os.listdir(corpus_folder) if os.path.isdir(os.path.join(corpus_folder, f))]

# Function to get corpus context from database
def get_corpus_context_from_db(corpus_folder, corpus, user_input):
    # Placeholder for the actual implementation of retrieving context from the corpus database
    return "Sample context from the corpus database."

# Function to define AI agents
def ai_agent(user_input, model, agent_type, metacognitive_type, voice_type, corpus, temperature, max_tokens, previous_responses=[]):
    # Combine agent type, metacognitive type, and voice type prompts
    combined_prompt = ""
    if agent_type != "None":
        combined_prompt += get_agent_prompt()[agent_type] + "\n\n"
    if metacognitive_type != "None":
        combined_prompt += get_metacognitive_prompt()[metacognitive_type] + "\n\n"
    if voice_type != "None":
        combined_prompt += get_voice_prompt()[voice_type] + "\n\n"

    # Include corpus context if selected
    if corpus != "None":
        corpus_context = get_corpus_context_from_db("corpus", corpus, user_input)
        combined_prompt += f"Context: {corpus_context}\n\n"

    # Include previous responses for context
    for i, response in enumerate(previous_responses):
        combined_prompt += f"Response {i+1}: {response}\n\n"

    final_prompt = f"{combined_prompt}User: {user_input}"

    # Generate response using call_ollama_endpoint
    response, _, _, _ = call_ollama_endpoint(model, prompt=final_prompt, temperature=temperature, max_tokens=max_tokens)

    return response

# Function to define agent parameters
def define_agent_block(name, agent_data=None):
    if agent_data is None:
        agent_data = {}
    model = st.selectbox(f"{name} Model", get_available_models(), key=f"{name}_model", index=get_available_models().index(agent_data.get('model')) if agent_data.get('model') in get_available_models() else 0)
    # Handle the case where 'agent_type' is not in agent_data
    agent_type_options = ["None"] + list(get_agent_prompt().keys())
    agent_type = st.selectbox(
        f"{name} Agent Type",
        agent_type_options,
        key=f"{name}_agent_type",
        index=agent_type_options.index(agent_data.get('agent_type')) if agent_data.get('agent_type') in agent_type_options else 0
    )
    # Handle the case where 'metacognitive_type' is not in agent_data
    metacognitive_options = ["None"] + list(get_metacognitive_prompt().keys())
    metacognitive_type = st.selectbox(
        f"{name} Metacognitive Type", 
        metacognitive_options, 
        key=f"{name}_metacognitive_type", 
        index=metacognitive_options.index(agent_data.get('metacognitive_type')) if agent_data.get('metacognitive_type') in metacognitive_options else 0
    )
    # Handle the case where 'voice_type' is not in agent_data
    voice_options = ["None"] + list(get_voice_prompt().keys())
    voice_type = st.selectbox(
        f"{name} Voice Type",
        voice_options,
        key=f"{name}_voice_type",
        index=voice_options.index(agent_data.get('voice_type')) if agent_data.get('voice_type') in voice_options else 0
    )
    corpus = st.selectbox(f"{name} Corpus", get_corpus_options(), key=f"{name}_corpus", index=get_corpus_options().index(agent_data.get('corpus')) if agent_data.get('corpus') in get_corpus_options() else 0)
    # Remove redundant 'value' argument
    temperature = st.slider(f"{name} Temperature", 0.0, 1.0, 0.7, key=f"{name}_temperature")
    # Remove redundant 'value' argument
    max_tokens = st.slider(f"{name} Max Tokens", 100, 32000, 4000, key=f"{name}_max_tokens")
    return {'model': model, 'agent_type': agent_type, 'metacognitive_type': metacognitive_type, 'voice_type': voice_type, 'corpus': corpus, 'temperature': temperature, 'max_tokens': max_tokens} # Include voice_type in the return

def generate_workflow(user_request: str):
    """
    Generates a workflow template based on a user request, including tasks and agents.

    Args:
        user_request: The user's request for the workflow.

    Returns:
        A tuple containing:
            - The generated workflow template as a list of Task objects, or None if generation failed.
            - A dictionary of generated agents, with agent names as keys and agent data as values.
    """
    # 1. Generate Workflow Template using LLM
    generation_prompt = f"""
    Generate a JSON list of tasks and agents to fulfill the following user request. 

    User Request: {user_request}

    Use a tree of thought approach to break down the user request into individual tasks. For each task, determine the most suitable agent type based on the task description and create an agent with a unique name.

    Available Agent Types:
    - Data Extractor: Extracts data from various sources, such as files or websites.
    - Data Analyst: Analyzes data, calculates statistics, and identifies trends.
    - Report Writer: Generates reports, summaries, and presentations based on data.
    - Code Generator: Writes code in various programming languages based on specifications.
    - Content Creator: Creates written content, such as articles, stories, and social media posts.
    - Language Translator: Translates text between different languages.
    - Chatbot: Engages in conversations and provides responses based on context.
    - Image Generator: Creates images based on descriptions or prompts.
    - Audio Transcriber: Transcribes audio recordings into text.
    - Task Planner: Breaks down complex tasks into smaller, manageable steps.

    JSON Output Structure:
    {{
        "tasks": [
            {{
                "name": "Task Name",
                "description": "Task Description",
                "deadline": "YYYY-MM-DD HH:MM:SS",
                "priority": "Low", "Medium", or "High",
                "completed": false,
                "agent": "Agent Name",
                "result": null 
            }},
            ...
        ],
        "agents": {{
            "Agent Name": {{
                "model": "Model Name",
                "agent_type": "Agent Type",
                "metacognitive_type": "Metacognitive Type",
                "voice_type": "Voice Type",
                "corpus": "Corpus Name",
                "temperature": 0.7,
                "max_tokens": 4000
            }},
            ...
        }}
    }}

    Example:
    {{
        "tasks": [
            {{
                "name": "Extract Data",
                "description": "Extract data from the CSV file at ./files/data.csv.",
                "deadline": "2024-01-01 12:00:00",
                "priority": "High",
                "completed": false,
                "agent": "Data Extractor Agent",
                "result": null
            }},
            {{
                "name": "Analyze Data",
                "description": "Analyze the extracted data.",
                "deadline": "2024-01-02 12:00:00",
                "priority": "Medium",
                "completed": false,
                "agent": "Data Analyst Agent",
                "result": null
            }},
            {{
                "name": "Generate Report",
                "description": "Generate a report from the analysis results.",
                "deadline": "2024-01-03 12:00:00",
                "priority": "Low",
                "completed": false,
                "agent": "Report Writer Agent",
                "result": null
            }}
        ],
        "agents": {{
            "Data Extractor Agent": {{
                "model": "mistral:7b-instruct-v0.2-q8_0",
                "agent_type": "Data Extractor",
                "metacognitive_type": "None",
                "voice_type": "None",
                "corpus": "None",
                "temperature": 0.7,
                "max_tokens": 4000
            }},
            "Data Analyst Agent": {{
                "model": "mistral:7b-instruct-v0.2-q8_0",
                "agent_type": "Data Analyst",
                "metacognitive_type": "None",
                "voice_type": "None",
                "corpus": "None",
                "temperature": 0.7,
                "max_tokens": 4000
            }},
            "Report Writer Agent": {{
                "model": "mistral:7b-instruct-v0.2-q8_0",
                "agent_type": "Report Writer",
                "metacognitive_type": "None",
                "voice_type": "None",
                "corpus": "None",
                "temperature": 0.7,
                "max_tokens": 4000
            }}
        }}
    }}
    """
    response = ollama.generate(model="mistral:7b-instruct-v0.2-q8_0", prompt=generation_prompt)
    generated_workflow = response['response']

    # 2. Preprocess generated workflow (remove extra characters and whitespace)
    # Find the starting position of the JSON list
    start_index = generated_workflow.find("{")
    if start_index != -1:
        generated_workflow = generated_workflow[start_index:]
    # Find the ending position of the JSON list
    end_index = generated_workflow.rfind("}")
    if end_index != -1:
        generated_workflow = generated_workflow[:end_index + 1]
    generated_workflow = generated_workflow.strip()  # Remove leading/trailing whitespace
    generated_workflow = generated_workflow.replace("'", '"')  # Replace single quotes with double quotes
    # Allow for more characters in the regular expression
    generated_workflow = re.sub(r"[^\w\s{}\[\]\":,./\-+]+", "", generated_workflow)  # Remove invalid characters

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

        # Create Task objects from the workflow data
        tasks = []
        for task_data in workflow_data['tasks']:
            task = Task(
                name=task_data.get("name"),
                agent=task_data.get("agent"),
                description=task_data.get("description"),
                deadline=pd.to_datetime(task_data.get("deadline"), errors='coerce'),
                priority=task_data.get("priority", "Medium"),
                completed=task_data.get("completed", False),
                result=task_data.get("result")
            )
            tasks.append(task)

        agents = workflow_data['agents']

        return tasks, agents

    except json.JSONDecodeError as e:
        print(f"Error loading or post-processing JSON: {e}")
        st.error(f"Failed to generate tasks. Invalid JSON: {e}")
        return None, None

def projects_main():
    # Load existing projects
    projects = load_projects()

    # App title
    st.title("🚀 Projects")

    # Display task statistics at the top
    st.subheader("📈 Task Statistics")
    if projects:
        total_tasks = 0
        completed_tasks = 0
        pending_tasks = 0
        high_priority_tasks = 0

        for project in projects:
            tasks = load_tasks(project)
            total_tasks += len(tasks)
            completed_tasks += sum(1 for task in tasks if task.completed)  # Use dot notation
            pending_tasks += sum(1 for task in tasks if not task.completed)  # Use dot notation
            high_priority_tasks += sum(1 for task in tasks if task.priority == 'High' and not task.completed)  # Use dot notation

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tasks", total_tasks)
        col2.metric("Completed Tasks", completed_tasks)
        col3.metric("Pending Tasks", pending_tasks)
        col4.metric("High Priority Pending", high_priority_tasks)
    else:
        st.info("No projects available for statistics.")

    # Project management
    with st.expander("🚀 Add New Project"):
        new_project = st.text_input("New Project Name")
        if st.button("Add Project"):
            if new_project and new_project not in projects:
                projects.append(new_project)
                save_projects(projects)
                st.success(f"Project '{new_project}' added successfully!")
                # Remove st.rerun() here
            elif new_project in projects:
                st.warning(f"Project '{new_project}' already exists.")
            else:
                st.warning("Please enter a project name.")

    # Project selection
    selected_project = st.selectbox("🚀 Select Project", projects)

    if selected_project:
        # Load tasks for the selected project
        tasks = load_tasks(selected_project)

        # Option to auto-generate tasks
        st.subheader("🤖 Auto-Generate Tasks")
        user_request = st.text_area("Enter your project request:")
        if st.button("Generate Tasks"):
            if user_request:
                generated_tasks, generated_agents = generate_workflow(user_request)
                if generated_tasks and generated_agents:
                    tasks.extend(generated_tasks)  # Add generated tasks to existing tasks
                    # Update agents in session state
                    if 'agents' not in st.session_state:
                        st.session_state.agents = {}
                    st.session_state.agents.update(generated_agents)
                    save_tasks(selected_project, tasks)
                    st.success("Tasks and agents generated and added to the project!")
                    # Remove st.rerun() here
                else:
                    st.error("Failed to generate tasks.")

        # Display and manage tasks
        st.subheader(f"📋 Tasks for {selected_project}")

        if tasks:
            df = pd.DataFrame([task.__dict__ for task in tasks])  # Convert Task objects to dictionaries
            df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')
            df = df[['name', 'description', 'deadline', 'priority', 'completed', 'agent', 'result']]

            # Convert the DataFrame to a Streamlit editable dataframe
            edited_df = st.data_editor(
                df,
                column_config={
                    "completed": st.column_config.CheckboxColumn(
                        "Completed",
                        help="Mark task as completed",
                        default=False,
                    ),
                    "deadline": st.column_config.DatetimeColumn(
                        "Deadline",
                        format="YYYY-MM-DD HH:mm:ss",
                    ),
                },
                hide_index=True,
                num_rows="dynamic"
            )

            # Update tasks based on edited dataframe
            updated_tasks = edited_df.to_dict('records')
            if updated_tasks != tasks:
                tasks = [Task(**task) for task in updated_tasks if pd.notna(task['deadline'])]  # Create Task objects from dictionaries
                save_tasks(selected_project, tasks)
                st.success("Tasks updated successfully!")
                # Remove st.rerun() here
        else:
            st.info(f"No tasks found for {selected_project}. Add a task to get started!")

        # Task input form
        st.subheader(f"📌 Add New Task to {selected_project}")
        task_name = st.text_input("Task Name")
        task_description = st.text_area("Task Description")

        # Create a 3-column layout for Deadline Date, Deadline Time, and Priority
        col1, col2, col3 = st.columns(3)
        with col1:
            task_deadline = st.date_input("Deadline Date")
        with col2:
            task_time = st.time_input("Deadline Time")
        with col3:
            task_priority = st.selectbox("Priority", ["Low", "Medium", "High"])

        # Dynamic agent selection
        if 'agents' not in st.session_state:
            st.session_state.agents = {}
        agents = st.session_state.agents
        agent_names = list(agents.keys())
        task_agent = st.selectbox("AI Agent", ["None"] + agent_names)

        if st.button("📌 Add Task"):
            deadline = pd.Timestamp(datetime.combine(task_deadline, task_time))
            if pd.notna(deadline):
                task = Task(
                    name=task_name,
                    description=task_description,
                    deadline=deadline,
                    priority=task_priority,
                    completed=False,
                    agent=task_agent,
                    result=None
                )
                tasks.append(task)
                save_tasks(selected_project, tasks)
                st.success("Task added successfully!")
                # Remove st.rerun() here
            else:
                st.error("Invalid deadline. Please select a valid date and time.")

        # Define AI agents
        st.subheader("🧑 AI Agents")
        # Use agents from session state
        if 'agents' not in st.session_state:
            st.session_state.agents = {}
        agents = st.session_state.agents
        agent_names = list(agents.keys())
        num_agents = st.number_input("Number of Agents", min_value=len(agent_names), max_value=10, value=len(agent_names))
        
        # Allow adding new agents
        if num_agents > len(agent_names):
            for i in range(len(agent_names), num_agents):
                agent_name = f"Agent {i+1}"
                agents[agent_name] = {'model': 'mistral:7b-instruct-v0.2-q8_0', 'agent_type': 'None', 'metacognitive_type': 'None', 'voice_type': 'None', 'corpus': 'None', 'temperature': 0.7, 'max_tokens': 4000}
                agent_names.append(agent_name)

        for agent_name in agent_names:
            with st.expander(f"🧑 {agent_name} Parameters"):
                agents[agent_name] = define_agent_block(agent_name, agents[agent_name])

        # Run AI agents on tasks
        if st.button("🏃‍♂️🏃‍♀️💨 Run AI Agents on Tasks"):
            previous_responses = []  # Store responses from previous agents
            for task in tasks:
                if task.agent != "None" and not task.completed:  # Use dot notation
                    if task.agent in agents:
                        agent_data = agents[task.agent]
                        model = agent_data['model']
                        agent_type = agent_data['agent_type']
                        metacognitive_type = agent_data['metacognitive_type']
                        voice_type = agent_data['voice_type']
                        corpus = agent_data['corpus']
                        temperature = agent_data['temperature']
                        max_tokens = agent_data['max_tokens']
                        with st.spinner(f"Running {task.agent} on task '{task.name}'..."):  # Use dot notation
                            result = ai_agent(task.description, model, agent_type, metacognitive_type, voice_type, corpus, temperature, max_tokens, previous_responses) # Pass voice_type to ai_agent
                            st.write(f"{task.agent} Output for '{task.name}': {result}")  # Use dot notation
                            task.result = result  # Update task result
                            previous_responses.append(result)  # Add response to context for next agent
                    else:
                        st.warning(f"Agent {task.agent} not defined. Please define the agent before running.")
            # Save tasks after running agents
            save_tasks(selected_project, tasks)

    # File upload and download
    with st.expander("📦 Import/Export Tasks"):

        uploaded_file = st.file_uploader("Upload Tasks JSON", type="json")
        if uploaded_file is not None:
            try:
                new_tasks = json.load(uploaded_file)
                for task in new_tasks:
                    task['deadline'] = pd.to_datetime(task['deadline'], errors='coerce')
                tasks = [task for task in new_tasks if pd.notna(task['deadline'])]
                save_tasks(selected_project, tasks)
                st.success("Tasks imported successfully!")
                # Remove st.rerun() here
            except json.JSONDecodeError:
                st.error("Error: Invalid JSON file. Please upload a valid JSON file.")

        if st.button("Download Tasks"):
            json_str = json.dumps(tasks, cls=DateTimeEncoder)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="{selected_project}_tasks.json">Download JSON file</a>'
            st.markdown(href, unsafe_allow_html=True)
