# projects.py
import streamlit as st
import pandas as pd
import json
import os
import uuid
from datetime import datetime
import base64
from ollama_utils import get_available_models, call_ollama_endpoint
from openai_utils import call_openai_api, OPENAI_MODELS
from ollama_utils import load_api_keys
from groq_utils import call_groq_api, GROQ_MODELS
from prompts import get_agent_prompt, get_metacognitive_prompt, get_voice_prompt
import re
from functools import lru_cache

class Task:
    def __init__(self, name, description, deadline=None, priority="Medium", completed=False, agent="None", result=None, task_id=None):
        self.task_id = task_id if task_id else str(uuid.uuid4())
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

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            if pd.isna(obj):
                return None
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, Task):
            return obj.__dict__
        return super().default(obj)

def load_projects():
    try:
        with open('projects/projects.json', 'r') as f:
            content = f.read()
            return json.loads(content) if content else []
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        st.error(f"Error loading projects: {str(e)}. Starting with an empty project list.")
        return []

def save_projects(projects):
    with open('projects/projects.json', 'w') as f:
        json.dump(projects, f)

def load_tasks(project_name):
    try:
        with open(f'projects/{project_name}_tasks.json', 'r') as f:
            content = f.read()
            task_data = json.loads(content) if content else []
        tasks = []
        for data in task_data:
            task = Task(
                name=data['name'],
                description=data['description'],
                deadline=pd.to_datetime(data['deadline'], errors='coerce'),
                priority=data['priority'],
                completed=data['completed'],
                agent=data['agent'],
                result=data['result'],
                task_id=data['task_id']
            )
            tasks.append(task)
        return tasks
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        st.error(f"Error loading tasks for {project_name}: {str(e)}. Starting with an empty task list.")
        return []

# Callback function to update task status in session state
def update_task_status(task_index, status, result=None):
    if task_index < len(st.session_state.bm_tasks):
        st.session_state.bm_tasks[task_index]["status"] = status
        if result is not None:
            st.session_state.bm_tasks[task_index]["result"] = result

def handle_user_input(step, task_data):
    """Handles user input for a specific task step."""
    user_input_config = step.get("user_input")
    if user_input_config:
        input_type = user_input_config["type"]
        prompt = user_input_config["prompt"]

        if input_type == "file_path":
            file_path = st.text_input(prompt, key=f"user_input_{step['agent']}")
            if file_path:
                task_data["file_path"] = file_path
            else:
                st.warning("Please provide a file path.")
                return False  # Indicate that user input is not complete

        elif input_type == "options":
            options = user_input_config.get("options", [])
            selected_option = st.selectbox(prompt, options, key=f"user_input_{step['agent']}")
            if selected_option:
                task_data["selected_option"] = selected_option
            else:
                st.warning("Please select an option.")
                return False  # Indicate that user input is not complete

        elif input_type == "confirmation":
            if not st.button(prompt, key=f"user_input_{step['agent']}"):
                st.warning("Task skipped due to unconfirmed user input.")
                return False  # Indicate that user input is not complete

    return True  # Indicate that user input is complete

def save_tasks(project_name, tasks):
    tasks = [task for task in tasks if pd.notna(task.deadline)]
    with open(f'projects/{project_name}_tasks.json', 'w') as f:
        json.dump([task.__dict__ for task in tasks], f, cls=DateTimeEncoder)

def load_agents(project_name):
    try:
        with open(f'projects/{project_name}_agents.json', 'r') as f:
            content = f.read()
            return json.loads(content) if content else {}
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        st.error(f"Error loading agents for {project_name}: {str(e)}. Starting with an empty agent list.")
        return {}

def save_agents(project_name, agents):
    with open(f'projects/{project_name}_agents.json', 'w') as f:
        json.dump(agents, f)

def get_corpus_options():
    corpus_folder = "corpus"
    if not os.path.exists(corpus_folder):
        os.makedirs(corpus_folder)
    return ["None"] + [f for f in os.listdir(corpus_folder) if os.path.isdir(os.path.join(corpus_folder, f))]

def get_corpus_context_from_db(corpus_folder, corpus, user_input):
    return "Sample context from the corpus database."


def get_all_models():
    return get_available_models() + OPENAI_MODELS + GROQ_MODELS

def ai_agent(user_input, model, agent_type, metacognitive_type, voice_type, corpus, temperature, max_tokens, previous_responses=[]):
    combined_prompt = ""
    if agent_type != "None":
        combined_prompt += get_agent_prompt()[agent_type] + "\n\n"
    if metacognitive_type != "None":
        combined_prompt += get_metacognitive_prompt()[metacognitive_type] + "\n\n"
    if voice_type != "None":
        combined_prompt += get_voice_prompt()[voice_type] + "\n\n"

    if corpus != "None":
        corpus_context = get_corpus_context_from_db("corpus", corpus, user_input)
        combined_prompt += f"Context: {corpus_context}\n\n"

    for i, response in enumerate(previous_responses):
        combined_prompt += f"Response {i+1}: {response}\n\n"

    final_prompt = f"{combined_prompt}User: {user_input}"

    api_keys = load_api_keys()  # Load API keys

    if model in OPENAI_MODELS:
        response = call_openai_api(model, [{"role": "user", "content": final_prompt}], temperature=temperature, max_tokens=max_tokens, openai_api_key=api_keys.get("openai_api_key"))
    elif model in GROQ_MODELS:
        response = call_groq_api(model, final_prompt, temperature=temperature, max_tokens=max_tokens, groq_api_key=api_keys.get("groq_api_key"))
    else:
        response, _, _, _ = call_ollama_endpoint(model, prompt=final_prompt, temperature=temperature, max_tokens=max_tokens)
    
    return response

def define_agent_block(name, agent_data=None):
    if agent_data is None:
        agent_data = {}
    model = st.selectbox(f"{name} Model", get_all_models(), key=f"{name}_model", index=get_all_models().index(agent_data.get('model')) if agent_data.get('model') in get_all_models() else 0)
    agent_type_options = ["None"] + list(get_agent_prompt().keys())
    agent_type = st.selectbox(
        f"{name} Agent Type",
        agent_type_options,
        key=f"{name}_agent_type",
        index=agent_type_options.index(agent_data.get('agent_type')) if agent_data.get('agent_type') in agent_type_options else 0
    )
    metacognitive_options = ["None"] + list(get_metacognitive_prompt().keys())
    metacognitive_type = st.selectbox(
        f"{name} Metacognitive Type", 
        metacognitive_options, 
        key=f"{name}_metacognitive_type", 
        index=metacognitive_options.index(agent_data.get('metacognitive_type')) if agent_data.get('metacognitive_type') in metacognitive_options else 0
    )
    voice_options = ["None"] + list(get_voice_prompt().keys())
    voice_type = st.selectbox(
        f"{name} Voice Type",
        voice_options,
        key=f"{name}_voice_type",
        index=voice_options.index(agent_data.get('voice_type')) if agent_data.get('voice_type') in voice_options else 0
    )
    corpus = st.selectbox(f"{name} Corpus", get_corpus_options(), key=f"{name}_corpus", index=get_corpus_options().index(agent_data.get('corpus')) if agent_data.get('corpus') in get_corpus_options() else 0)
    temperature = st.slider(f"{name} Temperature", 0.0, 1.0, agent_data.get('temperature', 0.7), key=f"{name}_temperature")
    max_tokens = st.slider(f"{name} Max Tokens", 4000, 128000, agent_data.get('max_tokens', 4000), key=f"{name}_max_tokens" , step=1000)
    return {'model': model, 'agent_type': agent_type, 'metacognitive_type': metacognitive_type, 'voice_type': voice_type, 'corpus': corpus, 'temperature': temperature, 'max_tokens': max_tokens}

class ProjectManagerAgent:
    def __init__(self, model: str, agent_type: str, temperature: float, max_tokens: int):
        self.model = model
        self.agent_type = agent_type
        self.temperature = temperature
        self.max_tokens = max_tokens

    @lru_cache(maxsize=None)
    def generate_workflow(self, user_request: str):
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
        """

        api_keys = load_api_keys()  # Load API keys

        if self.model in OPENAI_MODELS:
            response = call_openai_api(self.model, [{"role": "user", "content": generation_prompt}], temperature=self.temperature, max_tokens=self.max_tokens, openai_api_key=api_keys.get("openai_api_key"))
        elif self.model in GROQ_MODELS:
            response = call_groq_api(self.model, generation_prompt, temperature=self.temperature, max_tokens=self.max_tokens, groq_api_key=api_keys.get("groq_api_key"))
        else:
            response = call_ollama_endpoint(
                self.model,
                prompt=generation_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )[0]  # Assuming call_ollama_endpoint returns a tuple, we take the first element

        generated_workflow = response

        generated_workflow = generated_workflow.strip()
        generated_workflow = generated_workflow.replace("'", '"')
        generated_workflow = re.sub(r"[^\\w\\s{}\\\\\\[\\]\":,./\\-+]+", "", generated_workflow)

        if not generated_workflow.startswith("{"):
            generated_workflow = "{" + generated_workflow
        if not generated_workflow.endswith("}"):
            generated_workflow = generated_workflow + "}"

        generated_workflow = re.sub(r"(\\s*)'(\w+)'(\\s*):", r'\1"\2"\3:', generated_workflow)
        generated_workflow = re.sub(r"([}\]])(\\s*)(?=[{\\\\\\[\\]\"a-zA-Z])", r"\1,\2", generated_workflow)

        st.write("Generated Workflow:", generated_workflow)

        try:
            generated_workflow = generated_workflow.strip()
            if not generated_workflow.startswith("{"):
                generated_workflow = "{" + generated_workflow
            if not generated_workflow.endswith("}"):
                generated_workflow = generated_workflow + "}"
            
            workflow_data = json.loads(generated_workflow)

            tasks = []
            for task_data in workflow_data.get('tasks', []):
                task = Task(
                    name=task_data.get("name"),
                    description=task_data.get("description"),
                    deadline=pd.to_datetime(task_data.get("deadline"), errors='coerce'),
                    priority=task_data.get("priority", "Medium"),
                    completed=task_data.get("completed", False),
                    agent=task_data.get("agent"),
                    result=task_data.get("result")
                )
                tasks.append(task)

            agents = workflow_data.get('agents', {})

            return tasks, agents

        except json.JSONDecodeError as e:
            st.error(f"Error loading or post-processing JSON: {e}")
            st.error(f"Generated workflow: {generated_workflow}")
            return None, None

def initialize_session_state():
    if 'projects' not in st.session_state:
        st.session_state.projects = load_projects()
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = None
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []
    if 'agents' not in st.session_state:
        st.session_state.agents = {}
    if 'generated_tasks' not in st.session_state:
        st.session_state.generated_tasks = []
    if 'generated_agents' not in st.session_state:
        st.session_state.generated_agents = {}
    if 'project_manager_settings' not in st.session_state:
        all_models = get_all_models()
        st.session_state.project_manager_settings = {
            'model': all_models[0] if all_models else 'gpt-3.5-turbo',  # Use the first available model or a default
            'agent_type': 'Task Planner',
            'temperature': 0.7,
            'max_tokens': 4000,
        }

def projects_main():
    initialize_session_state()

    if 'agents' not in st.session_state:
        st.session_state.agents = {}

    projects = load_projects()

    st.title("🚀 Projects")

    # Sidebar configuration
    with st.sidebar:
        with st.expander("🤖 Project Manager Settings", expanded=False):
            all_models = get_all_models()  # Get all available models including Ollama, OpenAI, and Groq
            st.session_state.project_manager_settings['model'] = st.selectbox(
                "Select Model for Project Manager",
                all_models,
                index=all_models.index(st.session_state.project_manager_settings['model']) if st.session_state.project_manager_settings['model'] in all_models else 0
            )
            st.session_state.project_manager_settings['agent_type'] = "Task Planner"
            st.write(f"Agent Type: {st.session_state.project_manager_settings['agent_type']}")
            
            st.session_state.project_manager_settings['temperature'] = st.slider("Temperature", 0.0, 1.0, st.session_state.project_manager_settings['temperature'], step=0.1)
            st.session_state.project_manager_settings['max_tokens'] = st.slider("Max Tokens", 4000, 128000, st.session_state.project_manager_settings['max_tokens'], step=1000)

    # Display task statistics
    if projects:
        total_tasks = completed_tasks = pending_tasks = high_priority_tasks = 0
        for project in projects:
            tasks = load_tasks(project)
            total_tasks += len(tasks)
            completed_tasks += sum(1 for task in tasks if task.completed)
            pending_tasks += sum(1 for task in tasks if not task.completed)
            high_priority_tasks += sum(1 for task in tasks if task.priority == 'High' and not task.completed)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tasks", total_tasks)
        col2.metric("Completed Tasks", completed_tasks)
        col3.metric("Pending Tasks", pending_tasks)
        col4.metric("High Priority Pending", high_priority_tasks)
    else:
        st.info("No projects available for statistics.")

    col1, col2, col3, col4 = st.columns(4, gap="small", vertical_alignment="bottom")
    
    with col1:
        selected_project = st.selectbox("🚀 Select Project", projects)
        if selected_project:
            st.session_state.tasks = load_tasks(selected_project)
            st.session_state.agents.update(load_agents(selected_project))

    with col2:
        with st.expander("🚀 Add New Project"):
            new_project = st.text_input("New Project Name")
            if st.button("Add Project"):
                if new_project and new_project not in projects:
                    projects.append(new_project)
                    save_projects(projects)
                    st.success(f"🟢 Project '{new_project}' added successfully!")
                elif new_project in projects:
                    st.warning(f"Project '{new_project}' already exists.")
                else:
                    st.warning("Please enter a project name.")

    with col3:
        with st.expander("🗑️ Delete Project"):
            project_to_delete = st.selectbox("Select Project to Delete", projects)
            if st.button("Delete Project"):
                if project_to_delete in projects:
                    projects.remove(project_to_delete)
                    task_file = f'projects/{project_to_delete}_tasks.json'
                    agent_file = f'projects/{project_to_delete}_agents.json'
                    if os.path.exists(task_file):
                        os.remove(task_file)
                    if os.path.exists(agent_file):
                        os.remove(agent_file)
                    save_projects(projects)
                    st.success(f"🟢 Project '{project_to_delete}' deleted successfully!")
                else:
                    st.warning("Please select a valid project to delete.")

    with col4:
        with st.expander("📦 Import/Export Tasks"):
            uploaded_file = st.file_uploader("Upload Tasks JSON", type="json")
            if uploaded_file is not None:
                try:
                    new_tasks = json.load(uploaded_file)
                    for task in new_tasks:
                        task['deadline'] = pd.to_datetime(task['deadline'], errors='coerce')
                    tasks = [Task(**task) for task in new_tasks if pd.notna(task['deadline'])]
                    save_tasks(selected_project, tasks)
                    st.success("🟢 Tasks imported successfully!")
                except json.JSONDecodeError:
                    st.error("Error: Invalid JSON file. Please upload a valid JSON file.")

            if st.button("Download Tasks"):
                json_str = json.dumps([task.__dict__ for task in st.session_state.tasks], cls=DateTimeEncoder)
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:file/json;base64,{b64}" download="{selected_project}_tasks.json">Download JSON file</a>'
                st.markdown(href, unsafe_allow_html=True)

    # Option to auto-generate tasks
    st.subheader("🤖 Auto-Generate Tasks")
    user_request = st.text_area("Enter your project request:")

    if selected_project:
        if st.button("✅ Generate Tasks"):
            if user_request:
                with st.spinner("Generating tasks and agents..."):
                    # Create a ProjectManagerAgent instance
                    project_manager = ProjectManagerAgent(
                        model=st.session_state.project_manager_settings['model'],
                        agent_type=st.session_state.project_manager_settings['agent_type'],
                        temperature=st.session_state.project_manager_settings['temperature'],
                        max_tokens=st.session_state.project_manager_settings['max_tokens']
                    )

                    # Generate the workflow using the agent
                    generated_tasks, generated_agents = project_manager.generate_workflow(user_request)

                    if generated_tasks and generated_agents:
                        st.session_state.tasks.extend(generated_tasks)
                        st.session_state.agents.update(generated_agents)
                        st.session_state.generated_tasks = generated_tasks
                        st.session_state.generated_agents = generated_agents
                        save_tasks(selected_project, st.session_state.tasks)
                        save_agents(selected_project, st.session_state.agents)
                        st.success("🟢 Tasks and agents generated and added to the project!")
                        
                        # Display generated tasks
                        st.subheader("Generated Tasks")
                        for task in generated_tasks:
                            st.write(f"Task: {task.name}")
                            st.write(f"Description: {task.description}")
                            st.write(f"Deadline: {task.deadline}")
                            st.write(f"Priority: {task.priority}")
                            st.write(f"Agent: {task.agent}")
                            st.write("---")
                        
                        # Display generated agents
                        st.subheader("Generated Agents")
                        for agent_name, agent_data in generated_agents.items():
                            st.write(f"Agent: {agent_name}")
                            st.write(f"Model: {agent_data['model']}")
                            st.write(f"Agent Type: {agent_data['agent_type']}")
                            st.write("---")
                    else:
                        st.error("Failed to generate tasks. Please try again with a different request.")

        if selected_project:
            # Task input form
            with st.expander(f"📌 Manually add a new task to {selected_project}"):
                task_name = st.text_input("Task Name")
                task_description = st.text_area("Task Description")

                col1, col2, col3 = st.columns(3)
                with col1:
                    task_deadline = st.date_input("Deadline Date")
                with col2:
                    task_time = st.time_input("Deadline Time")
                with col3:
                    task_priority = st.selectbox("Priority", ["Low", "Medium", "High"])

                agent_names = list(st.session_state.agents.keys())
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
                        st.session_state.tasks.append(task)
                        save_tasks(selected_project, st.session_state.tasks)
                        st.success("🟢 Task added successfully!")
                    else:
                        st.error("Invalid deadline. Please select a valid date and time.")

        # Display and manage tasks
        st.subheader(f"📋 Tasks for {selected_project}")

        if st.session_state.tasks:
            df = pd.DataFrame([task.__dict__ for task in st.session_state.tasks])
            df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')
            df = df[['name', 'description', 'deadline', 'priority', 'completed', 'agent', 'result']]

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

            updated_tasks = edited_df.to_dict('records')
            if updated_tasks != [task.__dict__ for task in st.session_state.tasks]:
                st.session_state.tasks = [Task(**task) for task in updated_tasks if pd.notna(task['deadline'])]
                save_tasks(selected_project, st.session_state.tasks)
                st.success("🟢 Tasks updated successfully!")
        else:
            st.info(f"No tasks found for {selected_project}. Add a task to get started!")

        # Define AI agents
        st.subheader("🧑 AI Agents")
        agent_names = list(st.session_state.agents.keys())
        num_agents = st.number_input("Number of Agents", min_value=len(agent_names), max_value=10, value=len(agent_names))
        
        if num_agents > len(agent_names):
            for i in range(len(agent_names), num_agents):
                agent_name = f"Agent {i+1}"
                st.session_state.agents[agent_name] = {'model': 'mistral:instruct', 'agent_type': 'None', 'metacognitive_type': 'None', 'voice_type': 'None', 'corpus': 'None', 'temperature': 0.7, 'max_tokens': 4000}
                agent_names.append(agent_name)

        new_agent_names = []
        for agent_name in agent_names:
            with st.expander(f"🧑 {agent_name} Parameters"):
                new_agent_name = st.text_input(f"{agent_name} Name", value=agent_name, key=f"{agent_name}_name")
                new_agent_names.append(new_agent_name)
                st.session_state.agents[agent_name] = define_agent_block(agent_name, st.session_state.agents[agent_name])

        if st.button("🧑 Save Agent Settings"):
            renamed_agents = {new_name: st.session_state.agents[old_name] for new_name, old_name in zip(new_agent_names, agent_names)}
            st.session_state.agents = renamed_agents
            save_agents(selected_project, renamed_agents)
            st.success("🟢 Agent settings saved!")

        # Run AI agents on tasks
        if st.button("🏃‍♂️🏃‍♀️💨 Run AI Agents on Tasks"):
            previous_responses = []
            for task in st.session_state.tasks:
                if task.agent != "None" and not task.completed:
                    if task.agent in st.session_state.agents:
                        agent_data = st.session_state.agents[task.agent]
                        with st.spinner(f"Running {task.agent} on task '{task.name}'..."):
                            result = ai_agent(task.description, **agent_data, previous_responses=previous_responses)
                            st.write(f"{task.agent} Output for '{task.name}': {result}")
                            task.result = result
                            previous_responses.append(result)
                    else:
                        st.warning(f"Agent {task.agent} not defined. Please define the agent before running.")
            save_tasks(selected_project, st.session_state.tasks)
            st.success("🟢 AI agents completed their tasks!")

def load_api_keys():
    if os.path.exists("api_keys.json"):
        with open("api_keys.json", "r") as f:
            return json.load(f)
    return {}

if __name__ == "__main__":
    projects_main()