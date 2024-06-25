import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import base64
from ollama_utils import get_available_models, ollama
from prompts import get_agent_prompt, get_metacognitive_prompt, get_voice_prompt

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
            tasks = json.loads(content)
        for task in tasks:
            task['deadline'] = pd.to_datetime(task['deadline'], errors='coerce')
        return tasks
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        st.error(f"Error loading tasks for {project_name}: {str(e)}. Starting with an empty task list.")
        return []

# Function to save tasks for a specific project
def save_tasks(project_name, tasks):
    # Remove tasks with NaT deadlines
    tasks = [task for task in tasks if pd.notna(task['deadline'])]
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
from ollama_utils import call_ollama_endpoint  # Import the correct function

def ai_agent(user_input, model, agent_type, metacognitive_type, voice_type, corpus, temperature, max_tokens):
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
        final_prompt = f"{combined_prompt}Context: {corpus_context}\n\nUser: {user_input}"
    else:
        final_prompt = f"{combined_prompt}User: {user_input}"

    # Generate response using call_ollama_endpoint
    response, _, _, _ = call_ollama_endpoint(model, prompt=final_prompt, temperature=temperature, max_tokens=max_tokens)

    return response

# Function to define agent parameters
def define_agent_block(name):
    model = st.selectbox(f"{name} Model", get_available_models(), key=f"{name}_model")
    agent_type = st.selectbox(f"{name} Agent Type", ["None"] + list(get_agent_prompt().keys()), key=f"{name}_agent_type")
    metacognitive_type = st.selectbox(f"{name} Metacognitive Type", ["None"] + list(get_metacognitive_prompt().keys()), key=f"{name}_metacognitive_type")
    voice_type = st.selectbox(f"{name} Voice Type", ["None"] + list(get_voice_prompt().keys()), key=f"{name}_voice_type") # Add voice_type selection
    corpus = st.selectbox(f"{name} Corpus", get_corpus_options(), key=f"{name}_corpus")
    temperature = st.slider(f"{name} Temperature", 0.0, 1.0, 0.7, key=f"{name}_temperature")
    max_tokens = st.slider(f"{name} Max Tokens", 100, 32000, 4000, key=f"{name}_max_tokens")
    return model, agent_type, metacognitive_type, voice_type, corpus, temperature, max_tokens # Include voice_type in the return

def projects_main():
    # Load existing projects
    projects = load_projects()

    # App title
    st.title("Projects")

    # Display task statistics at the top
    st.subheader("Task Statistics")
    if projects:
        total_tasks = 0
        completed_tasks = 0
        pending_tasks = 0
        high_priority_tasks = 0
        
        for project in projects:
            tasks = load_tasks(project)
            total_tasks += len(tasks)
            completed_tasks += sum(1 for task in tasks if task['completed'])
            pending_tasks += sum(1 for task in tasks if not task['completed'])
            high_priority_tasks += sum(1 for task in tasks if task['priority'] == 'High' and not task['completed'])
            
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tasks", total_tasks)
        col2.metric("Completed Tasks", completed_tasks)
        col3.metric("Pending Tasks", pending_tasks)
        col4.metric("High Priority Pending", high_priority_tasks)
    else:
        st.info("No projects available for statistics.")

    # Project selection
    selected_project = st.selectbox("Select Project", projects)

    if selected_project:
        # Load tasks for the selected project
        tasks = load_tasks(selected_project)

        # Display and manage tasks
        st.subheader(f"Tasks for {selected_project}")

        if tasks:
            df = pd.DataFrame(tasks)
            df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')
            df = df[['name', 'description', 'deadline', 'priority', 'completed', 'agent']]

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
                tasks = [task for task in updated_tasks if pd.notna(task['deadline'])]
                save_tasks(selected_project, tasks)
                st.success("Tasks updated successfully!")
                st.rerun()
        else:
            st.info(f"No tasks found for {selected_project}. Add a task to get started!")

        # Task input form
        st.subheader(f"Add New Task to {selected_project}")
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
        num_agents = st.session_state.get('num_agents', 1)
        task_agent = st.selectbox("AI Agent", ["None"] + [f"Agent {i+1}" for i in range(num_agents)])

        if st.button("Add Task"):
            deadline = pd.Timestamp(datetime.combine(task_deadline, task_time))
            if pd.notna(deadline):
                task = {
                    "name": task_name,
                    "description": task_description,
                    "deadline": deadline,
                    "priority": task_priority,
                    "completed": False,
                    "agent": task_agent
                }
                tasks.append(task)
                save_tasks(selected_project, tasks)
                st.success("Task added successfully!")
                st.rerun()
            else:
                st.error("Invalid deadline. Please select a valid date and time.")

        # Define AI agents
        st.subheader("AI Agents")
        num_agents = st.number_input("Number of Agents", min_value=1, max_value=10, value=st.session_state.get('num_agents', 1))
        st.session_state.num_agents = num_agents

        agents = []
        for i in range(num_agents):
            with st.expander(f"Agent {i+1} Parameters"):
                agents.append(define_agent_block(f"Agent {i+1}"))

        # Run AI agents on tasks
        if st.button("Run AI Agents on Tasks"):
            for task in tasks:
                if task['agent'] != "None" and not task['completed']:
                    agent_index = int(task['agent'].split()[1]) - 1
                    if agent_index < len(agents):
                        model, agent_type, metacognitive_type, voice_type, corpus, temperature, max_tokens = agents[agent_index] # Unpack voice_type
                        with st.spinner(f"Running {task['agent']} on task '{task['name']}'..."):
                            result = ai_agent(task['description'], model, agent_type, metacognitive_type, voice_type, corpus, temperature, max_tokens) # Pass voice_type to ai_agent
                            st.write(f"{task['agent']} Output for '{task['name']}': {result}")
                    else:
                        st.warning(f"Agent {agent_index + 1} not defined. Please define the agent before running.")

    # Project management
    with st.expander("Add New Project"):
        st.subheader("Project Management")
        new_project = st.text_input("New Project Name")
        if st.button("Add Project"):
            if new_project and new_project not in projects:
                projects.append(new_project)
                save_projects(projects)
                st.success(f"Project '{new_project}' added successfully!")
                st.rerun()
            elif new_project in projects:
                st.warning(f"Project '{new_project}' already exists.")
            else:
                st.warning("Please enter a project name.")

    # File upload and download
    with st.expander("Import/Export Tasks"):
        st.subheader("Import/Export Tasks")

        uploaded_file = st.file_uploader("Upload Tasks JSON", type="json")
        if uploaded_file is not None:
            try:
                new_tasks = json.load(uploaded_file)
                for task in new_tasks:
                    task['deadline'] = pd.to_datetime(task['deadline'], errors='coerce')
                tasks = [task for task in new_tasks if pd.notna(task['deadline'])]
                save_tasks(selected_project, tasks)
                st.success("Tasks imported successfully!")
                st.rerun()
            except json.JSONDecodeError:
                st.error("Error: Invalid JSON file. Please upload a valid JSON file.")

        if st.button("Download Tasks"):
            json_str = json.dumps(tasks, cls=DateTimeEncoder)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="{selected_project}_tasks.json">Download JSON file</a>'
            st.markdown(href, unsafe_allow_html=True)
