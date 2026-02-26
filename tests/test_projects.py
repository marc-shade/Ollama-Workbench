"""
Test suite for projects.py - Project management system
"""

import pytest
import json
import os
import tempfile
import uuid
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, call, mock_open
from pathlib import Path
from datetime import datetime, date, time
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTask:
    """Test Task class"""
    
    def test_task_creation_with_defaults(self):
        """Test Task creation with default values"""
        from ollama_workbench.workflows.projects import Task
        
        task = Task("Test Task", "Test Description")
        
        assert task.name == "Test Task"
        assert task.description == "Test Description"
        assert task.deadline is None
        assert task.priority == "Medium"
        assert task.completed is False
        assert task.agent == "None"
        assert task.result is None
        assert task.task_id is not None
        assert isinstance(task.task_id, str)
    
    def test_task_creation_with_custom_values(self):
        """Test Task creation with custom values"""
        from ollama_workbench.workflows.projects import Task
        
        deadline = pd.Timestamp("2023-12-31 23:59:59")
        task_id = str(uuid.uuid4())
        
        task = Task(
            name="Custom Task",
            description="Custom Description",
            deadline=deadline,
            priority="High",
            completed=True,
            agent="Test Agent",
            result="Task Result",
            task_id=task_id
        )
        
        assert task.name == "Custom Task"
        assert task.description == "Custom Description"
        assert task.deadline == deadline
        assert task.priority == "High"
        assert task.completed is True
        assert task.agent == "Test Agent"
        assert task.result == "Task Result"
        assert task.task_id == task_id
    
    def test_task_creation_with_provided_id(self):
        """Test Task creation with provided task ID"""
        from ollama_workbench.workflows.projects import Task
        
        custom_id = "custom-task-id-123"
        task = Task("Test", "Description", task_id=custom_id)
        
        assert task.task_id == custom_id


class TestDateTimeEncoder:
    """Test DateTimeEncoder class"""
    
    def test_encode_pandas_timestamp(self):
        """Test encoding pandas Timestamp"""
        from ollama_workbench.workflows.projects import DateTimeEncoder, Task
        
        encoder = DateTimeEncoder()
        timestamp = pd.Timestamp("2023-01-01 12:00:00")
        
        result = encoder.default(timestamp)
        assert result == "2023-01-01 12:00:00"
    
    def test_encode_pandas_nat(self):
        """Test encoding pandas NaT (Not a Time)"""
        from ollama_workbench.workflows.projects import DateTimeEncoder
        
        encoder = DateTimeEncoder()
        nat = pd.NaT
        
        result = encoder.default(nat)
        assert result is None
    
    def test_encode_task_object(self):
        """Test encoding Task object"""
        from ollama_workbench.workflows.projects import DateTimeEncoder, Task
        
        encoder = DateTimeEncoder()
        task = Task("Test Task", "Description", priority="High")
        
        result = encoder.default(task)
        assert isinstance(result, dict)
        assert result["name"] == "Test Task"
        assert result["description"] == "Description"
        assert result["priority"] == "High"
    
    def test_encode_other_object(self):
        """Test encoding other objects (should call parent)"""
        from ollama_workbench.workflows.projects import DateTimeEncoder
        
        encoder = DateTimeEncoder()
        
        # Should raise TypeError for unsupported objects
        with pytest.raises(TypeError):
            encoder.default(object())


class TestDataPersistence:
    """Test data persistence functions"""
    
    @patch('projects.os.makedirs')
    @patch('projects.os.path.exists')
    def test_directory_creation(self, mock_exists, mock_makedirs):
        """Test projects directory creation"""
        mock_exists.return_value = False
        
        # Import should trigger directory creation
        import ollama_workbench.workflows.projects as projects

        
        mock_makedirs.assert_called_with('projects')
    
    def test_load_projects_existing_file(self):
        """Test loading projects from existing file"""
        from ollama_workbench.workflows.projects import load_projects
        
        test_projects = ["Project 1", "Project 2", "Project 3"]
        
        with patch('builtins.open', mock_open(read_data=json.dumps(test_projects))):
            result = load_projects()
        
        assert result == test_projects
    
    @patch('projects.os.path.exists')
    def test_load_projects_nonexistent_file(self, mock_exists):
        """Test loading projects when file doesn't exist"""
        from ollama_workbench.workflows.projects import load_projects
        
        mock_exists.return_value = False
        
        with patch('builtins.open', side_effect=FileNotFoundError):
            result = load_projects()
        
        assert result == []
    
    @patch('projects.st')
    def test_load_projects_json_error(self, mock_st):
        """Test loading projects with JSON decode error"""
        from ollama_workbench.workflows.projects import load_projects
        
        mock_st.error = Mock()
        
        with patch('builtins.open', mock_open(read_data="invalid json")):
            result = load_projects()
        
        assert result == []
        mock_st.error.assert_called_once()
    
    def test_save_projects(self):
        """Test saving projects"""
        from ollama_workbench.workflows.projects import save_projects
        
        test_projects = ["Project A", "Project B"]
        
        mock_file = Mock()
        with patch('builtins.open', mock_file):
            with patch('projects.json.dump') as mock_dump:
                save_projects(test_projects)
        
        mock_file.assert_called_once_with('projects/projects.json', 'w')
        mock_dump.assert_called_once_with(test_projects, mock_file().__enter__())
    
    def test_load_tasks_existing_file(self):
        """Test loading tasks from existing file"""
        from ollama_workbench.workflows.projects import load_tasks, Task
        
        task_data = [
            {
                "name": "Task 1",
                "description": "Description 1",
                "deadline": "2023-12-31 23:59:59",
                "priority": "High",
                "completed": False,
                "agent": "Agent 1",
                "result": None,
                "task_id": "task-1"
            }
        ]
        
        with patch('builtins.open', mock_open(read_data=json.dumps(task_data))):
            result = load_tasks("test_project")
        
        assert len(result) == 1
        assert isinstance(result[0], Task)
        assert result[0].name == "Task 1"
        assert result[0].description == "Description 1"
        assert result[0].priority == "High"
        assert result[0].task_id == "task-1"
    
    @patch('projects.st')
    def test_load_tasks_json_error(self, mock_st):
        """Test loading tasks with JSON decode error"""
        from ollama_workbench.workflows.projects import load_tasks
        
        mock_st.error = Mock()
        
        with patch('builtins.open', mock_open(read_data="invalid json")):
            result = load_tasks("test_project")
        
        assert result == []
        mock_st.error.assert_called_once()
    
    def test_save_tasks(self):
        """Test saving tasks"""
        from ollama_workbench.workflows.projects import save_tasks, Task
        
        tasks = [
            Task("Task 1", "Description 1", deadline=pd.Timestamp("2023-12-31")),
            Task("Task 2", "Description 2", deadline=pd.Timestamp("2024-01-01"))
        ]
        
        mock_file = Mock()
        with patch('builtins.open', mock_file):
            with patch('projects.json.dump') as mock_dump:
                save_tasks("test_project", tasks)
        
        mock_file.assert_called_once_with('projects/test_project_tasks.json', 'w')
        mock_dump.assert_called_once()
        
        # Verify tasks were converted to dict format
        call_args = mock_dump.call_args[0]
        assert len(call_args[0]) == 2  # Two tasks
        assert isinstance(call_args[0][0], dict)
    
    def test_save_tasks_filters_invalid_deadlines(self):
        """Test saving tasks filters out tasks with invalid deadlines"""
        from ollama_workbench.workflows.projects import save_tasks, Task
        
        tasks = [
            Task("Valid Task", "Description", deadline=pd.Timestamp("2023-12-31")),
            Task("Invalid Task", "Description", deadline=pd.NaT)  # Invalid deadline
        ]
        
        mock_file = Mock()
        with patch('builtins.open', mock_file):
            with patch('projects.json.dump') as mock_dump:
                save_tasks("test_project", tasks)
        
        # Should only save the task with valid deadline
        call_args = mock_dump.call_args[0]
        assert len(call_args[0]) == 1
        assert call_args[0][0]["name"] == "Valid Task"
    
    def test_load_agents_existing_file(self):
        """Test loading agents from existing file"""
        from ollama_workbench.workflows.projects import load_agents
        
        agent_data = {
            "Agent 1": {
                "model": "gpt-4",
                "agent_type": "Analyst",
                "temperature": 0.7
            }
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(agent_data))):
            result = load_agents("test_project")
        
        assert result == agent_data
    
    @patch('projects.st')
    def test_load_agents_json_error(self, mock_st):
        """Test loading agents with JSON decode error"""
        from ollama_workbench.workflows.projects import load_agents
        
        mock_st.error = Mock()
        
        with patch('builtins.open', mock_open(read_data="invalid json")):
            result = load_agents("test_project")
        
        assert result == {}
        mock_st.error.assert_called_once()
    
    def test_save_agents(self):
        """Test saving agents"""
        from ollama_workbench.workflows.projects import save_agents
        
        agents = {
            "Agent 1": {"model": "llama3", "temperature": 0.5},
            "Agent 2": {"model": "gpt-4", "temperature": 0.8}
        }
        
        mock_file = Mock()
        with patch('builtins.open', mock_file):
            with patch('projects.json.dump') as mock_dump:
                save_agents("test_project", agents)
        
        mock_file.assert_called_once_with('projects/test_project_agents.json', 'w')
        mock_dump.assert_called_once_with(agents, mock_file().__enter__())


class TestCorpusManagement:
    """Test corpus management functions"""
    
    @patch('projects.os.makedirs')
    @patch('projects.os.path.exists')
    @patch('projects.os.listdir')
    @patch('projects.os.path.isdir')
    def test_get_corpus_options(self, mock_isdir, mock_listdir, mock_exists, mock_makedirs):
        """Test getting corpus options"""
        from ollama_workbench.workflows.projects import get_corpus_options
        
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["corpus1", "corpus2", "file.txt"]
        mock_isdir.side_effect = lambda path: not path.endswith(".txt")
        
        result = get_corpus_options()
        
        assert result == ["None", "corpus1", "corpus2"]
        mock_exists.assert_called_with("corpus")
    
    @patch('projects.os.makedirs')
    @patch('projects.os.path.exists')
    def test_get_corpus_options_no_directory(self, mock_exists, mock_makedirs):
        """Test getting corpus options when directory doesn't exist"""
        from ollama_workbench.workflows.projects import get_corpus_options
        
        mock_exists.return_value = False
        
        result = get_corpus_options()
        
        mock_makedirs.assert_called_with("corpus")
        assert result == ["None"]
    
    def test_get_corpus_context_from_db(self):
        """Test getting corpus context from database"""
        from ollama_workbench.workflows.projects import get_corpus_context_from_db
        
        result = get_corpus_context_from_db("corpus", "test_corpus", "test input")
        
        # Currently returns a placeholder
        assert result == "Sample context from the corpus database."


class TestAIAgentIntegration:
    """Test AI agent integration"""
    
    @patch('projects.load_api_keys')
    @patch('projects.call_openai_api')
    @patch('projects.get_agent_prompt')
    @patch('projects.get_metacognitive_prompt')
    @patch('projects.get_voice_prompt')
    def test_ai_agent_openai(self, mock_voice, mock_metacog, mock_agent, mock_openai, mock_keys):
        """Test AI agent with OpenAI model"""
        from ollama_workbench.workflows.projects import ai_agent, OPENAI_MODELS
        
        # Setup mocks
        mock_keys.return_value = {"openai_api_key": "test_key"}
        mock_openai.return_value = "OpenAI response"
        mock_agent.return_value = {"Analyst": "You are an analyst"}
        mock_metacog.return_value = {"Logical": "Think logically"}
        mock_voice.return_value = {"Formal": "Speak formally"}
        
        with patch('projects.OPENAI_MODELS', ["gpt-4"]):
            result = ai_agent(
                user_input="Test input",
                model="gpt-4",
                agent_type="Analyst",
                metacognitive_type="Logical",
                voice_type="Formal",
                corpus="None",
                temperature=0.7,
                max_tokens=1000
            )
        
        assert result == "OpenAI response"
        mock_openai.assert_called_once()
        
        # Verify the prompt was constructed with all components
        call_args = mock_openai.call_args[0]
        prompt_content = call_args[1][0]["content"]
        assert "You are an analyst" in prompt_content
        assert "Think logically" in prompt_content
        assert "Speak formally" in prompt_content
        assert "Test input" in prompt_content
    
    @patch('projects.load_api_keys')
    @patch('projects.call_groq_api')
    @patch('projects.get_agent_prompt')
    def test_ai_agent_groq(self, mock_agent, mock_groq, mock_keys):
        """Test AI agent with Groq model"""
        from ollama_workbench.workflows.projects import ai_agent, GROQ_MODELS
        
        # Setup mocks
        mock_keys.return_value = {"groq_api_key": "test_key"}
        mock_groq.return_value = "Groq response"
        mock_agent.return_value = {"Creative": "Be creative"}
        
        with patch('projects.GROQ_MODELS', ["mixtral-8x7b"]):
            result = ai_agent(
                user_input="Creative task",
                model="mixtral-8x7b",
                agent_type="Creative",
                metacognitive_type="None",
                voice_type="None",
                corpus="None",
                temperature=0.9,
                max_tokens=2000
            )
        
        assert result == "Groq response"
        mock_groq.assert_called_once()
    
    @patch('projects.load_api_keys')
    @patch('projects.call_ollama_endpoint')
    @patch('projects.get_agent_prompt')
    def test_ai_agent_ollama(self, mock_agent, mock_ollama, mock_keys):
        """Test AI agent with Ollama model"""
        from ollama_workbench.workflows.projects import ai_agent
        
        # Setup mocks
        mock_keys.return_value = {}
        mock_ollama.return_value = ("Ollama response", None, None, None)
        mock_agent.return_value = {"Researcher": "Research thoroughly"}
        
        with patch('projects.OPENAI_MODELS', []):
            with patch('projects.GROQ_MODELS', []):
                result = ai_agent(
                    user_input="Research task",
                    model="llama3",
                    agent_type="Researcher",
                    metacognitive_type="None",
                    voice_type="None",
                    corpus="None",
                    temperature=0.3,
                    max_tokens=1500
                )
        
        assert result == "Ollama response"
        mock_ollama.assert_called_once()
    
    @patch('projects.load_api_keys')
    @patch('projects.call_ollama_endpoint')
    @patch('projects.get_corpus_context_from_db')
    def test_ai_agent_with_corpus(self, mock_corpus, mock_ollama, mock_keys):
        """Test AI agent with corpus context"""
        from ollama_workbench.workflows.projects import ai_agent
        
        # Setup mocks
        mock_keys.return_value = {}
        mock_ollama.return_value = ("Response with context", None, None, None)
        mock_corpus.return_value = "Relevant corpus context"
        
        with patch('projects.OPENAI_MODELS', []):
            with patch('projects.GROQ_MODELS', []):
                result = ai_agent(
                    user_input="Task with corpus",
                    model="llama3",
                    agent_type="None",
                    metacognitive_type="None",
                    voice_type="None",
                    corpus="TechCorpus",
                    temperature=0.5,
                    max_tokens=1000
                )
        
        assert result == "Response with context"
        mock_corpus.assert_called_with("corpus", "TechCorpus", "Task with corpus")
        
        # Verify corpus context was included in prompt
        call_args = mock_ollama.call_args[1]
        assert "Relevant corpus context" in call_args["prompt"]
    
    @patch('projects.load_api_keys')
    @patch('projects.call_ollama_endpoint')
    def test_ai_agent_with_previous_responses(self, mock_ollama, mock_keys):
        """Test AI agent with previous responses"""
        from ollama_workbench.workflows.projects import ai_agent
        
        # Setup mocks
        mock_keys.return_value = {}
        mock_ollama.return_value = ("Final response", None, None, None)
        
        previous_responses = ["First response", "Second response"]
        
        with patch('projects.OPENAI_MODELS', []):
            with patch('projects.GROQ_MODELS', []):
                result = ai_agent(
                    user_input="Continue task",
                    model="llama3",
                    agent_type="None",
                    metacognitive_type="None",
                    voice_type="None",
                    corpus="None",
                    temperature=0.7,
                    max_tokens=1000,
                    previous_responses=previous_responses
                )
        
        assert result == "Final response"
        
        # Verify previous responses were included in prompt
        call_args = mock_ollama.call_args[1]
        assert "Response 1: First response" in call_args["prompt"]
        assert "Response 2: Second response" in call_args["prompt"]


class TestAgentDefinition:
    """Test agent definition functionality"""
    
    @patch('projects.st')
    @patch('projects.get_all_models')
    @patch('projects.get_agent_prompt')
    @patch('projects.get_metacognitive_prompt')
    @patch('projects.get_voice_prompt')
    @patch('projects.get_corpus_options')
    def test_define_agent_block_new_agent(self, mock_corpus, mock_voice, mock_metacog, 
                                          mock_agent, mock_models, mock_st):
        """Test defining new agent block"""
        from ollama_workbench.workflows.projects import define_agent_block
        
        # Setup mocks
        mock_models.return_value = ["llama3", "gpt-4", "mixtral"]
        mock_agent.return_value = {"Analyst": "prompt", "Creative": "prompt"}
        mock_metacog.return_value = {"Logical": "prompt", "Intuitive": "prompt"}
        mock_voice.return_value = {"Formal": "prompt", "Casual": "prompt"}
        mock_corpus.return_value = ["None", "Tech", "Business"]
        
        # Mock Streamlit components
        mock_st.selectbox.side_effect = ["gpt-4", "Analyst", "Logical", "Formal", "Tech"]
        mock_st.slider.side_effect = [0.8, 2000]
        
        result = define_agent_block("TestAgent")
        
        # Verify result structure
        assert result["model"] == "gpt-4"
        assert result["agent_type"] == "Analyst"
        assert result["metacognitive_type"] == "Logical"
        assert result["voice_type"] == "Formal"
        assert result["corpus"] == "Tech"
        assert result["temperature"] == 0.8
        assert result["max_tokens"] == 2000
        
        # Verify Streamlit components were called
        assert mock_st.selectbox.call_count == 5
        assert mock_st.slider.call_count == 2
    
    @patch('projects.st')
    @patch('projects.get_all_models')
    @patch('projects.get_agent_prompt')
    @patch('projects.get_metacognitive_prompt')
    @patch('projects.get_voice_prompt')
    @patch('projects.get_corpus_options')
    def test_define_agent_block_existing_agent(self, mock_corpus, mock_voice, mock_metacog, 
                                               mock_agent, mock_models, mock_st):
        """Test defining agent block with existing data"""
        from ollama_workbench.workflows.projects import define_agent_block
        
        # Setup mocks
        mock_models.return_value = ["llama3", "gpt-4", "mixtral"]
        mock_agent.return_value = {"Analyst": "prompt", "Creative": "prompt"}
        mock_metacog.return_value = {"Logical": "prompt", "Intuitive": "prompt"}
        mock_voice.return_value = {"Formal": "prompt", "Casual": "prompt"}
        mock_corpus.return_value = ["None", "Tech", "Business"]
        
        # Mock Streamlit components to return existing values
        mock_st.selectbox.side_effect = ["mixtral", "Creative", "Intuitive", "Casual", "Business"]
        mock_st.slider.side_effect = [0.5, 3000]
        
        # Existing agent data
        existing_data = {
            "model": "mixtral",
            "agent_type": "Creative",
            "metacognitive_type": "Intuitive",
            "voice_type": "Casual",
            "corpus": "Business",
            "temperature": 0.5,
            "max_tokens": 3000
        }
        
        result = define_agent_block("ExistingAgent", existing_data)
        
        # Verify result matches existing data
        assert result["model"] == "mixtral"
        assert result["agent_type"] == "Creative"
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 3000


class TestProjectManagerAgent:
    """Test ProjectManagerAgent class"""
    
    def test_project_manager_agent_init(self):
        """Test ProjectManagerAgent initialization"""
        from ollama_workbench.workflows.projects import ProjectManagerAgent
        
        agent = ProjectManagerAgent(
            model="gpt-4",
            agent_type="Task Planner",
            temperature=0.7,
            max_tokens=4000
        )
        
        assert agent.model == "gpt-4"
        assert agent.agent_type == "Task Planner"
        assert agent.temperature == 0.7
        assert agent.max_tokens == 4000
    
    @patch('projects.load_api_keys')
    @patch('projects.call_openai_api')
    @patch('projects.st')
    def test_generate_workflow_openai_success(self, mock_st, mock_openai, mock_keys):
        """Test successful workflow generation with OpenAI"""
        from ollama_workbench.workflows.projects import ProjectManagerAgent, OPENAI_MODELS
        
        # Setup mocks
        mock_keys.return_value = {"openai_api_key": "test_key"}
        workflow_response = {
            "tasks": [
                {
                    "name": "Research Task",
                    "description": "Research the topic",
                    "deadline": "2023-12-31 23:59:59",
                    "priority": "High",
                    "completed": False,
                    "agent": "Research Agent",
                    "result": None
                }
            ],
            "agents": {
                "Research Agent": {
                    "model": "gpt-4",
                    "agent_type": "Researcher",
                    "metacognitive_type": "Analytical",
                    "voice_type": "Academic",
                    "corpus": "Scientific",
                    "temperature": 0.3,
                    "max_tokens": 2000
                }
            }
        }
        mock_openai.return_value = json.dumps(workflow_response)
        mock_st.write = Mock()
        
        # Create agent and generate workflow
        with patch('projects.OPENAI_MODELS', ["gpt-4"]):
            agent = ProjectManagerAgent("gpt-4", "Task Planner", 0.7, 4000)
            tasks, agents = agent.generate_workflow("Create a research project")
        
        # Verify results
        assert len(tasks) == 1
        assert tasks[0].name == "Research Task"
        assert tasks[0].description == "Research the topic"
        assert tasks[0].priority == "High"
        assert tasks[0].agent == "Research Agent"
        
        assert len(agents) == 1
        assert "Research Agent" in agents
        assert agents["Research Agent"]["model"] == "gpt-4"
        assert agents["Research Agent"]["agent_type"] == "Researcher"
        
        # Verify API was called
        mock_openai.assert_called_once()
    
    @patch('projects.load_api_keys')
    @patch('projects.call_groq_api')
    @patch('projects.st')
    def test_generate_workflow_groq_success(self, mock_st, mock_groq, mock_keys):
        """Test successful workflow generation with Groq"""
        from ollama_workbench.workflows.projects import ProjectManagerAgent, GROQ_MODELS
        
        # Setup mocks
        mock_keys.return_value = {"groq_api_key": "test_key"}
        workflow_response = {
            "tasks": [
                {
                    "name": "Analysis Task",
                    "description": "Analyze the data",
                    "deadline": "2024-01-15 12:00:00",
                    "priority": "Medium",
                    "completed": False,
                    "agent": "Data Analyst",
                    "result": None
                }
            ],
            "agents": {
                "Data Analyst": {
                    "model": "mixtral-8x7b",
                    "agent_type": "Data Analyst",
                    "metacognitive_type": "Logical",
                    "voice_type": "Technical",
                    "corpus": "Analytics",
                    "temperature": 0.5,
                    "max_tokens": 3000
                }
            }
        }
        mock_groq.return_value = json.dumps(workflow_response)
        mock_st.write = Mock()
        
        # Create agent and generate workflow
        with patch('projects.GROQ_MODELS', ["mixtral-8x7b"]):
            agent = ProjectManagerAgent("mixtral-8x7b", "Task Planner", 0.6, 3500)
            tasks, agents = agent.generate_workflow("Analyze dataset")
        
        # Verify results
        assert len(tasks) == 1
        assert tasks[0].name == "Analysis Task"
        assert tasks[0].agent == "Data Analyst"
        
        assert "Data Analyst" in agents
        assert agents["Data Analyst"]["model"] == "mixtral-8x7b"
        
        # Verify API was called
        mock_groq.assert_called_once()
    
    @patch('projects.load_api_keys')
    @patch('projects.call_ollama_endpoint')
    @patch('projects.st')
    def test_generate_workflow_ollama_success(self, mock_st, mock_ollama, mock_keys):
        """Test successful workflow generation with Ollama"""
        from ollama_workbench.workflows.projects import ProjectManagerAgent
        
        # Setup mocks
        mock_keys.return_value = {}
        workflow_response = {
            "tasks": [
                {
                    "name": "Content Creation",
                    "description": "Create content",
                    "deadline": "2024-02-01 15:30:00",
                    "priority": "Low",
                    "completed": False,
                    "agent": "Content Creator",
                    "result": None
                }
            ],
            "agents": {
                "Content Creator": {
                    "model": "llama3",
                    "agent_type": "Content Creator",
                    "metacognitive_type": "Creative",
                    "voice_type": "Engaging",
                    "corpus": "Marketing",
                    "temperature": 0.8,
                    "max_tokens": 2500
                }
            }
        }
        mock_ollama.return_value = (json.dumps(workflow_response), None, None, None)
        mock_st.write = Mock()
        
        # Create agent and generate workflow
        with patch('projects.OPENAI_MODELS', []):
            with patch('projects.GROQ_MODELS', []):
                agent = ProjectManagerAgent("llama3", "Task Planner", 0.8, 4000)
                tasks, agents = agent.generate_workflow("Create marketing content")
        
        # Verify results
        assert len(tasks) == 1
        assert tasks[0].name == "Content Creation"
        
        assert "Content Creator" in agents
        assert agents["Content Creator"]["model"] == "llama3"
        
        # Verify API was called
        mock_ollama.assert_called_once()
    
    @patch('projects.load_api_keys')
    @patch('projects.call_ollama_endpoint')
    @patch('projects.st')
    def test_generate_workflow_json_error(self, mock_st, mock_ollama, mock_keys):
        """Test workflow generation with JSON error"""
        from ollama_workbench.workflows.projects import ProjectManagerAgent
        
        # Setup mocks
        mock_keys.return_value = {}
        mock_ollama.return_value = ("invalid json response", None, None, None)
        mock_st.write = Mock()
        mock_st.error = Mock()
        
        # Create agent and generate workflow
        with patch('projects.OPENAI_MODELS', []):
            with patch('projects.GROQ_MODELS', []):
                agent = ProjectManagerAgent("llama3", "Task Planner", 0.7, 4000)
                tasks, agents = agent.generate_workflow("Invalid request")
        
        # Verify error handling
        assert tasks is None
        assert agents is None
        mock_st.error.assert_called()


class TestSessionStateManagement:
    """Test session state management"""
    
    @patch('projects.st')
    @patch('projects.load_projects')
    @patch('projects.get_all_models')
    def test_initialize_session_state(self, mock_models, mock_load_projects, mock_st):
        """Test session state initialization"""
        from ollama_workbench.workflows.projects import initialize_session_state
        
        # Setup mocks
        mock_load_projects.return_value = ["Project 1", "Project 2"]
        mock_models.return_value = ["llama3", "gpt-4"]
        mock_st.session_state = {}
        
        initialize_session_state()
        
        # Verify session state was initialized
        assert mock_st.session_state["projects"] == ["Project 1", "Project 2"]
        assert mock_st.session_state["selected_project"] is None
        assert mock_st.session_state["tasks"] == []
        assert mock_st.session_state["agents"] == {}
        assert mock_st.session_state["generated_tasks"] == []
        assert mock_st.session_state["generated_agents"] == {}
        assert "project_manager_settings" in mock_st.session_state
        
        # Verify project manager settings
        settings = mock_st.session_state["project_manager_settings"]
        assert settings["model"] == "llama3"  # First available model
        assert settings["agent_type"] == "Task Planner"
        assert settings["temperature"] == 0.7
        assert settings["max_tokens"] == 4000
    
    @patch('projects.st')
    @patch('projects.load_projects')
    @patch('projects.get_all_models')
    def test_initialize_session_state_no_models(self, mock_models, mock_load_projects, mock_st):
        """Test session state initialization with no models available"""
        from ollama_workbench.workflows.projects import initialize_session_state
        
        # Setup mocks
        mock_load_projects.return_value = []
        mock_models.return_value = []  # No models available
        mock_st.session_state = {}
        
        initialize_session_state()
        
        # Should fallback to default model
        settings = mock_st.session_state["project_manager_settings"]
        assert settings["model"] == "gpt-3.5-turbo"


class TestUserInputHandling:
    """Test user input handling"""
    
    @patch('projects.st')
    def test_handle_user_input_file_path_success(self, mock_st):
        """Test handling file path input successfully"""
        from ollama_workbench.workflows.projects import handle_user_input
        
        # Setup mocks
        mock_st.text_input.return_value = "/path/to/file.csv"
        mock_st.warning = Mock()
        
        step = {
            "agent": "Test Agent",
            "user_input": {
                "type": "file_path",
                "prompt": "Enter file path:"
            }
        }
        task_data = {}
        
        result = handle_user_input(step, task_data)
        
        assert result is True
        assert task_data["file_path"] == "/path/to/file.csv"
        mock_st.warning.assert_not_called()
    
    @patch('projects.st')
    def test_handle_user_input_file_path_empty(self, mock_st):
        """Test handling empty file path input"""
        from ollama_workbench.workflows.projects import handle_user_input
        
        # Setup mocks
        mock_st.text_input.return_value = ""
        mock_st.warning = Mock()
        
        step = {
            "agent": "Test Agent",
            "user_input": {
                "type": "file_path",
                "prompt": "Enter file path:"
            }
        }
        task_data = {}
        
        result = handle_user_input(step, task_data)
        
        assert result is False
        assert "file_path" not in task_data
        mock_st.warning.assert_called_with("Please provide a file path.")
    
    @patch('projects.st')
    def test_handle_user_input_options_success(self, mock_st):
        """Test handling options input successfully"""
        from ollama_workbench.workflows.projects import handle_user_input
        
        # Setup mocks
        mock_st.selectbox.return_value = "Option A"
        mock_st.warning = Mock()
        
        step = {
            "agent": "Test Agent",
            "user_input": {
                "type": "options",
                "prompt": "Select option:",
                "options": ["Option A", "Option B", "Option C"]
            }
        }
        task_data = {}
        
        result = handle_user_input(step, task_data)
        
        assert result is True
        assert task_data["selected_option"] == "Option A"
        mock_st.warning.assert_not_called()
    
    @patch('projects.st')
    def test_handle_user_input_confirmation_confirmed(self, mock_st):
        """Test handling confirmation input when confirmed"""
        from ollama_workbench.workflows.projects import handle_user_input
        
        # Setup mocks
        mock_st.button.return_value = True
        mock_st.warning = Mock()
        
        step = {
            "agent": "Test Agent",
            "user_input": {
                "type": "confirmation",
                "prompt": "Confirm action"
            }
        }
        task_data = {}
        
        result = handle_user_input(step, task_data)
        
        assert result is True
        mock_st.warning.assert_not_called()
    
    @patch('projects.st')
    def test_handle_user_input_confirmation_not_confirmed(self, mock_st):
        """Test handling confirmation input when not confirmed"""
        from ollama_workbench.workflows.projects import handle_user_input
        
        # Setup mocks
        mock_st.button.return_value = False
        mock_st.warning = Mock()
        
        step = {
            "agent": "Test Agent",
            "user_input": {
                "type": "confirmation",
                "prompt": "Confirm action"
            }
        }
        task_data = {}
        
        result = handle_user_input(step, task_data)
        
        assert result is False
        mock_st.warning.assert_called_with("Task skipped due to unconfirmed user input.")
    
    def test_handle_user_input_no_input_required(self):
        """Test handling step with no user input required"""
        from ollama_workbench.workflows.projects import handle_user_input
        
        step = {
            "agent": "Test Agent"
            # No user_input field
        }
        task_data = {}
        
        result = handle_user_input(step, task_data)
        
        assert result is True


class TestTaskStatusUpdate:
    """Test task status update functionality"""
    
    @patch('projects.st')
    def test_update_task_status_basic(self, mock_st):
        """Test basic task status update"""
        from ollama_workbench.workflows.projects import update_task_status
        
        # Setup mock session state
        mock_st.session_state = {
            "bm_tasks": [
                {"status": "Pending", "result": None},
                {"status": "In Progress", "result": None}
            ]
        }
        
        update_task_status(0, "Completed", "Task completed successfully")
        
        assert mock_st.session_state["bm_tasks"][0]["status"] == "Completed"
        assert mock_st.session_state["bm_tasks"][0]["result"] == "Task completed successfully"
    
    @patch('projects.st')
    def test_update_task_status_no_result(self, mock_st):
        """Test task status update without result"""
        from ollama_workbench.workflows.projects import update_task_status
        
        # Setup mock session state
        mock_st.session_state = {
            "bm_tasks": [
                {"status": "Pending", "result": None}
            ]
        }
        
        update_task_status(0, "In Progress")
        
        assert mock_st.session_state["bm_tasks"][0]["status"] == "In Progress"
        assert mock_st.session_state["bm_tasks"][0]["result"] is None
    
    @patch('projects.st')
    def test_update_task_status_invalid_index(self, mock_st):
        """Test task status update with invalid index"""
        from ollama_workbench.workflows.projects import update_task_status
        
        # Setup mock session state
        mock_st.session_state = {
            "bm_tasks": [
                {"status": "Pending", "result": None}
            ]
        }
        
        # Should not raise error with invalid index
        update_task_status(5, "Completed")
        
        # Original task should be unchanged
        assert mock_st.session_state["bm_tasks"][0]["status"] == "Pending"


class TestUtilityFunctions:
    """Test utility functions"""
    
    @patch('projects.get_available_models')
    def test_get_all_models(self, mock_available):
        """Test getting all available models"""
        from ollama_workbench.workflows.projects import get_all_models, OPENAI_MODELS, GROQ_MODELS
        
        mock_available.return_value = ["llama3", "mistral"]
        
        result = get_all_models()
        
        expected = ["llama3", "mistral"] + OPENAI_MODELS + GROQ_MODELS
        assert result == expected


class TestErrorHandling:
    """Test error handling scenarios"""
    
    @patch('projects.st')
    def test_load_projects_empty_file(self, mock_st):
        """Test loading projects from empty file"""
        from ollama_workbench.workflows.projects import load_projects
        
        mock_st.error = Mock()
        
        with patch('builtins.open', mock_open(read_data="")):
            result = load_projects()
        
        assert result == []
    
    @patch('projects.st')
    def test_load_tasks_empty_file(self, mock_st):
        """Test loading tasks from empty file"""
        from ollama_workbench.workflows.projects import load_tasks
        
        mock_st.error = Mock()
        
        with patch('builtins.open', mock_open(read_data="")):
            result = load_tasks("test_project")
        
        assert result == []
    
    @patch('projects.st')
    def test_load_agents_empty_file(self, mock_st):
        """Test loading agents from empty file"""
        from ollama_workbench.workflows.projects import load_agents
        
        mock_st.error = Mock()
        
        with patch('builtins.open', mock_open(read_data="")):
            result = load_agents("test_project")
        
        assert result == {}


class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    def test_complete_task_lifecycle(self):
        """Test complete task lifecycle"""
        from ollama_workbench.workflows.projects import Task, save_tasks, load_tasks, DateTimeEncoder
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save tasks
            tasks = [
                Task("Task 1", "Description 1", deadline=pd.Timestamp("2023-12-31")),
                Task("Task 2", "Description 2", deadline=pd.Timestamp("2024-01-01"), priority="High")
            ]
            
            with patch('projects.open', mock_open()) as mock_file:
                with patch('projects.json.dump') as mock_dump:
                    save_tasks("test_project", tasks)
                    
                    # Verify save was called correctly
                    assert mock_dump.called
                    saved_data = mock_dump.call_args[0][0]
                    assert len(saved_data) == 2
                    assert saved_data[0]["name"] == "Task 1"
                    assert saved_data[1]["priority"] == "High"
    
    def test_complete_agent_lifecycle(self):
        """Test complete agent lifecycle"""
        from ollama_workbench.workflows.projects import save_agents, load_agents
        
        # Create and save agents
        agents = {
            "Agent 1": {
                "model": "gpt-4",
                "agent_type": "Analyst",
                "temperature": 0.7
            },
            "Agent 2": {
                "model": "llama3",
                "agent_type": "Creative",
                "temperature": 0.9
            }
        }
        
        with patch('projects.open', mock_open()) as mock_file:
            with patch('projects.json.dump') as mock_dump:
                save_agents("test_project", agents)
                
                # Verify save was called correctly
                mock_dump.assert_called_once_with(agents, mock_file().__enter__())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
