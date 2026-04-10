"""
Test suite for brainstorm.py - Brainstorm workflow system
"""

import pytest
import json
import os
import tempfile
import subprocess
from unittest.mock import Mock, patch, MagicMock, call, mock_open
from pathlib import Path
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCustomConversableAgent:
    """Test CustomConversableAgent class"""
    
    @patch('ollama_workbench.workflows.brainstorm.Teachability')
    @patch('ollama_workbench.workflows.brainstorm.ConversableAgent.__init__')
    def test_custom_conversable_agent_init(self, mock_super_init, mock_teachability):
        """Test CustomConversableAgent initialization"""
        from ollama_workbench.workflows.brainstorm import CustomConversableAgent
        
        # Setup mocks
        mock_super_init.return_value = None
        mock_teachability_instance = Mock()
        mock_teachability.return_value = mock_teachability_instance
        
        # Test initialization
        llm_config = {"api_key": "test_key", "model": "gpt-4"}
        agent = CustomConversableAgent(
            name="Test Agent",
            llm_config=llm_config,
            agent_type="Analyst",
            identity="Professional",
            metacognitive_type="Reflective",
            voice_type="Formal",
            corpus="Technical",
            temperature=0.7,
            max_tokens=1000,
            presence_penalty=0.1,
            frequency_penalty=0.2,
            db_path="./test_db"
        )
        
        # Verify initialization
        assert agent.agent_type == "Analyst"
        assert agent.identity == "Professional"
        assert agent.metacognitive_type == "Reflective"
        assert agent.voice_type == "Formal"
        assert agent.corpus == "Technical"
        assert agent.temperature == 0.7
        assert agent.max_tokens == 1000
        assert agent.presence_penalty == 0.1
        assert agent.frequency_penalty == 0.2
        
        # Verify teachability was setup
        mock_teachability.assert_called_once_with(path_to_db_dir="./test_db")
        mock_teachability_instance.add_to_agent.assert_called_once_with(agent)
        
        # Verify API key was set in environment
        assert os.environ.get("OPENAI_API_KEY") == "test_key"
    
    @patch('ollama_workbench.workflows.brainstorm.Teachability')
    @patch('ollama_workbench.workflows.brainstorm.ConversableAgent.__init__')
    @patch('ollama_workbench.workflows.brainstorm.ConversableAgent.generate_reply')
    def test_generate_reply(self, mock_super_reply, mock_super_init, mock_teachability):
        """Test CustomConversableAgent generate_reply method"""
        from ollama_workbench.workflows.brainstorm import CustomConversableAgent
        
        # Setup mocks
        mock_super_init.return_value = None
        mock_teachability.return_value = Mock()
        mock_super_reply.return_value = "Generated response"
        
        # Create agent
        agent = CustomConversableAgent(
            name="Test Agent",
            llm_config={},
            agent_type="Creative",
            identity="Artist",
            metacognitive_type="Intuitive",
            voice_type="Casual",
            corpus="Creative",
            temperature=0.8,
            max_tokens=500,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            db_path="./test_db"
        )
        
        # Test generate_reply
        messages = [
            {"role": "user", "name": "User", "content": "Hello"},
            {"role": "assistant", "name": "Assistant", "content": "Hi there!"}
        ]
        sender = Mock()
        config = {}
        
        result = agent.generate_reply(messages, sender, config)
        
        # Verify the response
        assert result == "Generated response"
        
        # Verify the prompt was constructed correctly
        mock_super_reply.assert_called_once()
        call_args = mock_super_reply.call_args
        assert len(call_args[1]["messages"]) == 1
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "Test Agent" in prompt_content
        assert "Creative" in prompt_content
        assert "Artist" in prompt_content
        assert "user (User): Hello" in prompt_content
        assert "assistant (Assistant): Hi there!" in prompt_content


class TestAgentCreation:
    """Test agent creation functionality"""
    
    @patch('ollama_workbench.workflows.brainstorm.CustomConversableAgent')
    @patch('ollama_workbench.workflows.brainstorm.load_api_keys')
    def test_create_agent_openai(self, mock_load_keys, mock_agent_class):
        """Test creating agent with OpenAI model"""
        from ollama_workbench.workflows.brainstorm import create_agent, OPENAI_MODELS
        
        # Setup mocks
        mock_load_keys.return_value = {"openai_api_key": "test_openai_key"}
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        # Test settings for OpenAI model
        settings = {
            "name": "GPT Agent",
            "emoji": "🤖",
            "model": "gpt-4",
            "agent_type": "Analyst",
            "identity": "Professional",
            "metacognitive_type": "Logical",
            "voice_type": "Formal",
            "corpus": "Business",
            "temperature": 0.5,
            "max_tokens": 2000,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1,
            "db_path": "./gpt_db"
        }
        
        with patch('ollama_workbench.workflows.brainstorm.OPENAI_MODELS', ["gpt-4", "gpt-3.5-turbo"]):
            result = create_agent(settings)
        
        # Verify agent creation
        assert result == mock_agent
        mock_agent_class.assert_called_once()
        
        # Verify OpenAI configuration
        call_args = mock_agent_class.call_args
        assert call_args[1]["name"] == "🤖 GPT Agent"
        llm_config = call_args[1]["llm_config"]
        assert llm_config["api_key"] == "test_openai_key"
        assert llm_config["model"] == "gpt-4"
        assert llm_config["request_timeout"] == 120
    
    @patch('ollama_workbench.workflows.brainstorm.CustomConversableAgent')
    @patch('ollama_workbench.workflows.brainstorm.load_api_keys')
    def test_create_agent_groq(self, mock_load_keys, mock_agent_class):
        """Test creating agent with Groq model"""
        from ollama_workbench.workflows.brainstorm import create_agent, GROQ_MODELS
        
        # Setup mocks
        mock_load_keys.return_value = {"groq_api_key": "test_groq_key"}
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        # Test settings for Groq model
        settings = {
            "name": "Groq Agent",
            "emoji": "⚡",
            "model": "mixtral-8x7b-32768",
            "agent_type": "Creative",
            "identity": "Innovator",
            "metacognitive_type": "Intuitive",
            "voice_type": "Casual",
            "corpus": "General",
            "temperature": 0.9,
            "max_tokens": 1500,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.0,
            "db_path": "./groq_db"
        }
        
        with patch('ollama_workbench.workflows.brainstorm.GROQ_MODELS', ["mixtral-8x7b-32768", "llama2-70b-4096"]):
            result = create_agent(settings)
        
        # Verify agent creation
        assert result == mock_agent
        mock_agent_class.assert_called_once()
        
        # Verify Groq configuration
        call_args = mock_agent_class.call_args
        llm_config = call_args[1]["llm_config"]
        assert llm_config["api_key"] == "test_groq_key"
        assert llm_config["model"] == "mixtral-8x7b-32768"
    
    @patch('ollama_workbench.workflows.brainstorm.CustomConversableAgent')
    @patch('ollama_workbench.workflows.brainstorm.load_api_keys')
    def test_create_agent_ollama(self, mock_load_keys, mock_agent_class):
        """Test creating agent with Ollama model"""
        from ollama_workbench.workflows.brainstorm import create_agent
        
        # Setup mocks
        mock_load_keys.return_value = {}
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        # Test settings for Ollama model
        settings = {
            "name": "Llama Agent",
            "emoji": "🦙",
            "model": "llama3",
            "agent_type": "Researcher",
            "identity": "Scholar",
            "metacognitive_type": "Analytical",
            "voice_type": "Academic",
            "corpus": "Scientific",
            "temperature": 0.3,
            "max_tokens": 3000,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.5,
            "db_path": "./llama_db"
        }
        
        with patch('ollama_workbench.workflows.brainstorm.OPENAI_MODELS', []):
            with patch('ollama_workbench.workflows.brainstorm.GROQ_MODELS', []):
                result = create_agent(settings)
        
        # Verify agent creation
        assert result == mock_agent
        mock_agent_class.assert_called_once()
        
        # Verify Ollama configuration
        call_args = mock_agent_class.call_args
        llm_config = call_args[1]["llm_config"]
        assert llm_config["api_base"] == "http://localhost:11434/v1"
        assert llm_config["api_type"] == "open_ai"
        assert llm_config["model"] == "llama3"


class TestSettingsManagement:
    """Test settings management functions"""
    
    def test_load_agent_settings_existing_file(self):
        """Test loading agent settings from existing file"""
        from ollama_workbench.workflows.brainstorm import load_agent_settings
        
        test_settings = {
            "agents": [
                {
                    "name": "Test Agent",
                    "model": "llama3",
                    "temperature": 0.7
                }
            ]
        }
        
        with patch('ollama_workbench.workflows.brainstorm.os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(test_settings))):
                result = load_agent_settings()
        
        assert result == test_settings
    
    def test_load_agent_settings_nonexistent_file(self):
        """Test loading agent settings when file doesn't exist"""
        from ollama_workbench.workflows.brainstorm import load_agent_settings
        
        with patch('ollama_workbench.workflows.brainstorm.os.path.exists', return_value=False):
            result = load_agent_settings()
        
        assert result == {"agents": []}
    
    def test_save_agent_settings(self):
        """Test saving agent settings"""
        from ollama_workbench.workflows.brainstorm import save_agent_settings
        
        test_settings = {
            "agents": [
                {
                    "name": "New Agent",
                    "model": "gpt-4",
                    "temperature": 0.5
                }
            ]
        }
        
        mock_file = mock_open()
        with patch('builtins.open', mock_file):
            with patch('ollama_workbench.workflows.brainstorm.json.dump') as mock_dump:
                save_agent_settings(test_settings)
        
        mock_file.assert_called_once_with("brainstorm_agents_settings.json", 'w')
        mock_dump.assert_called_once_with(test_settings, mock_file().__enter__(), indent=2)


class TestWorkflowManagement:
    """Test workflow management functions"""
    
    @patch('ollama_workbench.workflows.brainstorm.os.makedirs')
    @patch('ollama_workbench.workflows.brainstorm.os.path.exists')
    @patch('ollama_workbench.workflows.brainstorm.os.listdir')
    def test_get_available_workflows(self, mock_listdir, mock_exists, mock_makedirs):
        """Test getting available workflows"""
        from ollama_workbench.workflows.brainstorm import get_available_workflows
        
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["workflow1.json", "workflow2.json", "other_file.txt"]
        
        result = get_available_workflows()
        
        assert result == ["workflow1", "workflow2"]
        mock_exists.assert_called_once()
        mock_makedirs.assert_not_called()
    
    @patch('ollama_workbench.workflows.brainstorm.os.makedirs')
    @patch('ollama_workbench.workflows.brainstorm.os.path.exists')
    @patch('ollama_workbench.workflows.brainstorm.os.listdir')
    def test_get_available_workflows_no_directory(self, mock_listdir, mock_exists, mock_makedirs):
        """Test getting available workflows when directory doesn't exist"""
        from ollama_workbench.workflows.brainstorm import get_available_workflows
        
        # Setup mocks
        mock_exists.return_value = False
        mock_listdir.return_value = []
        
        result = get_available_workflows()
        
        assert result == []
        mock_makedirs.assert_called_once_with("brainstorm_workflows")
    
    @patch('ollama_workbench.workflows.brainstorm.st')
    @patch('ollama_workbench.workflows.brainstorm.os.makedirs')
    @patch('ollama_workbench.workflows.brainstorm.os.path.exists')
    def test_save_workflow(self, mock_exists, mock_makedirs, mock_st):
        """Test saving workflow"""
        from ollama_workbench.workflows.brainstorm import save_workflow
        
        # Setup mocks
        mock_exists.return_value = False
        mock_st.success = Mock()
        
        agent_sequence = ["Agent 1", "Agent 2", "Agent 3"]
        
        mock_file = mock_open()
        with patch('builtins.open', mock_file):
            with patch('ollama_workbench.workflows.brainstorm.json.dump') as mock_dump:
                save_workflow("test_workflow", agent_sequence)
        
        # Verify directory creation and file saving
        mock_makedirs.assert_called_once_with("brainstorm_workflows")
        mock_file.assert_called_once_with("brainstorm_workflows/test_workflow.json", 'w')
        mock_dump.assert_called_once_with(agent_sequence, mock_file().__enter__(), indent=2)
        mock_st.success.assert_called_once_with("Workflow 'test_workflow' saved successfully!")
    
    @patch('ollama_workbench.workflows.brainstorm.os.path.exists')
    def test_load_workflow_existing(self, mock_exists):
        """Test loading existing workflow"""
        from ollama_workbench.workflows.brainstorm import load_workflow
        
        # Setup mocks
        mock_exists.return_value = True
        test_sequence = ["Agent A", "Agent B"]
        
        with patch('builtins.open', mock_open(read_data=json.dumps(test_sequence))):
            result = load_workflow("existing_workflow")
        
        assert result == test_sequence
    
    @patch('ollama_workbench.workflows.brainstorm.os.path.exists')
    def test_load_workflow_nonexistent(self, mock_exists):
        """Test loading non-existent workflow"""
        from ollama_workbench.workflows.brainstorm import load_workflow
        
        mock_exists.return_value = False
        
        result = load_workflow("nonexistent_workflow")
        
        assert result is None


class TestAgentEditing:
    """Test agent editing functionality"""
    
    @patch('ollama_workbench.workflows.brainstorm.st')
    @patch('ollama_workbench.workflows.brainstorm.get_all_models')
    @patch('ollama_workbench.workflows.brainstorm.get_voice_prompt')
    @patch('ollama_workbench.workflows.brainstorm.get_agent_prompt')
    @patch('ollama_workbench.workflows.brainstorm.get_identity_prompt')
    @patch('ollama_workbench.workflows.brainstorm.get_metacognitive_prompt')
    @patch('ollama_workbench.workflows.brainstorm.get_corpus_options')
    def test_edit_agent_settings(self, mock_corpus, mock_metacog, mock_identity, 
                                mock_agent, mock_voice, mock_models, mock_st):
        """Test agent settings editing UI"""
        from ollama_workbench.workflows.brainstorm import edit_agent_settings
        
        # Setup mocks
        mock_models.return_value = ["llama3", "gpt-4", "mixtral-8x7b"]
        mock_voice.return_value = {"Formal": "prompt1", "Casual": "prompt2"}
        mock_agent.return_value = {"Analyst": "prompt1", "Creative": "prompt2"}
        mock_identity.return_value = {"Professional": "prompt1", "Artist": "prompt2"}
        mock_metacog.return_value = {"Logical": "prompt1", "Intuitive": "prompt2"}
        mock_corpus.return_value = ["Tech", "Business"]
        
        # Mock Streamlit components
        mock_st.subheader = Mock()
        mock_st.columns.return_value = [Mock(), Mock(), Mock()]
        mock_st.text_input.return_value = "Updated Agent"
        mock_st.selectbox.side_effect = ["🐱", "gpt-4", "Formal", "Analyst", "Professional", "Logical", "Tech"]
        mock_st.slider.side_effect = [0.8, 2000, 0.2, 0.1]
        
        # Test agent settings
        agent_settings = {
            "name": "Test Agent",
            "emoji": "🐶",
            "model": "llama3",
            "voice_type": "Casual",
            "agent_type": "Creative",
            "identity": "Artist",
            "metacognitive_type": "Intuitive",
            "corpus": "Business",
            "temperature": 0.7,
            "max_tokens": 1000,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "db_path": "./test_db"
        }
        
        result = edit_agent_settings(agent_settings)
        
        # Verify UI components were called
        mock_st.subheader.assert_called_once()
        mock_st.columns.assert_called_once_with(3)
        assert mock_st.text_input.call_count >= 1
        assert mock_st.selectbox.call_count >= 6
        assert mock_st.slider.call_count >= 4
        
        # Verify returned settings
        assert result["name"] == "Updated Agent"
        assert result["emoji"] == "🐱"
        assert result["model"] == "gpt-4"


class TestAgentManagement:
    """Test agent management interface"""
    
    @patch('ollama_workbench.workflows.brainstorm.st')
    @patch('ollama_workbench.workflows.brainstorm.load_agent_settings')
    @patch('ollama_workbench.workflows.brainstorm.save_agent_settings')
    @patch('ollama_workbench.workflows.brainstorm.edit_agent_settings')
    @patch('ollama_workbench.workflows.brainstorm.get_all_models')
    def test_manage_agents_display(self, mock_models, mock_edit, mock_save, mock_load, mock_st):
        """Test agent management display"""
        from ollama_workbench.workflows.brainstorm import manage_agents
        
        # Setup mocks
        mock_load.return_value = {
            "agents": [
                {"name": "Agent 1", "emoji": "🐶", "model": "llama3"},
                {"name": "Agent 2", "emoji": "🐱", "model": "gpt-4"}
            ]
        }
        mock_models.return_value = ["llama3", "gpt-4"]
        mock_edit.side_effect = lambda x: x  # Return unchanged
        mock_st.subheader = Mock()
        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock()
        mock_st.button.return_value = False
        mock_st.rerun = Mock()
        
        manage_agents()
        
        # Verify components were called
        mock_st.subheader.assert_called_with("Manage Agents")
        mock_load.assert_called_once()
        mock_save.assert_called()
        assert mock_st.expander.call_count >= 2  # One per agent plus "Add New Agent"
    
    @patch('ollama_workbench.workflows.brainstorm.st')
    @patch('ollama_workbench.workflows.brainstorm.load_agent_settings')
    @patch('ollama_workbench.workflows.brainstorm.save_agent_settings')
    @patch('ollama_workbench.workflows.brainstorm.edit_agent_settings')
    @patch('ollama_workbench.workflows.brainstorm.get_all_models')
    def test_manage_agents_remove(self, mock_models, mock_edit, mock_save, mock_load, mock_st):
        """Test removing agent"""
        from ollama_workbench.workflows.brainstorm import manage_agents
        
        # Setup mocks
        initial_settings = {
            "agents": [
                {"name": "Agent 1", "emoji": "🐶", "model": "llama3"},
                {"name": "Agent 2", "emoji": "🐱", "model": "gpt-4"}
            ]
        }
        mock_load.return_value = initial_settings
        mock_models.return_value = ["llama3", "gpt-4"]
        mock_edit.side_effect = lambda x: x
        mock_st.subheader = Mock()
        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock()
        mock_st.button.side_effect = [True, False, False]  # Remove first agent
        mock_st.success = Mock()
        mock_st.rerun = Mock()
        
        manage_agents()
        
        # Verify agent was removed
        mock_save.assert_called()
        call_args = mock_save.call_args[0][0]
        assert len(call_args["agents"]) == 1  # One agent removed
        mock_st.success.assert_called_with("Agent Agent 1 removed.")
        mock_st.rerun.assert_called_once()
    
    @patch('ollama_workbench.workflows.brainstorm.st')
    @patch('ollama_workbench.workflows.brainstorm.load_agent_settings')
    @patch('ollama_workbench.workflows.brainstorm.save_agent_settings')
    @patch('ollama_workbench.workflows.brainstorm.edit_agent_settings')
    @patch('ollama_workbench.workflows.brainstorm.get_all_models')
    def test_manage_agents_add_new(self, mock_models, mock_edit, mock_save, mock_load, mock_st):
        """Test adding new agent"""
        from ollama_workbench.workflows.brainstorm import manage_agents
        
        # Setup mocks
        initial_settings = {"agents": []}
        mock_load.return_value = initial_settings
        mock_models.return_value = ["llama3", "gpt-4"]
        
        # Mock edit_agent_settings to return a new agent
        def edit_side_effect(agent):
            if agent["name"] == "":
                agent["name"] = "New Agent"
            return agent
        
        mock_edit.side_effect = edit_side_effect
        mock_st.subheader = Mock()
        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock()
        mock_st.button.side_effect = [True]  # Add Agent button clicked
        mock_st.success = Mock()
        mock_st.rerun = Mock()
        mock_st.error = Mock()
        
        manage_agents()
        
        # Verify agent was added
        mock_save.assert_called()
        call_args = mock_save.call_args[0][0]
        assert len(call_args["agents"]) == 1
        assert call_args["agents"][0]["name"] == "New Agent"
        mock_st.success.assert_called_with("Agent New Agent added successfully!")
        mock_st.rerun.assert_called_once()


class TestBrainstormSession:
    """Test brainstorm session functionality"""
    
    @patch('ollama_workbench.workflows.brainstorm.st')
    @patch('ollama_workbench.workflows.brainstorm.load_agent_settings')
    @patch('ollama_workbench.workflows.brainstorm.load_api_keys')
    @patch('ollama_workbench.workflows.brainstorm.create_agent')
    @patch('ollama_workbench.workflows.brainstorm.UserProxyAgent')
    @patch('ollama_workbench.workflows.brainstorm.GroupChat')
    @patch('ollama_workbench.workflows.brainstorm.GroupChatManager')
    @patch('ollama_workbench.workflows.brainstorm.get_available_workflows')
    def test_brainstorm_session_initialization(self, mock_workflows, mock_manager, 
                                             mock_group_chat, mock_user_proxy, 
                                             mock_create_agent, mock_load_keys, 
                                             mock_load_settings, mock_st):
        """Test brainstorm session initialization"""
        from ollama_workbench.workflows.brainstorm import brainstorm_session
        
        # Setup mocks
        mock_load_settings.return_value = {
            "agents": [{"name": "Test Agent", "model": "llama3"}]
        }
        mock_load_keys.return_value = {"openai_api_key": "test_key"}
        mock_create_agent.return_value = Mock()
        mock_user_proxy.return_value = Mock()
        mock_group_chat.return_value = Mock()
        mock_manager.return_value = Mock()
        mock_workflows.return_value = ["workflow1", "workflow2"]
        
        # Mock Streamlit components
        mock_st.session_state = {}
        mock_st.columns.return_value = [Mock(), Mock(), Mock()]
        mock_st.text_input.return_value = ""
        mock_st.selectbox.return_value = ""
        mock_st.subheader = Mock()
        mock_st.write = Mock()
        mock_st.number_input.return_value = 0
        mock_st.button.side_effect = [False, False]
        mock_st.success = Mock()
        mock_st.error = Mock()
        
        brainstorm_session(use_docker=False)
        
        # Verify initialization
        mock_load_settings.assert_called_once()
        mock_load_keys.assert_called_once()
        mock_create_agent.assert_called_once()
        mock_workflows.assert_called_once()
    
    @patch('ollama_workbench.workflows.brainstorm.st')
    @patch('ollama_workbench.workflows.brainstorm.load_agent_settings')
    @patch('ollama_workbench.workflows.brainstorm.load_api_keys')
    @patch('ollama_workbench.workflows.brainstorm.create_agent')
    @patch('ollama_workbench.workflows.brainstorm.load_workflow')
    @patch('ollama_workbench.workflows.brainstorm.get_available_workflows')
    def test_workflow_loading(self, mock_workflows, mock_load_workflow, 
                             mock_create_agent, mock_load_keys, 
                             mock_load_settings, mock_st):
        """Test workflow loading functionality"""
        from ollama_workbench.workflows.brainstorm import brainstorm_session
        
        # Setup mocks
        mock_load_settings.return_value = {"agents": []}
        mock_load_keys.return_value = {}
        mock_create_agent.return_value = Mock()
        mock_workflows.return_value = ["test_workflow"]
        mock_load_workflow.return_value = ["Agent 1", "Agent 2"]
        
        # Mock Streamlit components
        mock_st.session_state = {}
        mock_st.columns.return_value = [Mock(), Mock(), Mock()]
        mock_st.text_input.return_value = ""
        mock_st.selectbox.return_value = "test_workflow"
        mock_st.subheader = Mock()
        mock_st.write = Mock()
        mock_st.number_input.return_value = 0
        mock_st.button.side_effect = [False, False]
        mock_st.success = Mock()
        mock_st.rerun = Mock()
        
        brainstorm_session(use_docker=False)
        
        # Verify workflow loading
        mock_load_workflow.assert_called_with("test_workflow")
        mock_st.success.assert_called_with("Workflow 'test_workflow' loaded successfully!")
    
    @patch('ollama_workbench.workflows.brainstorm.st')
    @patch('ollama_workbench.workflows.brainstorm.load_agent_settings')
    @patch('ollama_workbench.workflows.brainstorm.load_api_keys')
    @patch('ollama_workbench.workflows.brainstorm.create_agent')
    @patch('ollama_workbench.workflows.brainstorm.save_workflow')
    @patch('ollama_workbench.workflows.brainstorm.get_available_workflows')
    def test_workflow_saving(self, mock_workflows, mock_save_workflow, 
                            mock_create_agent, mock_load_keys, 
                            mock_load_settings, mock_st):
        """Test workflow saving functionality"""
        from ollama_workbench.workflows.brainstorm import brainstorm_session
        
        # Setup mocks
        mock_load_settings.return_value = {"agents": []}
        mock_load_keys.return_value = {}
        mock_create_agent.return_value = Mock()
        mock_workflows.return_value = []
        
        # Mock Streamlit components
        mock_st.session_state = {"agent_sequence": ["Agent 1", "Agent 2"]}
        mock_st.columns.return_value = [Mock(), Mock(), Mock()]
        mock_st.text_input.return_value = "new_workflow"
        mock_st.selectbox.return_value = ""
        mock_st.subheader = Mock()
        mock_st.write = Mock()
        mock_st.number_input.return_value = 2
        mock_st.button.side_effect = [True, False]  # Save Workflow button clicked
        mock_st.error = Mock()
        
        brainstorm_session(use_docker=False)
        
        # Verify workflow saving
        mock_save_workflow.assert_called_with("new_workflow", ["Agent 1", "Agent 2"])


class TestDockerIntegration:
    """Test Docker integration functionality"""
    
    @patch('ollama_workbench.workflows.brainstorm.subprocess.run')
    def test_is_docker_running_success(self, mock_run):
        """Test Docker running check - success case"""
        from ollama_workbench.workflows.brainstorm import is_docker_running
        
        # Setup mock for successful Docker check
        mock_run.return_value = Mock()
        
        result = is_docker_running()
        
        assert result is True
        mock_run.assert_called_once_with(
            ["docker", "info"], 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    
    @patch('ollama_workbench.workflows.brainstorm.subprocess.run')
    def test_is_docker_running_not_running(self, mock_run):
        """Test Docker running check - Docker not running"""
        from ollama_workbench.workflows.brainstorm import is_docker_running
        
        # Setup mock for Docker not running
        mock_run.side_effect = subprocess.CalledProcessError(1, "docker")
        
        result = is_docker_running()
        
        assert result is False
    
    @patch('ollama_workbench.workflows.brainstorm.subprocess.run')
    def test_is_docker_running_not_installed(self, mock_run):
        """Test Docker running check - Docker not installed"""
        from ollama_workbench.workflows.brainstorm import is_docker_running
        
        # Setup mock for Docker not installed
        mock_run.side_effect = FileNotFoundError()
        
        result = is_docker_running()
        
        assert result is False


class TestMainInterface:
    """Test main brainstorm interface"""
    
    @patch('ollama_workbench.workflows.brainstorm.st')
    @patch('ollama_workbench.workflows.brainstorm.is_docker_running')
    @patch('ollama_workbench.workflows.brainstorm.brainstorm_session')
    @patch('ollama_workbench.workflows.brainstorm.manage_agents')
    def test_brainstorm_interface_docker_enabled(self, mock_manage, mock_session, 
                                                 mock_docker_running, mock_st):
        """Test brainstorm interface with Docker enabled"""
        from ollama_workbench.workflows.brainstorm import brainstorm_interface
        
        # Setup mocks
        mock_docker_running.return_value = True
        mock_st.title = Mock()
        mock_st.checkbox.return_value = True
        mock_st.tabs.return_value = [MagicMock(), MagicMock()]
        mock_st.warning = Mock()
        mock_st.info = Mock()
        mock_st.stop = Mock()
        
        brainstorm_interface()
        
        # Verify interface components
        mock_st.title.assert_called_with("🧠 Brainstorm")
        mock_docker_running.assert_called_once()
        mock_session.assert_called_once_with(True)
        mock_manage.assert_called_once()
        mock_st.warning.assert_not_called()
        mock_st.stop.assert_not_called()
    
    @patch('ollama_workbench.workflows.brainstorm.st')
    @patch('ollama_workbench.workflows.brainstorm.is_docker_running')
    @patch('ollama_workbench.workflows.brainstorm.brainstorm_session')
    @patch('ollama_workbench.workflows.brainstorm.manage_agents')
    def test_brainstorm_interface_docker_disabled(self, mock_manage, mock_session, 
                                                  mock_docker_running, mock_st):
        """Test brainstorm interface with Docker disabled"""
        from ollama_workbench.workflows.brainstorm import brainstorm_interface
        
        # Setup mocks
        mock_docker_running.return_value = False
        mock_st.title = Mock()
        mock_st.checkbox.return_value = False
        mock_st.tabs.return_value = [MagicMock(), MagicMock()]
        
        brainstorm_interface()
        
        # Verify interface components
        mock_st.title.assert_called_with("🧠 Brainstorm")
        mock_docker_running.assert_not_called()
        mock_session.assert_called_once_with(False)
        mock_manage.assert_called_once()
        
        # Verify AUTOGEN_USE_DOCKER environment variable was set
        assert os.environ.get("AUTOGEN_USE_DOCKER") == "0"
    
    @patch('ollama_workbench.workflows.brainstorm.st')
    @patch('ollama_workbench.workflows.brainstorm.is_docker_running')
    def test_brainstorm_interface_docker_not_running(self, mock_docker_running, mock_st):
        """Test brainstorm interface when Docker is not running"""
        from ollama_workbench.workflows.brainstorm import brainstorm_interface
        
        # Setup mocks
        mock_docker_running.return_value = False
        mock_st.title = Mock()
        mock_st.checkbox.return_value = True  # User wants Docker but it's not running
        mock_st.warning = Mock()
        mock_st.info = Mock()
        mock_st.stop = Mock()
        
        brainstorm_interface()
        
        # Verify warning and stop
        mock_st.warning.assert_called_once()
        mock_st.info.assert_called_once()
        mock_st.stop.assert_called_once()


class TestConstants:
    """Test module constants and configurations"""
    
    def test_animal_emojis_list(self):
        """Test that animal emojis list is properly defined"""
        from ollama_workbench.workflows.brainstorm import ANIMAL_EMOJIS
        
        assert isinstance(ANIMAL_EMOJIS, list)
        assert len(ANIMAL_EMOJIS) > 50  # Should have many emojis
        assert "🐶" in ANIMAL_EMOJIS
        assert "🐱" in ANIMAL_EMOJIS
        assert "🦙" in ANIMAL_EMOJIS
    
    def test_settings_file_constants(self):
        """Test that file constants are properly defined"""
        from ollama_workbench.workflows.brainstorm import SETTINGS_FILE, WORKFLOWS_DIR, API_KEYS_FILE
        
        assert SETTINGS_FILE == "brainstorm_agents_settings.json"
        assert WORKFLOWS_DIR == "brainstorm_workflows"
        assert API_KEYS_FILE == "api_keys.json"
    
    def test_environment_variable_setup(self):
        """Test that environment variables are set"""
        import ollama_workbench.workflows.brainstorm as brainstorm

        
        # The module should set TOKENIZERS_PARALLELISM to "false"
        assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"


class TestImportHandling:
    """Test import handling and fallbacks"""
    
    @patch('ollama_workbench.workflows.brainstorm.get_corpus_options')
    def test_corpus_options_import_success(self, mock_get_corpus):
        """Test successful import of get_corpus_options"""
        mock_get_corpus.return_value = ["Tech", "Business", "Science"]
        
        # Re-import to test the function
        from ollama_workbench.workflows.brainstorm import get_corpus_options
        result = get_corpus_options()
        
        assert result == ["Tech", "Business", "Science"]
    
    def test_corpus_options_import_failure(self):
        """Test fallback when get_corpus_options import fails"""
        # This tests the fallback function defined in the try/except block
        with patch.dict('sys.modules', {'files_management': None}):
            # Force a re-import to trigger the ImportError
            import importlib
            import ollama_workbench.workflows.brainstorm as brainstorm

            importlib.reload(brainstorm)
            
            # The fallback function should return an empty list
            result = brainstorm.get_corpus_options()
            assert result == []


class TestErrorHandling:
    """Test error handling scenarios"""
    
    @patch('ollama_workbench.workflows.brainstorm.json.load')
    @patch('ollama_workbench.workflows.brainstorm.os.path.exists')
    def test_load_settings_json_error(self, mock_exists, mock_json_load):
        """Test handling of JSON loading errors"""
        from ollama_workbench.workflows.brainstorm import load_agent_settings
        
        mock_exists.return_value = True
        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        with patch('builtins.open', mock_open()):
            with pytest.raises(json.JSONDecodeError):
                load_agent_settings()
    
    @patch('ollama_workbench.workflows.brainstorm.CustomConversableAgent')
    @patch('ollama_workbench.workflows.brainstorm.load_api_keys')
    def test_create_agent_missing_api_key(self, mock_load_keys, mock_agent_class):
        """Test creating agent with missing API key"""
        from ollama_workbench.workflows.brainstorm import create_agent
        
        # Setup mocks - missing OpenAI key
        mock_load_keys.return_value = {}
        mock_agent_class.return_value = Mock()
        
        settings = {
            "name": "Test Agent",
            "emoji": "🤖",
            "model": "gpt-4",
            "agent_type": "Analyst",
            "identity": "Professional",
            "metacognitive_type": "Logical",
            "voice_type": "Formal",
            "corpus": "Business",
            "temperature": 0.5,
            "max_tokens": 2000,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "db_path": "./test_db"
        }
        
        with patch('ollama_workbench.workflows.brainstorm.OPENAI_MODELS', ["gpt-4"]):
            result = create_agent(settings)
        
        # Should still create agent, but with None API key
        assert result is not None
        call_args = mock_agent_class.call_args
        llm_config = call_args[1]["llm_config"]
        assert llm_config["api_key"] is None


class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    def test_complete_workflow_cycle(self):
        """Test complete workflow save/load cycle"""
        from ollama_workbench.workflows.brainstorm import save_workflow, load_workflow
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('ollama_workbench.workflows.brainstorm.WORKFLOWS_DIR', tmp_dir):
                # Save a workflow
                agent_sequence = ["Agent 1", "Agent 2", "Agent 3"]
                with patch('ollama_workbench.workflows.brainstorm.st') as mock_st:
                    mock_st.success = Mock()
                    save_workflow("test_integration", agent_sequence)
                
                # Load the workflow
                loaded_sequence = load_workflow("test_integration")
                
                # Verify the cycle
                assert loaded_sequence == agent_sequence
    
    def test_agent_settings_cycle(self):
        """Test complete agent settings save/load cycle"""
        from ollama_workbench.workflows.brainstorm import save_agent_settings, load_agent_settings
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with patch('ollama_workbench.workflows.brainstorm.SETTINGS_FILE', tmp_path):
                # Save settings
                test_settings = {
                    "agents": [
                        {
                            "name": "Integration Test Agent",
                            "model": "llama3",
                            "temperature": 0.8,
                            "max_tokens": 1500
                        }
                    ]
                }
                save_agent_settings(test_settings)
                
                # Load settings
                loaded_settings = load_agent_settings()
                
                # Verify the cycle
                assert loaded_settings == test_settings
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
