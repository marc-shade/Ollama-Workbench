"""
Test suite for build_manager.py - Build manager workflow system
"""

import pytest
import json
import os
import tempfile
import queue
import threading
from unittest.mock import Mock, patch, MagicMock, call
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBuildManager:
    """Test BuildManager class"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Create a test message queue
        self.test_queue = queue.Queue()
        self.update_callback = Mock()
    
    @patch('build_manager.os.makedirs')
    @patch('build_manager.os.listdir')
    def test_build_manager_initialization(self, mock_listdir, mock_makedirs):
        """Test BuildManager initialization"""
        from build_manager import BuildManager
        
        # Setup mocks
        mock_listdir.return_value = []
        
        # Test initialization
        manager = BuildManager(self.test_queue, self.update_callback)
        
        assert manager.name == "Build Manager"
        assert manager.message_queue == self.test_queue
        assert manager.update_task_callback == self.update_callback
        assert manager.cancel_flag is False
        assert manager.agents == {}
        assert manager.workflows == {}
    
    @patch('build_manager.os.makedirs')
    @patch('build_manager.os.listdir')
    @patch('build_manager.Agent.from_json')
    def test_load_agents_success(self, mock_from_json, mock_listdir, mock_makedirs):
        """Test successful agent loading"""
        from build_manager import BuildManager
        
        # Setup mocks
        mock_listdir.return_value = ["agent1.json", "agent2.json", "not_json.txt"]
        mock_agent1 = Mock()
        mock_agent1.name = "Agent 1"
        mock_agent2 = Mock()
        mock_agent2.name = "Agent 2"
        mock_from_json.side_effect = [mock_agent1, mock_agent2]
        
        # Test agent loading
        manager = BuildManager(self.test_queue, self.update_callback)
        
        assert len(manager.agents) == 2
        assert "Agent 1" in manager.agents
        assert "Agent 2" in manager.agents
        assert mock_from_json.call_count == 2
    
    @patch('build_manager.os.makedirs')
    @patch('build_manager.os.listdir')
    @patch('build_manager.Agent.from_json')
    def test_load_agents_error(self, mock_from_json, mock_listdir, mock_makedirs):
        """Test agent loading with error"""
        from build_manager import BuildManager
        
        # Setup mocks
        mock_listdir.return_value = ["bad_agent.json"]
        mock_from_json.side_effect = Exception("JSON parse error")
        
        # Test that exception is raised during initialization
        with pytest.raises(Exception):
            BuildManager(self.test_queue, self.update_callback)
    
    @patch('build_manager.os.makedirs')
    @patch('build_manager.os.listdir')
    def test_load_workflows_success(self, mock_listdir, mock_makedirs):
        """Test successful workflow loading"""
        from build_manager import BuildManager
        
        # Create temporary workflow files
        workflow1_data = {"name": "Workflow 1", "steps": []}
        workflow2_data = {"name": "Workflow 2", "steps": []}
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            workflow1_path = os.path.join(tmp_dir, "workflow1.json")
            workflow2_path = os.path.join(tmp_dir, "workflow2.json")
            
            with open(workflow1_path, 'w') as f:
                json.dump(workflow1_data, f)
            with open(workflow2_path, 'w') as f:
                json.dump(workflow2_data, f)
            
            # Mock listdir to return our test files
            def mock_listdir_side_effect(path):
                if "workflows" in path:
                    return ["workflow1.json", "workflow2.json"]
                return []
            
            mock_listdir.side_effect = mock_listdir_side_effect
            
            # Test workflow loading
            manager = BuildManager(self.test_queue, self.update_callback, workflows_dir=tmp_dir)
            
            assert len(manager.workflows) == 2
            assert "Workflow 1" in manager.workflows
            assert "Workflow 2" in manager.workflows
    
    @patch('build_manager.ollama.generate')
    def test_process_request_data_analysis(self, mock_generate):
        """Test processing data analysis request"""
        from build_manager import BuildManager
        
        # Setup mocks
        mock_generate.return_value = {"response": "data_analysis"}
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.workflows = {"Data Analysis Workflow": {"steps": []}}
            manager.cancel_flag = False
            
            with patch.object(manager, 'execute_workflow') as mock_execute:
                manager.process_request("Analyze this dataset")
                mock_execute.assert_called_with("Data Analysis Workflow", "Analyze this dataset")
    
    @patch('build_manager.ollama.generate')
    def test_process_request_code_generation(self, mock_generate):
        """Test processing code generation request"""
        from build_manager import BuildManager
        
        # Setup mocks
        mock_generate.return_value = {"response": "code_generation"}
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.workflows = {"Code Generation Workflow": {"steps": []}}
            manager.cancel_flag = False
            
            with patch.object(manager, 'execute_workflow') as mock_execute:
                manager.process_request("Generate a Python function")
                mock_execute.assert_called_with("Code Generation Workflow", "Generate a Python function")
    
    @patch('build_manager.ollama.generate')
    def test_process_request_unknown_classification(self, mock_generate):
        """Test processing request with unknown classification"""
        from build_manager import BuildManager
        
        # Setup mocks
        mock_generate.return_value = {"response": "unknown_category"}
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.cancel_flag = False
            manager.update_task_callback = Mock()
            
            # Should handle unknown classification gracefully
            manager.process_request("Unclear request")
            manager.update_task_callback.assert_called_with(0, "Failed", "Error: Unable to classify user request")
    
    @patch('build_manager.st')
    def test_execute_workflow_with_user_input(self, mock_st):
        """Test workflow execution with user input"""
        from build_manager import BuildManager
        from ollama_workbench.workflows.agents import Agent
        from ollama_workbench.workflows.projects import Task
        
        # Setup workflow with user input
        workflow = {
            "steps": [
                {
                    "agent": "Test Agent",
                    "task_description": "Process file {file_path}",
                    "inputs": ["file_path"],
                    "user_input": {
                        "type": "file_path",
                        "prompt": "Enter file path:"
                    }
                }
            ]
        }
        
        # Setup mocks
        mock_st.text_input.return_value = "/path/to/file.csv"
        mock_st.session_state = {}
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.workflows = {"Test Workflow": workflow}
            manager.agents = {}
            manager.message_queue = queue.Queue()
            
            manager.execute_workflow("Test Workflow", "Test request")
            
            # Verify task was added to queue
            assert not manager.message_queue.empty()
            task_message = manager.message_queue.get()
            assert task_message["task"].description == "Process file /path/to/file.csv"
    
    @patch('build_manager.st')
    def test_execute_workflow_with_options_input(self, mock_st):
        """Test workflow execution with options input"""
        from build_manager import BuildManager
        
        # Setup workflow with options input
        workflow = {
            "steps": [
                {
                    "agent": "Test Agent",
                    "task_description": "Use option {selected_option}",
                    "inputs": ["selected_option"],
                    "user_input": {
                        "type": "options",
                        "prompt": "Select option:",
                        "options": ["Option A", "Option B"]
                    }
                }
            ]
        }
        
        # Setup mocks
        mock_st.selectbox.return_value = "Option A"
        mock_st.session_state = {}
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.workflows = {"Test Workflow": workflow}
            manager.agents = {}
            manager.message_queue = queue.Queue()
            
            manager.execute_workflow("Test Workflow", "Test request")
            
            # Verify task was added to queue
            assert not manager.message_queue.empty()
            task_message = manager.message_queue.get()
            assert task_message["task"].description == "Use option Option A"
    
    @patch('build_manager.st')
    def test_execute_workflow_confirmation_declined(self, mock_st):
        """Test workflow execution with declined confirmation"""
        from build_manager import BuildManager
        
        # Setup workflow with confirmation
        workflow = {
            "steps": [
                {
                    "agent": "Test Agent",
                    "task_description": "Proceed with task",
                    "inputs": [],
                    "user_input": {
                        "type": "confirmation",
                        "prompt": "Confirm action"
                    }
                }
            ]
        }
        
        # Setup mocks - button returns False (not confirmed)
        mock_st.button.return_value = False
        mock_st.session_state = {}
        mock_st.warning = Mock()
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.workflows = {"Test Workflow": workflow}
            manager.agents = {}
            manager.message_queue = queue.Queue()
            
            manager.execute_workflow("Test Workflow", "Test request")
            
            # Verify warning was shown and no task added
            mock_st.warning.assert_called()
            assert manager.message_queue.empty()
    
    def test_run_processes_all_tasks(self):
        """Test that run processes all tasks in queue"""
        from build_manager import BuildManager
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.message_queue = queue.Queue()
            manager.cancel_flag = False
            
            # Add test tasks to queue
            task1 = {"task": Mock(), "agent": Mock(), "step_index": 0}
            task2 = {"task": Mock(), "agent": Mock(), "step_index": 1}
            manager.message_queue.put(task1)
            manager.message_queue.put(task2)
            
            with patch.object(manager, 'execute_task') as mock_execute:
                with patch('build_manager.st') as mock_st:
                    mock_st.session_state = {}
                    manager.run()
                
                # Verify both tasks were executed
                assert mock_execute.call_count == 2
                mock_execute.assert_has_calls([call(task1), call(task2)])
    
    @patch('build_manager.call_ollama_endpoint')
    def test_execute_task_data_extractor(self, mock_ollama):
        """Test executing Data Extractor task"""
        from build_manager import BuildManager
        from ollama_workbench.workflows.agents import Agent
        from ollama_workbench.workflows.projects import Task
        
        # Setup mocks
        mock_ollama.return_value = ("col1,col2\nval1,val2", None, None, None)
        
        # Create test objects
        agent = Agent("Data Extractor", [], {}, "test-model")
        task = Task("extract_data", "Extract data from CSV")
        task_event = threading.Event()
        task_event.set()
        
        task_message = {
            "agent": agent,
            "task": task,
            "step_index": 0,
            "project_name": "test",
            "task_event": task_event
        }
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.update_task_callback = Mock()
            
            manager.execute_task(task_message)
            
            # Verify task was completed
            assert task.completed is True
            assert isinstance(task.result, list)
            manager.update_task_callback.assert_called_with(0, "Completed", task.result)
    
    @patch('build_manager.call_ollama_endpoint')
    def test_execute_task_data_analyst(self, mock_ollama):
        """Test executing Data Analyst task"""
        from build_manager import BuildManager
        from ollama_workbench.workflows.agents import Agent
        from ollama_workbench.workflows.projects import Task
        
        # Setup mocks
        mock_ollama.return_value = ('{"mean": 5.5, "count": 10}', None, None, None)
        
        # Create test objects
        agent = Agent("Data Analyst", [], {}, "test-model")
        task = Task("analyze_data", "Analyze the data")
        task_event = threading.Event()
        task_event.set()
        
        task_message = {
            "agent": agent,
            "task": task,
            "step_index": 0,
            "project_name": "test",
            "task_event": task_event
        }
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.update_task_callback = Mock()
            
            manager.execute_task(task_message)
            
            # Verify task was completed
            assert task.completed is True
            assert task.result["mean"] == 5.5
            assert task.result["count"] == 10
    
    @patch('build_manager.call_ollama_endpoint')
    def test_execute_task_data_analyst_invalid_json(self, mock_ollama):
        """Test executing Data Analyst task with invalid JSON"""
        from build_manager import BuildManager
        from ollama_workbench.workflows.agents import Agent
        from ollama_workbench.workflows.projects import Task
        
        # Setup mocks
        mock_ollama.return_value = ("invalid json response", None, None, None)
        
        # Create test objects
        agent = Agent("Data Analyst", [], {}, "test-model")
        task = Task("analyze_data", "Analyze the data")
        task_event = threading.Event()
        task_event.set()
        
        task_message = {
            "agent": agent,
            "task": task,
            "step_index": 0,
            "project_name": "test",
            "task_event": task_event
        }
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.update_task_callback = Mock()
            
            manager.execute_task(task_message)
            
            # Verify task failed
            assert task.completed is False
            assert "Error: Invalid JSON response" in task.result
    
    @patch('build_manager.call_ollama_endpoint')
    def test_execute_task_report_writer(self, mock_ollama):
        """Test executing Report Writer task"""
        from build_manager import BuildManager
        from ollama_workbench.workflows.agents import Agent
        from ollama_workbench.workflows.projects import Task
        
        # Setup mocks
        mock_ollama.return_value = ("Generated report content", None, None, None)
        
        # Create test objects
        agent = Agent("Report Writer", [], {}, "test-model")
        task = Task("write_report", "Write a report")
        task_event = threading.Event()
        task_event.set()
        
        task_message = {
            "agent": agent,
            "task": task,
            "step_index": 0,
            "project_name": "test",
            "task_event": task_event
        }
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.update_task_callback = Mock()
            
            manager.execute_task(task_message)
            
            # Verify task was completed
            assert task.completed is True
            assert task.result == "Generated report content"
    
    def test_cancel_workflow(self):
        """Test canceling workflow"""
        from build_manager import BuildManager
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.message_queue = queue.Queue()
            manager.cancel_flag = False
            manager.parallel_task_counts = {"group1": 2}
            
            # Add tasks to queue
            manager.message_queue.put("task1")
            manager.message_queue.put("task2")
            
            manager.cancel_workflow()
            
            # Verify cancellation
            assert manager.cancel_flag is True
            assert manager.message_queue.empty()
            assert manager.parallel_task_counts == {}
    
    def test_reset_workflow(self):
        """Test resetting workflow"""
        from build_manager import BuildManager
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.message_queue = queue.Queue()
            manager.cancel_flag = True
            manager.parallel_task_counts = {"group1": 1}
            
            # Add tasks to queue
            manager.message_queue.put("task1")
            
            with patch('build_manager.st') as mock_st:
                mock_st.session_state = {
                    'bm_tasks': [{"id": 1}],
                    'workflow_start_time': "2023-01-01"
                }
                
                manager.reset_workflow()
                
                # Verify reset
                assert manager.cancel_flag is False
                assert manager.message_queue.empty()
                assert manager.parallel_task_counts == {}
                assert mock_st.session_state['bm_tasks'] == []
                assert mock_st.session_state['workflow_start_time'] is None
    
    @patch('build_manager.st')
    def test_cancel_task_by_id(self, mock_st):
        """Test canceling task by ID"""
        from build_manager import BuildManager
        
        # Setup session state with tasks
        mock_st.session_state = {
            'bm_tasks': [
                {"task_id": "task1", "name": "Agent1", "status": "Running"},
                {"task_id": "task2", "name": "Agent2", "status": "Pending"}
            ]
        }
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.message_queue = queue.Queue()
            manager.agents = {"Agent1": Mock()}
            
            # Add task message to queue
            task_event = threading.Event()
            task_message = {
                "task": Mock(),
                "task_event": task_event
            }
            task_message["task"].task_id = "task1"
            manager.message_queue.put(task_message)
            
            with patch.object(manager, 'cancel_task_by_index') as mock_cancel:
                manager.cancel_task("task1")
                
                # Verify cancel was called with correct index
                mock_cancel.assert_called_with(0)
    
    def test_cancel_task_by_index(self):
        """Test canceling task by index"""
        from build_manager import BuildManager
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.message_queue = queue.Queue()
            manager.agents = {"Agent1": Mock()}
            
            # Setup task message in queue
            task_event = threading.Event()
            task_message = {
                "task": Mock(),
                "task_event": task_event
            }
            task_message["task"].task_id = "task1"
            manager.message_queue.put(task_message)
            
            with patch('build_manager.st') as mock_st:
                mock_st.session_state = {
                    'bm_tasks': [
                        {"task_id": "task1", "name": "Agent1", "status": "Running"}
                    ]
                }
                
                manager.cancel_task_by_index(0)
                
                # Verify task status was updated
                assert mock_st.session_state['bm_tasks'][0]["status"] == "Canceled"
    
    def test_pause_task(self):
        """Test pausing task"""
        from build_manager import BuildManager
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.message_queue = queue.Queue()
            manager.update_task_callback = Mock()
            
            # Setup task message in queue
            task_event = threading.Event()
            task_event.set()  # Initially running
            task_message = {
                "task": Mock(),
                "task_event": task_event
            }
            task_message["task"].task_id = "task1"
            manager.message_queue.put(task_message)
            
            with patch('build_manager.st') as mock_st:
                mock_st.session_state = {
                    'bm_tasks': [
                        {"task_id": "task1", "name": "Agent1", "status": "Running"}
                    ]
                }
                
                manager.pause_task("task1")
                
                # Verify task was paused
                assert not task_event.is_set()
                manager.update_task_callback.assert_called_with(0, "Paused", None)
    
    def test_resume_task(self):
        """Test resuming task"""
        from build_manager import BuildManager
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.message_queue = queue.Queue()
            manager.update_task_callback = Mock()
            
            # Setup task message in queue
            task_event = threading.Event()
            task_event.clear()  # Initially paused
            task_message = {
                "task": Mock(),
                "task_event": task_event
            }
            task_message["task"].task_id = "task1"
            manager.message_queue.put(task_message)
            
            with patch('build_manager.st') as mock_st:
                mock_st.session_state = {
                    'bm_tasks': [
                        {"task_id": "task1", "name": "Agent1", "status": "Paused"}
                    ]
                }
                
                manager.resume_task("task1")
                
                # Verify task was resumed
                assert task_event.is_set()
                manager.update_task_callback.assert_called_with(0, "In Progress", None)


class TestWorkflowGeneration:
    """Test workflow generation functionality"""
    
    @patch('build_manager.ollama.generate')
    def test_generate_workflow_success(self, mock_generate):
        """Test successful workflow generation"""
        from build_manager import BuildManager
        
        # Setup mock response
        workflow_json = {
            "name": "Test Workflow",
            "description": "Test description",
            "steps": [
                {
                    "agent": "Test Agent",
                    "task_description": "Test task",
                    "inputs": ["input1"],
                    "outputs": ["output1"],
                    "depends_on": None,
                    "parallel_group": None,
                    "user_input": None,
                    "condition": []
                }
            ]
        }
        
        mock_generate.return_value = {"response": json.dumps(workflow_json)}
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            
            with patch.object(manager, 'validate_workflow_template', return_value=(True, "")):
                result = manager.generate_workflow("Create a data analysis workflow")
                
                assert result is not None
                parsed_result = json.loads(result)
                assert parsed_result["name"] == "Test Workflow"
                assert len(parsed_result["steps"]) == 1
    
    @patch('build_manager.ollama.generate')
    def test_generate_workflow_invalid_json(self, mock_generate):
        """Test workflow generation with invalid JSON"""
        from build_manager import BuildManager
        
        # Setup mock response with invalid JSON
        mock_generate.return_value = {"response": "{ invalid json }"}
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            
            result = manager.generate_workflow("Create a workflow")
            
            assert result is None
    
    @patch('build_manager.ollama.generate')
    def test_generate_workflow_validation_fails(self, mock_generate):
        """Test workflow generation with validation failure"""
        from build_manager import BuildManager
        
        # Setup mock response
        workflow_json = {"name": "Test", "steps": []}  # Missing description
        mock_generate.return_value = {"response": json.dumps(workflow_json)}
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            
            result = manager.generate_workflow("Create a workflow")
            
            assert result is None  # Should fail validation
    
    def test_validate_workflow_template_valid(self):
        """Test validation of valid workflow template"""
        from build_manager import BuildManager
        
        valid_workflow = {
            "name": "Valid Workflow",
            "description": "A valid workflow",
            "steps": [
                {
                    "agent": "Test Agent",
                    "task_description": "Do something",
                    "inputs": ["input1"],
                    "outputs": ["output1"],
                    "depends_on": None,
                    "parallel_group": "group1",
                    "condition": [
                        {
                            "expression": "True",
                            "next_step": 0
                        }
                    ]
                }
            ]
        }
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            
            is_valid, error = manager.validate_workflow_template(json.dumps(valid_workflow))
            
            assert is_valid is True
            assert error == ""
    
    def test_validate_workflow_template_invalid_json(self):
        """Test validation of invalid JSON"""
        from build_manager import BuildManager
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            
            is_valid, error = manager.validate_workflow_template("{ invalid json }")
            
            assert is_valid is False
            assert "Invalid JSON" in error
    
    def test_validate_workflow_template_missing_fields(self):
        """Test validation with missing required fields"""
        from build_manager import BuildManager
        
        # Missing description field
        invalid_workflow = {
            "name": "Test",
            "steps": []
        }
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            
            is_valid, error = manager.validate_workflow_template(json.dumps(invalid_workflow))
            
            assert is_valid is False
            assert "Missing required field: 'description'" in error
    
    def test_validate_workflow_template_invalid_step(self):
        """Test validation with invalid step"""
        from build_manager import BuildManager
        
        # Step missing required fields
        invalid_workflow = {
            "name": "Test",
            "description": "Test workflow",
            "steps": [
                {
                    "agent": "Test Agent"
                    # Missing task_description, inputs, outputs
                }
            ]
        }
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            
            is_valid, error = manager.validate_workflow_template(json.dumps(invalid_workflow))
            
            assert is_valid is False
            assert "missing required field" in error.lower()
    
    def test_validate_workflow_template_invalid_depends_on(self):
        """Test validation with invalid depends_on field"""
        from build_manager import BuildManager
        
        # depends_on references non-existent step
        invalid_workflow = {
            "name": "Test",
            "description": "Test workflow",
            "steps": [
                {
                    "agent": "Test Agent",
                    "task_description": "Do something",
                    "inputs": [],
                    "outputs": [],
                    "depends_on": 5  # Invalid - only 1 step exists
                }
            ]
        }
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            
            is_valid, error = manager.validate_workflow_template(json.dumps(invalid_workflow))
            
            assert is_valid is False
            assert "depends_on" in error
    
    def test_validate_workflow_template_invalid_condition(self):
        """Test validation with invalid condition"""
        from build_manager import BuildManager
        
        # Condition with invalid next_step
        invalid_workflow = {
            "name": "Test",
            "description": "Test workflow",
            "steps": [
                {
                    "agent": "Test Agent",
                    "task_description": "Do something",
                    "inputs": [],
                    "outputs": [],
                    "condition": [
                        {
                            "expression": "True",
                            "next_step": 10  # Invalid - only 1 step exists
                        }
                    ]
                }
            ]
        }
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            
            is_valid, error = manager.validate_workflow_template(json.dumps(invalid_workflow))
            
            assert is_valid is False
            assert "next_step" in error


class TestErrorHandlingAndLogging:
    """Test error handling and logging functionality"""
    
    @patch('build_manager.logging.error')
    def test_initialization_error_logging(self, mock_logging):
        """Test error logging during initialization"""
        from build_manager import BuildManager
        
        with patch('build_manager.os.listdir', side_effect=Exception("Directory error")):
            with patch.object(BuildManager, 'cancel_workflow') as mock_cancel:
                manager = BuildManager(queue.Queue(), Mock())
                
                # Verify error was logged and workflow canceled
                mock_logging.assert_called()
                mock_cancel.assert_called()
    
    @patch('build_manager.ollama.generate')
    @patch('build_manager.logging.error')
    def test_process_request_error_logging(self, mock_logging, mock_generate):
        """Test error logging in process_request"""
        from build_manager import BuildManager
        
        mock_generate.side_effect = Exception("Ollama error")
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.cancel_flag = False
            manager.update_task_callback = Mock()
            
            manager.process_request("Test request")
            
            # Verify error was logged
            mock_logging.assert_called()
            manager.update_task_callback.assert_called_with(0, "Failed", "Critical Error: Ollama error")
    
    @patch('build_manager.call_ollama_endpoint')
    @patch('build_manager.logging.error')
    def test_execute_task_error_logging(self, mock_logging, mock_ollama):
        """Test error logging in execute_task"""
        from build_manager import BuildManager
        from ollama_workbench.workflows.agents import Agent
        from ollama_workbench.workflows.projects import Task
        
        # Setup mocks
        mock_ollama.side_effect = Exception("Execution error")
        
        # Create test objects
        agent = Agent("Test Agent", [], {}, "test-model")
        task = Task("test_task", "Test task")
        task_event = threading.Event()
        task_event.set()
        
        task_message = {
            "agent": agent,
            "task": task,
            "step_index": 0,
            "project_name": "test",
            "task_event": task_event
        }
        
        with patch.object(BuildManager, '__init__', lambda x, y, z: None):
            manager = BuildManager(None, None)
            manager.update_task_callback = Mock()
            
            manager.execute_task(task_message)
            
            # Verify error was logged
            mock_logging.assert_called()
            manager.update_task_callback.assert_called_with(0, "Failed", "Error: Execution error")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
