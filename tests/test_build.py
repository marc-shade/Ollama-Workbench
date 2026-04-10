"""
Test suite for build.py - Build workflow system
"""

import pytest
import json
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBuildUtilities:
    """Test build utility functions"""
    
    def test_load_json_file_existing(self):
        """Test loading existing JSON file"""
        from ollama_workbench.workflows.build import load_json_file
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            test_data = {"key": "value", "number": 42}
            json.dump(test_data, tmp)
            tmp_path = tmp.name
        
        try:
            result = load_json_file(tmp_path)
            assert result == test_data
        finally:
            os.unlink(tmp_path)
    
    def test_load_json_file_nonexistent(self):
        """Test loading non-existent JSON file returns empty dict"""
        from ollama_workbench.workflows.build import load_json_file
        
        result = load_json_file("nonexistent_file.json")
        assert result == {}
    
    def test_save_json_file(self):
        """Test saving JSON file"""
        from ollama_workbench.workflows.build import save_json_file
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            test_data = {"test": "data", "array": [1, 2, 3]}
            save_json_file(tmp_path, test_data)
            
            with open(tmp_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data == test_data
        finally:
            os.unlink(tmp_path)
    
    @patch('ollama_workbench.workflows.build.save_json_file')
    @patch('ollama_workbench.workflows.build.load_json_file')
    def test_api_key_management(self, mock_load, mock_save):
        """Test API key loading and saving"""
        from ollama_workbench.workflows.build import load_api_keys, save_api_keys
        from ollama_workbench.providers.openai_utils import set_openai_api_key
        
        # Test load_api_keys
        mock_load.return_value = {"existing_key": "value"}
        result = load_api_keys()
        mock_load.assert_called_with("api_keys.json")
        assert result == {"existing_key": "value"}
        
        # Test save_api_keys
        test_keys = {"openai_api_key": "test_key"}
        save_api_keys(test_keys)
        mock_save.assert_called_with("api_keys.json", test_keys)
        
        # Test set_openai_api_key
        mock_load.return_value = {}
        set_openai_api_key("new_key")
        expected_keys = {"openai_api_key": "new_key"}
        mock_save.assert_called_with("api_keys.json", expected_keys)
    
    @patch('ollama_workbench.workflows.build.save_json_file')
    @patch('ollama_workbench.workflows.build.load_json_file')
    def test_settings_management(self, mock_load, mock_save):
        """Test settings loading and saving"""
        from ollama_workbench.workflows.build import load_settings, save_settings
        
        # Test load_settings
        mock_load.return_value = {"setting1": "value1"}
        result = load_settings()
        mock_load.assert_called_with("build_settings.json")
        assert result == {"setting1": "value1"}
        
        # Test save_settings
        test_settings = {"temperature": 0.7, "model": "gpt-4"}
        save_settings(test_settings)
        mock_save.assert_called_with("build_settings.json", test_settings)


class TestAPIIntegrations:
    """Test API integration functions"""
    
    @patch('ollama_workbench.workflows.build.openai.ChatCompletion.create')
    @patch('ollama_workbench.workflows.build.load_api_keys')
    def test_call_openai_api_success(self, mock_load_keys, mock_openai):
        """Test successful OpenAI API call"""
        from ollama_workbench.workflows.build import call_openai_api
        
        # Setup mocks
        mock_load_keys.return_value = {"openai_api_key": "test_key"}
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = {'content': 'Test response'}
        mock_openai.return_value = mock_response
        
        # Test API call
        messages = [{"role": "user", "content": "Hello"}]
        result = call_openai_api("gpt-4", messages)
        
        assert result == "Test response"
        mock_openai.assert_called_once()
    
    @patch('ollama_workbench.workflows.build.load_api_keys')
    def test_call_openai_api_missing_key(self, mock_load_keys):
        """Test OpenAI API call with missing key"""
        from ollama_workbench.workflows.build import call_openai_api
        
        mock_load_keys.return_value = {}
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(ValueError, match="Invalid or missing OpenAI API key"):
            call_openai_api("gpt-4", messages)
    
    def test_call_openai_api_invalid_parameters(self):
        """Test OpenAI API call with invalid parameters"""
        from ollama_workbench.workflows.build import call_openai_api
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(ValueError, match="Invalid value for one of the numerical parameters"):
            call_openai_api("gpt-4", messages, temperature="invalid")
    
    @patch('ollama_workbench.workflows.build.get_available_groq_models')
    def test_is_groq_model(self, mock_get_groq):
        """Test Groq model detection"""
        from ollama_workbench.workflows.build import is_groq_model
        
        mock_get_groq.return_value = ["mixtral-8x7b", "llama2-70b"]
        
        assert is_groq_model("mixtral-8x7b") is True
        assert is_groq_model("gpt-4") is False


class TestSearchFunctionality:
    """Test search functionality"""
    
    @patch('ollama_workbench.workflows.build.DDGS')
    def test_perform_search_duckduckgo(self, mock_ddgs):
        """Test DuckDuckGo search"""
        from ollama_workbench.workflows.build import perform_search
        
        # Setup mock
        mock_search_instance = Mock()
        mock_search_instance.text.return_value = [
            {"title": "Result 1", "href": "http://example1.com"},
            {"title": "Result 2", "href": "http://example2.com"}
        ]
        mock_ddgs.return_value.__enter__.return_value = mock_search_instance
        
        # Test search
        results = perform_search("test query", "duckduckgo", {}, 2)
        
        expected = [
            {"title": "Result 1", "url": "http://example1.com"},
            {"title": "Result 2", "url": "http://example2.com"}
        ]
        assert results == expected
        mock_search_instance.text.assert_called_with("test query", max_results=2)
    
    @patch('ollama_workbench.workflows.build.build')
    def test_perform_search_google(self, mock_build_service):
        """Test Google search"""
        from ollama_workbench.workflows.build import perform_search
        
        # Setup mock
        mock_service = Mock()
        mock_cse = Mock()
        mock_search = Mock()
        mock_search.execute.return_value = {
            "items": [
                {"title": "Google Result 1", "link": "http://google1.com"},
                {"title": "Google Result 2", "link": "http://google2.com"}
            ]
        }
        mock_cse.list.return_value = mock_search
        mock_service.cse.return_value = mock_cse
        mock_build_service.return_value = mock_service
        
        # Test search
        api_keys = {"google_api_key": "test_key", "google_cse_id": "test_cse"}
        results = perform_search("test query", "google", api_keys, 2)
        
        expected = [
            {"title": "Google Result 1", "url": "http://google1.com"},
            {"title": "Google Result 2", "url": "http://google2.com"}
        ]
        assert results == expected
    
    def test_perform_search_missing_keys(self):
        """Test search with missing API keys"""
        from ollama_workbench.workflows.build import perform_search
        
        # Test Google without keys
        results = perform_search("test", "google", {}, 5)
        assert results == []
        
        # Test SerpAPI without keys
        results = perform_search("test", "serpapi", {}, 5)
        assert results == []
    
    def test_perform_search_unsupported_method(self):
        """Test search with unsupported method"""
        from ollama_workbench.workflows.build import perform_search
        
        results = perform_search("test", "unsupported", {}, 5)
        assert results == []


class TestAgentTasks:
    """Test agent task functions"""
    
    def test_create_agent_context(self):
        """Test agent context creation"""
        from ollama_workbench.workflows.build import create_agent_context
        
        project_state = {
            "status": "In Progress",
            "current_step": "coding",
            "iterations": 2,
            "max_iterations": 5,
            "previous_tasks": ["task1", "task2"]
        }
        
        context = create_agent_context(project_state, "current task")
        
        assert context["project_state"]["status"] == "In Progress"
        assert context["current_task"] == "current task"
        assert context["previous_tasks"] == ["task1", "task2"]
    
    @patch('ollama_workbench.workflows.build.OpenAI')
    @patch('ollama_workbench.workflows.build.load_api_keys')
    def test_manager_agent_task_openai(self, mock_load_keys, mock_openai_class):
        """Test manager agent task with OpenAI"""
        from ollama_workbench.workflows.build import manager_agent_task
        
        # Setup mocks
        mock_load_keys.return_value = {"openai_api_key": "test_key"}
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"analysis": "test", "work_plan": "plan", "priorities": [], "instructions": "do this", "create_files": false}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Test manager task
        context = {"project_state": {"status": "active"}, "current_task": "test"}
        result, error = manager_agent_task(context, "gpt-4", 0.7, 1000, "test_key")
        
        assert result["analysis"] == "test"
        assert result["instructions"] == "do this"
        assert error is None
    
    @patch('ollama_workbench.workflows.build.call_groq_api')
    def test_manager_agent_task_groq(self, mock_groq):
        """Test manager agent task with Groq"""
        from ollama_workbench.workflows.build import manager_agent_task
        
        # Setup mock
        mock_groq.return_value = '{"analysis": "groq test", "work_plan": "groq plan", "priorities": [], "instructions": "groq task", "create_files": true}'
        
        with patch('ollama_workbench.workflows.build.GROQ_MODELS', ["mixtral-8x7b"]):
            context = {"project_state": {"status": "active"}, "current_task": "test"}
            result, error = manager_agent_task(context, "mixtral-8x7b", 0.7, 1000)
            
            assert result["analysis"] == "groq test"
            assert result["create_files"] is True
            assert error is None
    
    @patch('ollama_workbench.workflows.build.call_ollama_endpoint')
    def test_manager_agent_task_ollama(self, mock_ollama):
        """Test manager agent task with Ollama"""
        from ollama_workbench.workflows.build import manager_agent_task
        
        # Setup mock
        mock_ollama.return_value = ('{"analysis": "ollama test", "work_plan": "ollama plan", "priorities": [], "instructions": "ollama task", "create_files": false}', None, None, None)
        
        context = {"project_state": {"status": "active"}, "current_task": "test"}
        result, error = manager_agent_task(context, "llama3", 0.7, 1000)
        
        assert result["analysis"] == "ollama test"
        assert result["create_files"] is False
        assert error is None
    
    @patch('ollama_workbench.workflows.build.call_ollama_endpoint')
    def test_manager_agent_task_json_error(self, mock_ollama):
        """Test manager agent task with JSON parsing error"""
        from ollama_workbench.workflows.build import manager_agent_task
        
        # Setup mock with invalid JSON
        mock_ollama.return_value = ('invalid json response', None, None, None)
        
        context = {"project_state": {"status": "active"}, "current_task": "test"}
        result, error = manager_agent_task(context, "llama3", 0.7, 1000)
        
        assert "Unable to generate work plan" in result["work_plan"]
        assert "Review and fix JSON parsing issues" in result["priorities"]


class TestCodeGeneration:
    """Test code generation and file creation"""
    
    @patch('ollama_workbench.workflows.build.call_openai_api')
    def test_coding_agent_task_openai(self, mock_openai):
        """Test coding agent task with OpenAI"""
        from ollama_workbench.workflows.build import coding_agent_task
        
        # Setup mock
        mock_openai.return_value = '{"implementation": "code here", "explanation": "This is test code"}'
        
        with patch('ollama_workbench.workflows.build.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '{"implementation": "code here", "explanation": "This is test code"}'
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            result = coding_agent_task(
                "Generate a function",
                "gpt-4",
                openai_api_key="test_key"
            )
            
            assert result["implementation"] == "code here"
            assert result["explanation"] == "This is test code"
    
    @patch('ollama_workbench.workflows.build.call_groq_api')
    def test_coding_agent_task_groq(self, mock_groq):
        """Test coding agent task with Groq"""
        from ollama_workbench.workflows.build import coding_agent_task
        
        # Setup mock
        mock_groq.return_value = '{"code": "groq generated code", "notes": "Groq implementation"}'
        
        with patch('ollama_workbench.workflows.build.GROQ_MODELS', ["mixtral-8x7b"]):
            result = coding_agent_task(
                "Create a class",
                "mixtral-8x7b",
                groq_api_key="test_key"
            )
            
            assert result["code"] == "groq generated code"
            assert result["notes"] == "Groq implementation"
    
    @patch('ollama_workbench.workflows.build.call_ollama_endpoint')
    def test_coding_agent_task_ollama(self, mock_ollama):
        """Test coding agent task with Ollama"""
        from ollama_workbench.workflows.build import coding_agent_task
        
        # Setup mock
        mock_ollama.return_value = ('{"function": "def test(): pass", "description": "Test function"}', None, None, None)
        
        result = coding_agent_task("Write a test function", "llama3")
        
        assert result["function"] == "def test(): pass"
        assert result["description"] == "Test function"
    
    def test_coding_agent_task_empty_prompt(self):
        """Test coding agent task with empty prompt"""
        from ollama_workbench.workflows.build import coding_agent_task
        
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            coding_agent_task("", "llama3")
    
    @patch('ollama_workbench.workflows.build.call_ollama_endpoint')
    def test_coding_agent_task_with_search(self, mock_ollama):
        """Test coding agent task with search results"""
        from ollama_workbench.workflows.build import coding_agent_task
        
        mock_ollama.return_value = ('{"code": "enhanced code", "notes": "Used search data"}', None, None, None)
        
        search_results = [{"title": "Example", "url": "http://example.com"}]
        result = coding_agent_task(
            "Generate code",
            "llama3",
            use_search=True,
            search_results=search_results
        )
        
        assert result["code"] == "enhanced code"
        # Verify search results were included in the prompt
        call_args = mock_ollama.call_args[1]
        assert "Search Results:" in call_args["prompt"]


class TestFileOperations:
    """Test file operations and parsing"""
    
    def test_parse_folder_structure_valid(self):
        """Test parsing valid folder structure"""
        from ollama_workbench.workflows.build import parse_folder_structure
        
        structure_text = '''
        Here is the folder structure:
        <folder_structure>
        {
            "src": {
                "main.py": null,
                "utils": {
                    "helper.py": null
                }
            },
            "tests": {
                "test_main.py": null
            }
        }
        </folder_structure>
        '''
        
        result = parse_folder_structure(structure_text)
        
        assert result is not None
        assert "src" in result
        assert result["src"]["main.py"] is None
        assert "utils" in result["src"]
        assert result["src"]["utils"]["helper.py"] is None
    
    def test_parse_folder_structure_invalid_json(self):
        """Test parsing invalid JSON folder structure"""
        from ollama_workbench.workflows.build import parse_folder_structure
        
        structure_text = '''
        <folder_structure>
        { invalid json }
        </folder_structure>
        '''
        
        result = parse_folder_structure(structure_text)
        assert result is None
    
    def test_parse_folder_structure_no_tags(self):
        """Test parsing text without folder structure tags"""
        from ollama_workbench.workflows.build import parse_folder_structure
        
        structure_text = "No folder structure here"
        result = parse_folder_structure(structure_text)
        assert result is None
    
    def test_extract_code_blocks(self):
        """Test extracting code blocks from text"""
        from ollama_workbench.workflows.build import extract_code_blocks
        
        text = '''
        Here are the files:
        
        Filename: main.py
        ```python
        def main():
            print("Hello World")
        ```
        
        Filename: config.json
        ```json
        {"key": "value"}
        ```
        '''
        
        result = extract_code_blocks(text)
        
        assert "main.py" in result
        assert "def main():" in result["main.py"]
        assert "config.json" in result
        assert '{"key": "value"}' in result["config.json"]
    
    def test_extract_code_blocks_no_matches(self):
        """Test extracting code blocks with no matches"""
        from ollama_workbench.workflows.build import extract_code_blocks
        
        text = "No code blocks here"
        result = extract_code_blocks(text)
        assert result == {}
    
    def test_save_file(self):
        """Test saving file to project directory"""
        from ollama_workbench.workflows.build import save_file
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            content = "print('Hello World')"
            filename = "test.py"
            
            save_file(content, filename, tmp_dir)
            
            file_path = Path(tmp_dir) / "code" / filename
            assert file_path.exists()
            assert file_path.read_text() == content
    
    def test_create_repository_files(self):
        """Test creating repository files"""
        from ollama_workbench.workflows.build import create_repository_files
        
        refined_output = '''
        <folder_structure>
        {"src": {"main.py": null}}
        </folder_structure>
        
        Filename: src/main.py
        ```python
        def hello():
            return "Hello World"
        ```
        '''
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            
            created_files = create_repository_files(project_dir, refined_output)
            
            assert "src/main.py" in created_files
            assert "def hello():" in created_files["src/main.py"]
            
            # Verify file was actually created
            file_path = project_dir / "src" / "main.py"
            assert file_path.exists()
    
    def test_dump_repository(self):
        """Test dumping repository contents"""
        from ollama_workbench.workflows.build import dump_repository
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test files
            (Path(tmp_dir) / "test.py").write_text("print('test')")
            (Path(tmp_dir) / "config.json").write_text('{"test": true}')
            (Path(tmp_dir) / "README.md").write_text("# Test Project")
            (Path(tmp_dir) / "binary.exe").write_bytes(b"binary data")
            
            result = dump_repository(tmp_dir)
            
            assert "test.py" in result
            assert "config.json" in result
            assert "README.md" in result
            assert "binary.exe" not in result  # Binary files should be skipped
            
            assert result["test.py"] == "print('test')"
            assert result["config.json"] == '{"test": true}'


class TestProjectBuilding:
    """Test project building functions"""
    
    def test_build_project_success(self):
        """Test successful project build"""
        from ollama_workbench.workflows.build import build_project
        
        result = build_project("Create a calculator", {}, "Command-line Tool")
        
        assert result["success"] is True
        assert "project_dir" in result
        assert result["project_dir"].exists()
    
    def test_build_project_with_repo_contents(self):
        """Test project build with repository contents"""
        from ollama_workbench.workflows.build import build_project
        
        repo_contents = {
            "main.py": "print('hello')",
            "config.json": '{"name": "test"}'
        }
        
        result = build_project("Improve this code", repo_contents, "Python Script")
        
        assert result["success"] is True
        assert "project_dir" in result
    
    @patch('ollama_workbench.workflows.build.call_openai_api')
    def test_generate_readme(self, mock_openai):
        """Test README generation"""
        from ollama_workbench.workflows.build import generate_readme
        
        mock_openai.return_value = "# Test Project\n\nThis is a test README."
        
        readme = generate_readme(
            "Test Project",
            "Create a test app",
            "Streamlit App",
            "Refined output here",
            "gpt-4"
        )
        
        assert "# Test Project" in readme
        assert "This is a test README" in readme
    
    @patch('ollama_workbench.workflows.build.subprocess.run')
    def test_execute_code_command_line(self, mock_run):
        """Test executing command-line code"""
        from ollama_workbench.workflows.build import execute_code
        
        mock_run.return_value.stdout = "Command executed successfully"
        
        code = "print('Hello World')"
        result = execute_code(code, "Command-line Tool")
        
        assert result == "Command executed successfully"
        mock_run.assert_called_once()
    
    @patch('ollama_workbench.workflows.build.subprocess.Popen')
    def test_execute_code_streamlit(self, mock_popen):
        """Test executing Streamlit app"""
        from ollama_workbench.workflows.build import execute_code
        
        code = "import streamlit as st\nst.write('Hello')"
        result = execute_code(code, "Streamlit App")
        
        assert "Streamlit app launched" in result
        mock_popen.assert_called_once()
    
    def test_execute_code_unsupported_type(self):
        """Test executing unsupported project type"""
        from ollama_workbench.workflows.build import execute_code
        
        code = "print('test')"
        result = execute_code(code, "Unsupported Type")
        
        assert "Unsupported project type" in result


class TestRefinerTasks:
    """Test refiner task functionality"""
    
    @patch('ollama_workbench.workflows.build.call_openai_api')
    def test_refine_task_openai(self, mock_openai):
        """Test refine task with OpenAI"""
        from ollama_workbench.workflows.build import refine_task
        
        mock_openai.return_value = "Refined output with improved code and structure."
        
        result = refine_task(
            "Create a calculator",
            "gpt-4",
            ["Result 1", "Result 2"],
            "test_file",
            "calculator",
            openai_api_key="test_key"
        )
        
        assert result == "Refined output with improved code and structure."
        mock_openai.assert_called_once()
    
    @patch('ollama_workbench.workflows.build.call_groq_api')
    def test_refine_task_groq(self, mock_groq):
        """Test refine task with Groq"""
        from ollama_workbench.workflows.build import refine_task
        
        mock_groq.return_value = "Groq refined output"
        
        with patch('ollama_workbench.workflows.build.is_groq_model', return_value=True):
            result = refine_task(
                "Create a tool",
                "mixtral-8x7b",
                ["Task result"],
                "test_file",
                "tool",
                groq_api_key="test_key"
            )
            
            assert result == "Groq refined output"
    
    @patch('ollama_workbench.workflows.build.call_ollama_endpoint')
    def test_refine_task_ollama(self, mock_ollama):
        """Test refine task with Ollama"""
        from ollama_workbench.workflows.build import refine_task
        
        mock_ollama.return_value = ("Ollama refined output", None, None, None)
        
        result = refine_task(
            "Create an app",
            "llama3",
            ["Sub-task result"],
            "test_file",
            "app"
        )
        
        assert result == "Ollama refined output"
    
    @patch('ollama_workbench.workflows.build.call_ollama_endpoint')
    def test_refine_task_continuation(self, mock_ollama):
        """Test refine task with continuation"""
        from ollama_workbench.workflows.build import refine_task
        
        # First call returns long response, second call returns continuation
        mock_ollama.side_effect = [
            ("A" * 8000, None, None, None),  # Long response triggers continuation
            ("Continuation text", None, None, None)
        ]
        
        result = refine_task(
            "Create complex app",
            "llama3",
            ["Complex task result"],
            "test_file",
            "complex_app"
        )
        
        assert "A" * 8000 in result
        assert "Continuation text" in result
        assert mock_ollama.call_count == 2


class TestStreamlitInterface:
    """Test Streamlit interface components"""
    
    @patch('ollama_workbench.workflows.build.st')
    @patch('ollama_workbench.workflows.build.get_all_models')
    def test_build_interface_initialization(self, mock_get_models, mock_st):
        """Test build interface initialization"""
        from ollama_workbench.workflows.build import build_interface
        
        # Setup mocks
        mock_get_models.return_value = ["llama3", "gpt-4", "mixtral-8x7b"]
        mock_st.session_state = {}
        mock_st.title.return_value = None
        mock_st.progress.return_value = Mock()
        mock_st.empty.return_value = Mock()
        mock_st.sidebar = Mock()
        mock_st.sidebar.expander.return_value.__enter__ = Mock()
        mock_st.sidebar.expander.return_value.__exit__ = Mock()
        mock_st.selectbox.return_value = "llama3"
        mock_st.slider.return_value = 0.7
        mock_st.button.return_value = False
        mock_st.radio.return_value = "Enter Project Request"
        mock_st.text_area.return_value = ""
        mock_st.checkbox.return_value = False
        mock_st.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
        
        # Test interface initialization
        build_interface()
        
        # Verify key components were called
        mock_st.title.assert_called_once()
        mock_st.progress.assert_called_once()
        mock_get_models.assert_called_once()


class TestErrorHandling:
    """Test error handling throughout the build system"""
    
    @patch('ollama_workbench.workflows.build.call_ollama_endpoint')
    def test_manager_agent_task_exception(self, mock_ollama):
        """Test manager agent task with exception"""
        from ollama_workbench.workflows.build import manager_agent_task
        
        mock_ollama.side_effect = Exception("Connection error")
        
        context = {"project_state": {"status": "active"}, "current_task": "test"}
        result, error = manager_agent_task(context, "llama3", 0.7, 1000)
        
        assert result == {}
        assert error is None  # Function handles exceptions internally
    
    @patch('ollama_workbench.workflows.build.call_ollama_endpoint')
    def test_coding_agent_task_exception(self, mock_ollama):
        """Test coding agent task with exception"""
        from ollama_workbench.workflows.build import coding_agent_task
        
        mock_ollama.side_effect = Exception("Model error")
        
        result = coding_agent_task("Generate code", "llama3")
        
        assert "error" in result
        assert "Model error" in result["error"]
    
    @patch('ollama_workbench.workflows.build.call_ollama_endpoint')
    def test_refine_task_exception(self, mock_ollama):
        """Test refine task with exception"""
        from ollama_workbench.workflows.build import refine_task
        
        mock_ollama.side_effect = Exception("Refiner error")
        
        result = refine_task(
            "Create app",
            "llama3",
            ["task result"],
            "test_file",
            "app"
        )
        
        assert "error occurred during the refine task" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
