"""
Test suite for prompts.py - Prompt management functionality
"""

import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open, call
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFilePathManagement:
    """Test prompt file path management"""
    
    @patch('prompts.SCRIPT_DIR', '/test/dir')
    @patch('prompts.os.path.exists')
    @patch('prompts.os.makedirs')
    def test_get_prompts_file_path_creates_directory(self, mock_makedirs, mock_exists):
        """Test that prompts directory is created if it doesn't exist"""
        from ollama_workbench.ui.prompts import get_prompts_file_path
        
        mock_exists.return_value = False
        
        result = get_prompts_file_path("agent")
        
        expected_path = "/test/dir/prompts/agent_prompts.json"
        assert result == expected_path
        mock_makedirs.assert_called_once_with("/test/dir/prompts")
    
    @patch('prompts.SCRIPT_DIR', '/test/dir')
    @patch('prompts.os.path.exists')
    @patch('prompts.os.makedirs')
    def test_get_prompts_file_path_existing_directory(self, mock_makedirs, mock_exists):
        """Test file path when directory already exists"""
        from ollama_workbench.ui.prompts import get_prompts_file_path
        
        mock_exists.return_value = True
        
        result = get_prompts_file_path("metacognitive")
        
        expected_path = "/test/dir/prompts/metacognitive_prompts.json"
        assert result == expected_path
        mock_makedirs.assert_not_called()
    
    def test_get_prompts_file_path_different_types(self):
        """Test file paths for different prompt types"""
        from ollama_workbench.ui.prompts import get_prompts_file_path
        
        with patch('prompts.SCRIPT_DIR', '/base'):
            with patch('prompts.os.path.exists', return_value=True):
                assert get_prompts_file_path("agent").endswith("agent_prompts.json")
                assert get_prompts_file_path("voice").endswith("voice_prompts.json")
                assert get_prompts_file_path("identity").endswith("identity_prompts.json")


class TestPromptLoading:
    """Test prompt loading functionality"""
    
    @patch('prompts.get_prompts_file_path')
    @patch('prompts.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('prompts.json.load')
    def test_load_prompts_existing_file(self, mock_json_load, mock_file, mock_exists, mock_path):
        """Test loading prompts from existing file"""
        from ollama_workbench.ui.prompts import load_prompts
        
        mock_path.return_value = "/path/to/agent_prompts.json"
        mock_exists.return_value = True
        test_prompts = {"Agent1": "Test prompt"}
        mock_json_load.return_value = test_prompts
        
        result = load_prompts("agent")
        
        assert result == test_prompts
        mock_exists.assert_called_once_with("/path/to/agent_prompts.json")
        mock_file.assert_called_once_with("/path/to/agent_prompts.json", "r")
        mock_json_load.assert_called_once()
    
    @patch('prompts.get_prompts_file_path')
    @patch('prompts.os.path.exists')
    def test_load_prompts_nonexistent_file(self, mock_exists, mock_path):
        """Test loading prompts when file doesn't exist"""
        from ollama_workbench.ui.prompts import load_prompts
        
        mock_path.return_value = "/path/to/nonexistent.json"
        mock_exists.return_value = False
        
        result = load_prompts("agent")
        
        assert result == {}
    
    @patch('prompts.get_prompts_file_path')
    @patch('prompts.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('prompts.json.load')
    def test_load_prompts_json_decode_error(self, mock_json_load, mock_file, mock_exists, mock_path):
        """Test loading prompts with JSON decode error"""
        from ollama_workbench.ui.prompts import load_prompts
        
        mock_path.return_value = "/path/to/corrupted.json"
        mock_exists.return_value = True
        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
        
        with pytest.raises(json.JSONDecodeError):
            load_prompts("agent")


class TestPromptSaving:
    """Test prompt saving functionality"""
    
    @patch('prompts.get_prompts_file_path')
    @patch('builtins.open', new_callable=mock_open)
    @patch('prompts.json.dump')
    def test_save_prompts_success(self, mock_json_dump, mock_file, mock_path):
        """Test successful prompt saving"""
        from ollama_workbench.ui.prompts import save_prompts
        
        mock_path.return_value = "/path/to/agent_prompts.json"
        test_prompts = {"Agent1": "Test prompt", "Agent2": "Another prompt"}
        
        save_prompts("agent", test_prompts)
        
        mock_file.assert_called_once_with("/path/to/agent_prompts.json", "w")
        mock_json_dump.assert_called_once_with(test_prompts, mock_file(), indent=4)
    
    @patch('prompts.get_prompts_file_path')
    @patch('builtins.open', new_callable=mock_open)
    @patch('prompts.json.dump')
    def test_save_prompts_different_types(self, mock_json_dump, mock_file, mock_path):
        """Test saving different types of prompts"""
        from ollama_workbench.ui.prompts import save_prompts
        
        mock_path.side_effect = [
            "/path/agent.json",
            "/path/voice.json",
            "/path/metacognitive.json"
        ]
        
        save_prompts("agent", {"Agent": "prompt"})
        save_prompts("voice", {"Voice": "prompt"})
        save_prompts("metacognitive", {"Meta": "prompt"})
        
        assert mock_json_dump.call_count == 3
        assert mock_file.call_count == 3
    
    @patch('prompts.get_prompts_file_path')
    @patch('builtins.open', new_callable=mock_open)
    @patch('prompts.json.dump')
    def test_save_prompts_file_error(self, mock_json_dump, mock_file, mock_path):
        """Test saving prompts with file write error"""
        from ollama_workbench.ui.prompts import save_prompts
        
        mock_path.return_value = "/path/to/readonly.json"
        mock_file.side_effect = PermissionError("Permission denied")
        
        with pytest.raises(PermissionError):
            save_prompts("agent", {"test": "prompt"})


class TestAgentPrompts:
    """Test agent prompt management"""
    
    @patch('prompts.load_prompts')
    @patch('prompts.save_prompts')
    def test_get_agent_prompt_existing(self, mock_save, mock_load):
        """Test getting agent prompts when they exist"""
        from ollama_workbench.ui.prompts import get_agent_prompt
        
        existing_prompts = {
            "Custom Agent": {
                "prompt": "Custom prompt",
                "model_voice": "en-US-Wavenet-B"
            }
        }
        mock_load.return_value = existing_prompts
        
        result = get_agent_prompt()
        
        assert result == existing_prompts
        mock_load.assert_called_once_with("agent")
        mock_save.assert_called_once()  # Should save updated prompts
    
    @patch('prompts.load_prompts')
    @patch('prompts.save_prompts')
    def test_get_agent_prompt_create_default(self, mock_save, mock_load):
        """Test creating default agent prompts when none exist"""
        from ollama_workbench.ui.prompts import get_agent_prompt
        
        mock_load.return_value = {}
        
        result = get_agent_prompt()
        
        assert "General Assistant" in result
        assert "Code Assistant" in result
        assert "Technical Writer" in result
        
        # Check structure of default prompts
        for key, value in result.items():
            assert isinstance(value, dict)
            assert "prompt" in value
            assert "model_voice" in value
        
        mock_save.assert_called_once_with("agent", result)
    
    @patch('prompts.load_prompts')
    @patch('prompts.save_prompts')
    def test_get_agent_prompt_convert_string_prompts(self, mock_save, mock_load):
        """Test converting old string prompts to new dict format"""
        from ollama_workbench.ui.prompts import get_agent_prompt
        
        old_prompts = {
            "Old Agent": "This is an old string prompt",
            "New Agent": {
                "prompt": "This is a new dict prompt",
                "model_voice": "en-US-Wavenet-C"
            }
        }
        mock_load.return_value = old_prompts
        
        result = get_agent_prompt()
        
        # Old string prompt should be converted
        assert result["Old Agent"]["prompt"] == "This is an old string prompt"
        assert result["Old Agent"]["model_voice"] == "en-US-Wavenet-A"
        
        # New dict prompt should remain unchanged
        assert result["New Agent"]["prompt"] == "This is a new dict prompt"
        assert result["New Agent"]["model_voice"] == "en-US-Wavenet-C"
        
        mock_save.assert_called_once()
    
    @patch('prompts.load_prompts')
    @patch('prompts.save_prompts')
    def test_get_agent_prompt_add_missing_voice(self, mock_save, mock_load):
        """Test adding missing model_voice to existing prompts"""
        from ollama_workbench.ui.prompts import get_agent_prompt
        
        prompts_without_voice = {
            "Agent1": {
                "prompt": "Test prompt without voice"
            }
        }
        mock_load.return_value = prompts_without_voice
        
        result = get_agent_prompt()
        
        assert result["Agent1"]["model_voice"] == "en-US-Wavenet-A"
        mock_save.assert_called_once()


class TestOtherPromptTypes:
    """Test metacognitive, voice, and identity prompts"""
    
    @patch('prompts.load_prompts')
    @patch('prompts.save_prompts')
    def test_get_metacognitive_prompt_existing(self, mock_save, mock_load):
        """Test getting existing metacognitive prompts"""
        from ollama_workbench.ui.prompts import get_metacognitive_prompt
        
        existing_prompts = {
            "Custom": "Custom metacognitive prompt"
        }
        mock_load.return_value = existing_prompts
        
        result = get_metacognitive_prompt()
        
        assert result == existing_prompts
        mock_load.assert_called_once_with("metacognitive")
        mock_save.assert_not_called()
    
    @patch('prompts.load_prompts')
    @patch('prompts.save_prompts')
    def test_get_metacognitive_prompt_create_default(self, mock_save, mock_load):
        """Test creating default metacognitive prompts"""
        from ollama_workbench.ui.prompts import get_metacognitive_prompt
        
        mock_load.return_value = {}
        
        result = get_metacognitive_prompt()
        
        assert "Analytical" in result
        assert "Intuitive" in result
        assert "Collaborative" in result
        
        mock_save.assert_called_once_with("metacognitive", result)
    
    @patch('prompts.load_prompts')
    @patch('prompts.save_prompts')
    def test_get_voice_prompt_existing(self, mock_save, mock_load):
        """Test getting existing voice prompts"""
        from ollama_workbench.ui.prompts import get_voice_prompt
        
        existing_prompts = {
            "Friendly": "I speak in a very friendly manner"
        }
        mock_load.return_value = existing_prompts
        
        result = get_voice_prompt()
        
        assert result == existing_prompts
        mock_load.assert_called_once_with("voice")
        mock_save.assert_not_called()
    
    @patch('prompts.load_prompts')
    @patch('prompts.save_prompts')
    def test_get_voice_prompt_create_default(self, mock_save, mock_load):
        """Test creating default voice prompts"""
        from ollama_workbench.ui.prompts import get_voice_prompt
        
        mock_load.return_value = {}
        
        result = get_voice_prompt()
        
        assert "Professional" in result
        assert "Casual" in result
        assert "Technical" in result
        
        mock_save.assert_called_once_with("voice", result)
    
    @patch('prompts.load_prompts')
    def test_get_identity_prompt(self, mock_load):
        """Test getting identity prompts"""
        from ollama_workbench.ui.prompts import get_identity_prompt
        
        test_prompts = {"Identity1": "Test identity"}
        mock_load.return_value = test_prompts
        
        result = get_identity_prompt()
        
        assert result == test_prompts
        mock_load.assert_called_once_with("identity")


class TestStreamlitInterface:
    """Test Streamlit interface functionality"""
    
    @pytest.fixture
    def mock_streamlit(self):
        """Mock Streamlit components"""
        with patch('prompts.st') as mock_st:
            mock_st.title = Mock()
            mock_st.selectbox = Mock()
            mock_st.markdown = Mock()
            mock_st.expander = Mock()
            mock_st.text_area = Mock()
            mock_st.file_uploader = Mock()
            mock_st.button = Mock()
            mock_st.data_editor = Mock()
            mock_st.success = Mock()
            mock_st.rerun = Mock()
            yield mock_st
    
    @patch('prompts.get_agent_prompt')
    @patch('prompts.save_prompts')
    def test_manage_prompts_agent_selection(self, mock_save, mock_get_agent, mock_streamlit):
        """Test managing agent prompts"""
        from ollama_workbench.ui.prompts import manage_prompts
        
        mock_streamlit.selectbox.return_value = "Agent"
        mock_streamlit.button.return_value = False
        
        test_prompts = {
            "Test Agent": {
                "prompt": "Test prompt",
                "model_voice": "en-US-Wavenet-A"
            }
        }
        mock_get_agent.return_value = test_prompts
        
        # Mock expander context manager
        mock_expander = Mock()
        mock_expander.__enter__ = Mock(return_value=mock_expander)
        mock_expander.__exit__ = Mock(return_value=None)
        mock_streamlit.expander.return_value = mock_expander
        
        mock_streamlit.text_area.return_value = "Updated prompt"
        mock_streamlit.selectbox.side_effect = ["Agent", "en-US-Wavenet-B"]
        mock_streamlit.file_uploader.return_value = None
        
        manage_prompts()
        
        mock_streamlit.title.assert_called_once_with("✨ Prompts")
        mock_get_agent.assert_called_once()
    
    @patch('prompts.get_metacognitive_prompt')
    def test_manage_prompts_metacognitive_selection(self, mock_get_meta, mock_streamlit):
        """Test managing metacognitive prompts"""
        from ollama_workbench.ui.prompts import manage_prompts
        
        mock_streamlit.selectbox.return_value = "Metacognitive"
        mock_streamlit.button.return_value = False
        
        test_prompts = {"Analytical": "Test analytical prompt"}
        mock_get_meta.return_value = test_prompts
        
        mock_streamlit.data_editor.return_value = test_prompts
        
        manage_prompts()
        
        mock_get_meta.assert_called_once()
        mock_streamlit.data_editor.assert_called_once()
    
    @patch('prompts.get_voice_prompt')
    def test_manage_prompts_voice_selection(self, mock_get_voice, mock_streamlit):
        """Test managing voice prompts"""
        from ollama_workbench.ui.prompts import manage_prompts
        
        mock_streamlit.selectbox.return_value = "Voice"
        mock_streamlit.button.return_value = False
        
        test_prompts = {"Professional": "Professional tone"}
        mock_get_voice.return_value = test_prompts
        
        mock_streamlit.data_editor.return_value = test_prompts
        
        manage_prompts()
        
        mock_get_voice.assert_called_once()
        mock_streamlit.data_editor.assert_called_once()
    
    @patch('prompts.get_identity_prompt')
    def test_manage_prompts_identity_selection(self, mock_get_identity, mock_streamlit):
        """Test managing identity prompts"""
        from ollama_workbench.ui.prompts import manage_prompts
        
        mock_streamlit.selectbox.return_value = "Identity"
        mock_streamlit.button.return_value = False
        
        test_prompts = {"Expert": "Expert identity"}
        mock_get_identity.return_value = test_prompts
        
        mock_streamlit.data_editor.return_value = test_prompts
        
        manage_prompts()
        
        mock_get_identity.assert_called_once()
        mock_streamlit.data_editor.assert_called_once()
    
    @patch('prompts.load_prompts')
    def test_manage_prompts_model_voice_selection(self, mock_load, mock_streamlit):
        """Test managing model voice prompts"""
        from ollama_workbench.ui.prompts import manage_prompts
        
        mock_streamlit.selectbox.return_value = "Model Voice"
        mock_streamlit.button.return_value = False
        
        test_prompts = {"Voice1": "Voice setting"}
        mock_load.return_value = test_prompts
        
        mock_streamlit.data_editor.return_value = test_prompts
        
        manage_prompts()
        
        mock_load.assert_called_once_with("model_voice")
        mock_streamlit.data_editor.assert_called_once()
    
    @patch('prompts.get_agent_prompt')
    @patch('prompts.save_prompts')
    def test_manage_prompts_save_agent(self, mock_save, mock_get_agent, mock_streamlit):
        """Test saving agent prompts"""
        from ollama_workbench.ui.prompts import manage_prompts
        
        mock_streamlit.selectbox.return_value = "Agent"
        mock_streamlit.button.return_value = True  # Save button clicked
        
        test_prompts = {
            "Agent1": {
                "prompt": "Test prompt",
                "model_voice": "en-US-Wavenet-A"
            }
        }
        mock_get_agent.return_value = test_prompts
        
        # Mock expander context manager
        mock_expander = Mock()
        mock_expander.__enter__ = Mock(return_value=mock_expander)
        mock_expander.__exit__ = Mock(return_value=None)
        mock_streamlit.expander.return_value = mock_expander
        
        mock_streamlit.text_area.return_value = "Test prompt"
        mock_streamlit.selectbox.side_effect = ["Agent", "en-US-Wavenet-A"]
        mock_streamlit.file_uploader.return_value = None
        
        manage_prompts()
        
        mock_save.assert_called_once_with("agent", test_prompts)
        mock_streamlit.success.assert_called_once()
        mock_streamlit.rerun.assert_called_once()
    
    @patch('prompts.get_metacognitive_prompt')
    @patch('prompts.save_prompts')
    def test_manage_prompts_save_metacognitive(self, mock_save, mock_get_meta, mock_streamlit):
        """Test saving metacognitive prompts"""
        from ollama_workbench.ui.prompts import manage_prompts
        
        mock_streamlit.selectbox.return_value = "Metacognitive"
        mock_streamlit.button.return_value = True  # Save button clicked
        
        original_prompts = {"Analytical": "Original prompt"}
        edited_prompts = {"Analytical": "Edited prompt"}
        
        mock_get_meta.return_value = original_prompts
        mock_streamlit.data_editor.return_value = edited_prompts
        
        manage_prompts()
        
        mock_save.assert_called_once_with("metacognitive", edited_prompts)
        mock_streamlit.success.assert_called_once()
        mock_streamlit.rerun.assert_called_once()


class TestVRMIntegration:
    """Test VRM model integration"""
    
    @patch('prompts.global_vrm_loader')
    @patch('prompts.os.path.exists')
    @patch('prompts.os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_vrm_model_upload(self, mock_file, mock_makedirs, mock_exists, mock_vrm_loader):
        """Test VRM model upload functionality"""
        from ollama_workbench.ui.prompts import manage_prompts
        
        with patch('prompts.st') as mock_st:
            # Setup mocks
            mock_st.title = Mock()
            mock_st.selectbox.return_value = "Agent"
            mock_st.button.return_value = False
            mock_st.markdown = Mock()
            mock_st.success = Mock()
            
            # Mock file upload
            mock_vrm_file = Mock()
            mock_vrm_file.name = "test_model.vrm"
            mock_vrm_file.getvalue.return_value = b"fake vrm data"
            mock_st.file_uploader.return_value = mock_vrm_file
            
            # Mock other UI components
            mock_expander = Mock()
            mock_expander.__enter__ = Mock(return_value=mock_expander)
            mock_expander.__exit__ = Mock(return_value=None)
            mock_st.expander.return_value = mock_expander
            
            mock_st.text_area.return_value = "Test prompt"
            
            mock_exists.return_value = False  # Directory doesn't exist
            
            with patch('prompts.get_agent_prompt') as mock_get_agent:
                mock_get_agent.return_value = {
                    "Test Agent": {
                        "prompt": "Test prompt",
                        "model_voice": "en-US-Wavenet-A"
                    }
                }
                
                with patch('prompts.SCRIPT_DIR', '/test/dir'):
                    manage_prompts()
        
        # Verify directory creation
        mock_makedirs.assert_called_with("/test/dir/agent_models")
        
        # Verify file save
        mock_file.assert_called_with("/test/dir/agent_models/test_model.vrm", "wb")
        mock_file().write.assert_called_with(b"fake vrm data")
        
        # Verify VRM loader called
        mock_vrm_loader.load_model.assert_called_once()
        
        # Verify success message
        mock_st.success.assert_called()


class TestAgentPromptManagement:
    """Test agent prompt specific management features"""
    
    @patch('prompts.get_agent_prompt')
    @patch('prompts.save_prompts')
    def test_add_new_agent_prompt(self, mock_save, mock_get_agent):
        """Test adding new agent prompt"""
        from ollama_workbench.ui.prompts import manage_prompts
        
        with patch('prompts.st') as mock_st:
            mock_st.title = Mock()
            mock_st.selectbox.return_value = "Agent"
            mock_st.markdown = Mock()
            
            # Mock existing prompts
            existing_prompts = {
                "Agent1": {
                    "prompt": "Existing prompt",
                    "model_voice": "en-US-Wavenet-A"
                }
            }
            mock_get_agent.return_value = existing_prompts
            
            # Mock expander for existing agents
            mock_expander = Mock()
            mock_expander.__enter__ = Mock(return_value=mock_expander)
            mock_expander.__exit__ = Mock(return_value=None)
            mock_st.expander.return_value = mock_expander
            
            mock_st.text_area.return_value = "Existing prompt"
            mock_st.selectbox.side_effect = ["Agent", "en-US-Wavenet-A"]
            mock_st.file_uploader.return_value = None
            
            # Mock "Add New Agent Prompt" button clicked
            mock_st.button.side_effect = [True, False]  # Add button, then Save button
            mock_st.success = Mock()
            mock_st.rerun = Mock()
            
            manage_prompts()
        
        # Verify new prompt was added
        assert len(existing_prompts) == 2
        assert "New Agent 2" in existing_prompts
        assert existing_prompts["New Agent 2"]["prompt"] == ""
        assert existing_prompts["New Agent 2"]["model_voice"] == "en-US-Wavenet-A"
        
        mock_st.success.assert_called_with("Added new agent prompt: New Agent 2")
        mock_st.rerun.assert_called_once()


class TestErrorHandling:
    """Test error handling scenarios"""
    
    @patch('prompts.get_prompts_file_path')
    @patch('prompts.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_prompts_file_permission_error(self, mock_file, mock_exists, mock_path):
        """Test loading prompts with file permission error"""
        from ollama_workbench.ui.prompts import load_prompts
        
        mock_path.return_value = "/restricted/file.json"
        mock_exists.return_value = True
        mock_file.side_effect = PermissionError("Permission denied")
        
        with pytest.raises(PermissionError):
            load_prompts("agent")
    
    @patch('prompts.get_prompts_file_path')
    @patch('builtins.open', new_callable=mock_open)
    @patch('prompts.json.dump')
    def test_save_prompts_json_error(self, mock_json_dump, mock_file, mock_path):
        """Test saving prompts with JSON serialization error"""
        from ollama_workbench.ui.prompts import save_prompts
        
        mock_path.return_value = "/path/to/file.json"
        mock_json_dump.side_effect = TypeError("Object not JSON serializable")
        
        with pytest.raises(TypeError):
            save_prompts("agent", {"test": Mock()})  # Mock object not serializable
    
    @patch('prompts.load_prompts')
    @patch('prompts.save_prompts')
    def test_get_agent_prompt_save_error(self, mock_save, mock_load):
        """Test agent prompt loading with save error"""
        from ollama_workbench.ui.prompts import get_agent_prompt
        
        mock_load.return_value = {}
        mock_save.side_effect = IOError("Disk full")
        
        with pytest.raises(IOError):
            get_agent_prompt()


class TestIntegration:
    """Test integration scenarios"""
    
    def test_module_imports(self):
        """Test that all required modules can be imported"""
        import ollama_workbench.ui.prompts as prompts

        
        # Test that main functions exist
        assert hasattr(prompts, 'get_prompts_file_path')
        assert hasattr(prompts, 'load_prompts')
        assert hasattr(prompts, 'save_prompts')
        assert hasattr(prompts, 'get_agent_prompt')
        assert hasattr(prompts, 'get_metacognitive_prompt')
        assert hasattr(prompts, 'get_voice_prompt')
        assert hasattr(prompts, 'get_identity_prompt')
        assert hasattr(prompts, 'manage_prompts')
    
    @patch('prompts.load_prompts')
    @patch('prompts.save_prompts')
    def test_prompt_type_consistency(self, mock_save, mock_load):
        """Test consistency across different prompt types"""
        from ollama_workbench.ui.prompts import get_agent_prompt, get_metacognitive_prompt, get_voice_prompt
        
        # Mock loading empty prompts to trigger default creation
        mock_load.return_value = {}
        
        agent_prompts = get_agent_prompt()
        meta_prompts = get_metacognitive_prompt()
        voice_prompts = get_voice_prompt()
        
        # All should return dictionaries
        assert isinstance(agent_prompts, dict)
        assert isinstance(meta_prompts, dict)
        assert isinstance(voice_prompts, dict)
        
        # All should have content
        assert len(agent_prompts) > 0
        assert len(meta_prompts) > 0
        assert len(voice_prompts) > 0
        
        # Agent prompts should have complex structure
        for key, value in agent_prompts.items():
            assert isinstance(value, dict)
            assert "prompt" in value
            assert "model_voice" in value
    
    @patch('prompts.SCRIPT_DIR', '/test/workbench')
    def test_directory_structure_consistency(self):
        """Test that directory paths are consistent"""
        from ollama_workbench.ui.prompts import get_prompts_file_path
        
        with patch('prompts.os.path.exists', return_value=True):
            agent_path = get_prompts_file_path("agent")
            voice_path = get_prompts_file_path("voice")
            meta_path = get_prompts_file_path("metacognitive")
        
        # All should be in the same prompts directory
        assert "/test/workbench/prompts/" in agent_path
        assert "/test/workbench/prompts/" in voice_path
        assert "/test/workbench/prompts/" in meta_path
        
        # Should have correct filename patterns
        assert agent_path.endswith("agent_prompts.json")
        assert voice_path.endswith("voice_prompts.json")
        assert meta_path.endswith("metacognitive_prompts.json")


class TestDefaultPromptContent:
    """Test default prompt content and structure"""
    
    @patch('prompts.load_prompts')
    @patch('prompts.save_prompts')
    def test_default_agent_prompts_content(self, mock_save, mock_load):
        """Test content of default agent prompts"""
        from ollama_workbench.ui.prompts import get_agent_prompt
        
        mock_load.return_value = {}
        
        result = get_agent_prompt()
        
        # Check default agents exist
        assert "General Assistant" in result
        assert "Code Assistant" in result
        assert "Technical Writer" in result
        
        # Check structure and content
        general = result["General Assistant"]
        assert "helpful AI assistant" in general["prompt"].lower()
        assert general["model_voice"] == "en-US-Wavenet-A"
        
        code = result["Code Assistant"]
        assert "programming" in code["prompt"].lower() or "code" in code["prompt"].lower()
        assert code["model_voice"] == "en-US-Wavenet-B"
        
        technical = result["Technical Writer"]
        assert "technical" in technical["prompt"].lower() or "documentation" in technical["prompt"].lower()
        assert technical["model_voice"] == "en-US-Wavenet-C"
    
    @patch('prompts.load_prompts')
    @patch('prompts.save_prompts')
    def test_default_metacognitive_prompts_content(self, mock_save, mock_load):
        """Test content of default metacognitive prompts"""
        from ollama_workbench.ui.prompts import get_metacognitive_prompt
        
        mock_load.return_value = {}
        
        result = get_metacognitive_prompt()
        
        # Check default metacognitive types
        assert "Analytical" in result
        assert "Intuitive" in result
        assert "Collaborative" in result
        
        # Check content makes sense
        assert "systematic" in result["Analytical"].lower() or "analyz" in result["Analytical"].lower()
        assert "intuitive" in result["Intuitive"].lower() or "creative" in result["Intuitive"].lower()
        assert "collaborative" in result["Collaborative"].lower() or "involving" in result["Collaborative"].lower()
    
    @patch('prompts.load_prompts')
    @patch('prompts.save_prompts')
    def test_default_voice_prompts_content(self, mock_save, mock_load):
        """Test content of default voice prompts"""
        from ollama_workbench.ui.prompts import get_voice_prompt
        
        mock_load.return_value = {}
        
        result = get_voice_prompt()
        
        # Check default voice types
        assert "Professional" in result
        assert "Casual" in result
        assert "Technical" in result
        
        # Check content makes sense
        assert "professional" in result["Professional"].lower()
        assert "casual" in result["Casual"].lower() or "friendly" in result["Casual"].lower()
        assert "technical" in result["Technical"].lower() or "precise" in result["Technical"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
