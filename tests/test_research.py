"""
Test suite for research.py - Research workflow system
"""

import pytest
import json
import os
import tempfile
import sqlite3
from unittest.mock import Mock, patch, MagicMock, call, mock_open
from pathlib import Path
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAPIKeyManagement:
    """Test API key management functions"""
    
    def test_load_api_keys_existing_file(self):
        """Test loading API keys from existing file"""
        from ollama_workbench.providers.ollama_utils import load_api_keys
        import ollama_workbench.providers.ollama_utils as ou

        test_keys = {
            "openai_api_key": "test_openai_key",
            "google_api_key": "test_google_key",
            "serpapi_api_key": "test_serpapi_key"
        }

        # Clear cache to ensure fresh read
        ou._api_keys_cache = None
        ou._api_keys_cache_time = 0

        with patch('ollama_workbench.providers.ollama_utils.os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(test_keys))):
                result = load_api_keys()
                assert result == test_keys

        # Clear cache after test
        ou._api_keys_cache = None
        ou._api_keys_cache_time = 0
    
    def test_load_api_keys_nonexistent_file(self):
        """Test loading API keys when file doesn't exist"""
        from ollama_workbench.workflows.research import load_api_keys
        
        with patch('ollama_workbench.workflows.research.os.path.exists', return_value=False):
            result = load_api_keys()
            assert result == {}
    
    def test_save_api_keys(self):
        """Test saving API keys to file"""
        from ollama_workbench.providers.ollama_utils import save_api_keys
        import ollama_workbench.providers.ollama_utils as ou

        test_keys = {
            "openai_api_key": "new_key",
            "groq_api_key": "groq_key"
        }

        m = mock_open()
        with patch('builtins.open', m):
            with patch('ollama_workbench.providers.ollama_utils.os.chmod') as mock_chmod:
                save_api_keys(test_keys)
                m.assert_called_once()
                mock_chmod.assert_called_once_with("api_keys.json", 0o600)

        # Verify cache was invalidated
        assert ou._api_keys_cache is None


class TestDatabaseFunctions:
    """Test database functions"""
    
    def setup_method(self):
        """Setup test database"""
        self.test_db = ":memory:"
    
    @patch('ollama_workbench.workflows.research.sqlite3.connect')
    def test_init_db(self, mock_connect):
        """Test database initialization"""
        from ollama_workbench.workflows.research import init_db
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        
        init_db()
        
        mock_connect.assert_called_with('research_reports.db')
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
        # Context manager handles cleanup
    
    @patch('ollama_workbench.workflows.research.sqlite3.connect')
    @patch('ollama_workbench.workflows.research.datetime')
    def test_save_report(self, mock_datetime, mock_connect):
        """Test saving report to database"""
        from ollama_workbench.workflows.research import save_report
        
        # Setup mocks
        mock_datetime.now.return_value.strftime.return_value = "2023-01-01 12:00:00"
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        
        # Test save report
        save_report("Test Title", "Test Content")
        
        mock_connect.assert_called_with('research_reports.db')
        mock_cursor.execute.assert_called_with(
            "INSERT INTO reports (title, content, date) VALUES (?, ?, ?)",
            ("Test Title", "Test Content", "2023-01-01 12:00:00")
        )
        mock_conn.commit.assert_called_once()
        # Context manager handles cleanup
    
    @patch('ollama_workbench.workflows.research.sqlite3.connect')
    def test_get_all_reports(self, mock_connect):
        """Test getting all reports from database"""
        from ollama_workbench.workflows.research import get_all_reports
        
        # Setup mocks
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (1, "Report 1", "2023-01-01"),
            (2, "Report 2", "2023-01-02")
        ]
        
        # Test get all reports
        result = get_all_reports()
        
        mock_connect.assert_called_with('research_reports.db')
        mock_cursor.execute.assert_called_with("SELECT id, title, date FROM reports")
        assert result == [(1, "Report 1", "2023-01-01"), (2, "Report 2", "2023-01-02")]
        # Context manager handles cleanup
    
    @patch('ollama_workbench.workflows.research.sqlite3.connect')
    def test_get_report_content(self, mock_connect):
        """Test getting specific report content"""
        from ollama_workbench.workflows.research import get_report_content
        
        # Setup mocks
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = ("Report content here",)
        
        # Test get report content
        result = get_report_content(1)
        
        mock_connect.assert_called_with('research_reports.db')
        mock_cursor.execute.assert_called_with("SELECT content FROM reports WHERE id=?", (1,))
        assert result == "Report content here"
        # Context manager handles cleanup
    
    @patch('ollama_workbench.workflows.research.sqlite3.connect')
    def test_delete_report(self, mock_connect):
        """Test deleting report from database"""
        from ollama_workbench.workflows.research import delete_report
        
        # Setup mocks
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        
        # Test delete report
        delete_report(1)
        
        mock_connect.assert_called_with('research_reports.db')
        mock_cursor.execute.assert_called_with("DELETE FROM reports WHERE id=?", (1,))
        mock_conn.commit.assert_called_once()
        # Context manager handles cleanup


class TestResearchModelSettings:
    """Test research model settings management"""
    
    def test_load_research_model_settings_existing(self):
        """Test loading research model settings from existing file"""
        from ollama_workbench.workflows.research import load_research_model_settings
        
        test_settings = {
            "manager_model": "gpt-4",
            "manager_temperature": 0.7,
            "agent_model": "llama3",
            "agent_temperature": 0.8
        }
        
        with patch('ollama_workbench.workflows.research.os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(test_settings))):
                result = load_research_model_settings()
                assert result == test_settings
    
    def test_load_research_model_settings_nonexistent(self):
        """Test loading research model settings when file doesn't exist"""
        from ollama_workbench.workflows.research import load_research_model_settings
        
        with patch('ollama_workbench.workflows.research.os.path.exists', return_value=False):
            result = load_research_model_settings()
            assert result == {}
    
    def test_save_research_model_settings(self):
        """Test saving research model settings"""
        from ollama_workbench.workflows.research import save_research_model_settings
        
        test_settings = {
            "manager_model": "mixtral-8x7b",
            "manager_temperature": 0.5,
            "manager_max_tokens": 2000
        }
        
        mock_file = mock_open()
        with patch('builtins.open', mock_file):
            with patch('ollama_workbench.workflows.research.json.dump') as mock_dump:
                save_research_model_settings(test_settings)
                
                mock_file.assert_called_once_with("research_models.json", "w")
                mock_dump.assert_called_once_with(test_settings, mock_file().__enter__(), indent=4)


class TestExportFunctions:
    """Test export functionality"""
    
    @patch('ollama_workbench.workflows.research.SimpleDocTemplate')
    @patch('ollama_workbench.workflows.research.os.path.join')
    def test_export_to_pdf(self, mock_join, mock_pdf_doc):
        """Test PDF export functionality"""
        from ollama_workbench.workflows.research import export_to_pdf
        
        # Setup mocks
        mock_join.return_value = "/path/to/test.pdf"
        mock_doc_instance = Mock()
        mock_pdf_doc.return_value = mock_doc_instance
        
        # Test content with different sections
        content = """Final Report:
This is the main report content.

References:
1. Source 1
2. Source 2

Search Results:
Result 1 content

Result 2 content"""
        
        result = export_to_pdf(content, "test.pdf")
        
        # Verify PDF document was created
        mock_pdf_doc.assert_called_once()
        mock_doc_instance.build.assert_called_once()
        assert result == "/path/to/test.pdf"
    
    @patch('ollama_workbench.workflows.research.os.path.join')
    def test_export_to_txt(self, mock_join):
        """Test TXT export functionality"""
        from ollama_workbench.workflows.research import export_to_txt
        
        # Setup mocks
        mock_join.return_value = "/path/to/test.txt"
        
        content = "This is test content for TXT export."
        
        mock_file = mock_open()
        with patch('builtins.open', mock_file):
            result = export_to_txt(content, "test.txt")
            
            mock_file.assert_called_once_with("/path/to/test.txt", 'w', encoding='utf-8')
            mock_file().__enter__().write.assert_called_once_with(content)
            assert result == "/path/to/test.txt"


class TestResearchInterface:
    """Test research interface components"""
    
    @patch('ollama_workbench.workflows.research.get_all_reports', return_value=[])
    @patch('ollama_workbench.workflows.research.st')
    @patch('ollama_workbench.workflows.research.init_db')
    @patch('ollama_workbench.workflows.research.load_api_keys')
    @patch('ollama_workbench.workflows.research.load_research_model_settings')
    @patch('ollama_workbench.workflows.research.get_available_models')
    def test_research_interface_initialization(self, mock_get_models, mock_load_settings,
                                             mock_load_keys, mock_init_db, mock_st,
                                             mock_get_reports):
        """Test research interface initialization"""
        from ollama_workbench.workflows.research import research_interface

        # Setup mocks
        mock_load_keys.return_value = {"openai_api_key": "test_key"}
        mock_load_settings.return_value = {"manager_model": "gpt-4"}
        mock_get_models.return_value = ["llama3", "gpt-4", "mixtral-8x7b"]
        mock_st.sidebar = MagicMock()
        mock_st.text_input.return_value = ""
        mock_st.selectbox.return_value = "gpt-4"
        mock_st.slider.return_value = 0.7
        mock_st.button.return_value = False
        mock_st.text_area.return_value = ""
        
        # Test interface initialization
        research_interface()
        
        # Verify key functions were called
        mock_init_db.assert_called_once()
        mock_load_keys.assert_called()
        mock_load_settings.assert_called()
        mock_get_models.assert_called()
        mock_st.title.assert_called()
    
    @patch('ollama_workbench.workflows.research.get_all_reports', return_value=[])
    @patch('ollama_workbench.workflows.research.st')
    @patch('ollama_workbench.workflows.research.save_api_keys')
    @patch('ollama_workbench.workflows.research.init_db')
    @patch('ollama_workbench.workflows.research.load_api_keys')
    @patch('ollama_workbench.workflows.research.load_research_model_settings')
    @patch('ollama_workbench.workflows.research.get_available_models')
    def test_api_key_saving(self, mock_get_models, mock_load_settings,
                           mock_load_keys, mock_init_db, mock_save_keys, mock_st,
                           mock_get_reports):
        """Test API key saving functionality"""
        from ollama_workbench.workflows.research import research_interface

        # Setup mocks
        mock_load_keys.return_value = {}
        mock_load_settings.return_value = {}
        mock_get_models.return_value = ["llama3"]
        mock_st.sidebar = MagicMock()

        # Mock text inputs for API keys
        def text_input_side_effect(label, **kwargs):
            if "API Key" in label:
                return "test_key_value"
            return ""
        
        mock_st.text_input.side_effect = text_input_side_effect
        mock_st.selectbox.return_value = "llama3"
        mock_st.slider.return_value = 0.7
        mock_st.button.side_effect = [True, False, False]  # Save API Keys button clicked
        mock_st.text_area.return_value = ""
        
        research_interface()
        
        # Verify save_api_keys was called
        mock_save_keys.assert_called_once()
    
    @patch('ollama_workbench.workflows.research.get_all_reports', return_value=[])
    @patch('ollama_workbench.workflows.research.st')
    @patch('ollama_workbench.workflows.research.save_research_model_settings')
    @patch('ollama_workbench.workflows.research.init_db')
    @patch('ollama_workbench.workflows.research.load_api_keys')
    @patch('ollama_workbench.workflows.research.load_research_model_settings')
    @patch('ollama_workbench.workflows.research.get_available_models')
    def test_model_settings_saving(self, mock_get_models, mock_load_settings,
                                  mock_load_keys, mock_init_db, mock_save_settings, mock_st,
                                  mock_get_reports):
        """Test model settings saving functionality"""
        from ollama_workbench.workflows.research import research_interface

        # Setup mocks
        mock_load_keys.return_value = {}
        mock_load_settings.return_value = {}
        mock_get_models.return_value = ["llama3", "gpt-4"]
        mock_st.sidebar = MagicMock()
        mock_st.text_input.return_value = ""
        mock_st.selectbox.return_value = "gpt-4"
        mock_st.slider.return_value = 0.8
        mock_st.button.side_effect = [False, True, False]  # Save Model Settings button clicked
        mock_st.text_area.return_value = ""
        
        research_interface()
        
        # Verify save_research_model_settings was called
        mock_save_settings.assert_called_once()
        call_args = mock_save_settings.call_args[0][0]
        assert "manager_model" in call_args
        assert "agent_model" in call_args
    
    @patch('ollama_workbench.workflows.research.get_all_reports', return_value=[])
    @patch('ollama_workbench.workflows.research.st')
    @patch('ollama_workbench.workflows.research.SearchManager')
    @patch('ollama_workbench.workflows.research.save_report')
    @patch('ollama_workbench.workflows.research.init_db')
    @patch('ollama_workbench.workflows.research.load_api_keys')
    @patch('ollama_workbench.workflows.research.load_research_model_settings')
    @patch('ollama_workbench.workflows.research.get_available_models')
    def test_research_execution(self, mock_get_models, mock_load_settings,
                               mock_load_keys, mock_init_db, mock_save_report,
                               mock_search_manager_class, mock_st, mock_get_reports):
        """Test research execution functionality"""
        from ollama_workbench.workflows.research import research_interface
        
        # Setup mocks
        mock_load_keys.return_value = {"openai_api_key": "test_key"}
        mock_load_settings.return_value = {
            "manager_model": "gpt-4",
            "manager_temperature": 0.7,
            "manager_max_tokens": 4000,
            "agent_model": "llama3",
            "agent_temperature": 0.8,
            "agent_max_tokens": 3000
        }
        mock_get_models.return_value = ["llama3", "gpt-4"]
        
        # Mock SearchManager
        mock_search_manager = Mock()
        mock_search_manager.run_research.return_value = [
            ("Agent 1 Report", "Agent 1 content"),
            ("Final Report", "This is the final research report"),
            ("References", ["Source 1", "Source 2"])
        ]
        mock_search_manager_class.return_value = mock_search_manager
        
        # Mock Streamlit components
        mock_st.sidebar = MagicMock()
        mock_st.text_input.return_value = ""
        mock_st.selectbox.return_value = "medium"
        mock_st.slider.return_value = 0.7
        mock_st.button.side_effect = [False, False, True]  # Start Research button clicked
        mock_st.text_area.return_value = "Research climate change impacts"
        
        research_interface()
        
        # Verify SearchManager was created and research was run
        mock_search_manager_class.assert_called_once()
        mock_search_manager.run_research.assert_called_once()
        mock_save_report.assert_called_once()
    
    @patch('ollama_workbench.workflows.research.st')
    @patch('ollama_workbench.workflows.research.get_all_reports')
    @patch('ollama_workbench.workflows.research.get_report_content')
    @patch('ollama_workbench.workflows.research.export_to_pdf')
    @patch('ollama_workbench.workflows.research.export_to_txt')
    @patch('ollama_workbench.workflows.research.delete_report')
    @patch('ollama_workbench.workflows.research.init_db')
    @patch('ollama_workbench.workflows.research.load_api_keys')
    @patch('ollama_workbench.workflows.research.load_research_model_settings')
    @patch('ollama_workbench.workflows.research.get_available_models')
    def test_saved_reports_management(self, mock_get_models, mock_load_settings, 
                                     mock_load_keys, mock_init_db, mock_delete_report,
                                     mock_export_txt, mock_export_pdf, mock_get_content,
                                     mock_get_reports, mock_st):
        """Test saved reports management functionality"""
        from ollama_workbench.workflows.research import research_interface
        
        # Setup mocks
        mock_load_keys.return_value = {}
        mock_load_settings.return_value = {}
        mock_get_models.return_value = ["llama3"]
        mock_get_reports.return_value = [
            (1, "Report 1", "2023-01-01"),
            (2, "Report 2", "2023-01-02")
        ]
        mock_get_content.return_value = "Report content here"
        mock_export_pdf.return_value = "/path/to/report.pdf"
        mock_export_txt.return_value = "/path/to/report.txt"
        
        # Mock Streamlit components
        mock_st.sidebar = MagicMock()
        mock_st.text_input.return_value = ""
        mock_st.selectbox.return_value = "short"
        mock_st.slider.return_value = 0.7
        mock_st.columns.return_value = [MagicMock() for _ in range(5)]
        # 3 top-level buttons + 4 buttons per report (2 reports); View clicked on report 1
        mock_st.button.side_effect = [False, False, False, True] + [False] * 7
        mock_st.text_area.return_value = ""

        research_interface()

        # Verify report management functions were called
        mock_get_reports.assert_called_once()
        mock_get_content.assert_called_with(1)
    
    @patch('ollama_workbench.workflows.research.st')
    @patch('ollama_workbench.workflows.research.get_all_reports')
    @patch('ollama_workbench.workflows.research.delete_report')
    @patch('ollama_workbench.workflows.research.init_db')
    @patch('ollama_workbench.workflows.research.load_api_keys')
    @patch('ollama_workbench.workflows.research.load_research_model_settings')
    @patch('ollama_workbench.workflows.research.get_available_models')
    def test_delete_report_functionality(self, mock_get_models, mock_load_settings, 
                                        mock_load_keys, mock_init_db, mock_delete_report,
                                        mock_get_reports, mock_st):
        """Test report deletion functionality"""
        from ollama_workbench.workflows.research import research_interface
        
        # Setup mocks
        mock_load_keys.return_value = {}
        mock_load_settings.return_value = {}
        mock_get_models.return_value = ["llama3"]
        mock_get_reports.return_value = [(1, "Report 1", "2023-01-01")]
        
        # Mock Streamlit components
        mock_st.sidebar = MagicMock()
        mock_st.text_input.return_value = ""
        mock_st.selectbox.return_value = "short"
        mock_st.slider.return_value = 0.7
        mock_st.columns.return_value = [MagicMock() for _ in range(5)]
        # 3 top-level buttons + view/pdf/txt/delete for the single report; Delete clicked
        mock_st.button.side_effect = [False] * 6 + [True]
        mock_st.text_area.return_value = ""
        mock_st.rerun = Mock()
        
        research_interface()
        
        # Verify delete was called
        mock_delete_report.assert_called_with(1)
        mock_st.rerun.assert_called_once()


class TestFileOperations:
    """Test file operations and directory management"""
    
    def test_files_directory_creation(self):
        """Test files directory creation on module import.

        The module is already imported by the time this test runs, so a plain
        re-import would be a no-op; reload the module with os.makedirs patched
        to observe the import-time side effect.
        """
        import importlib
        import ollama_workbench.workflows.research as research

        with patch('os.makedirs') as mock_makedirs:
            importlib.reload(research)

        expected_dir = os.path.join(
            os.path.dirname(os.path.abspath(research.__file__)), 'files'
        )
        mock_makedirs.assert_called_with(expected_dir, exist_ok=True)


class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    def test_full_database_workflow(self):
        """Test complete database workflow.

        The in-memory connection must be created BEFORE sqlite3.connect is
        patched (research.sqlite3 is the shared stdlib module, so the patch
        would otherwise intercept this test's own connect call too).
        """
        from ollama_workbench.workflows.research import init_db, save_report, get_all_reports, get_report_content, delete_report

        conn = sqlite3.connect(":memory:")
        try:
            with patch('ollama_workbench.workflows.research.sqlite3.connect', return_value=conn):
                # Initialize database
                init_db()

                # Save a report
                with patch('ollama_workbench.workflows.research.datetime') as mock_datetime:
                    mock_datetime.now.return_value.strftime.return_value = "2023-01-01 12:00:00"
                    save_report("Test Report", "Test Content")

                # Get all reports
                reports = get_all_reports()
                assert len(reports) == 1
                assert reports[0][1] == "Test Report"

                # Get specific report content
                content = get_report_content(reports[0][0])
                assert content == "Test Content"

                # Delete report
                delete_report(reports[0][0])
                reports_after_delete = get_all_reports()
                assert len(reports_after_delete) == 0
        finally:
            conn.close()
    
    def test_export_functions_integration(self):
        """Test export functions with realistic content"""
        from ollama_workbench.workflows.research import export_to_txt
        
        content = """Final Report:

This is a comprehensive research report on artificial intelligence.

References:
1. Smith, J. (2023). AI Advances. Tech Journal.
2. Brown, A. (2023). Machine Learning Trends. AI Review.

Search Results:

Agent 1:
Found information about AI trends in 2023.

Agent 2:
Collected data on machine learning applications."""
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('ollama_workbench.workflows.research.files_dir', tmp_dir):
                txt_path = export_to_txt(content, "test_report.txt")
                
                # Verify file was created and content is correct
                assert os.path.exists(txt_path)
                with open(txt_path, 'r', encoding='utf-8') as f:
                    saved_content = f.read()
                assert saved_content == content


class TestErrorHandling:
    """Test error handling scenarios"""
    
    @patch('ollama_workbench.workflows.research.sqlite3.connect')
    def test_database_connection_error(self, mock_connect):
        """Test database connection error handling"""
        from ollama_workbench.workflows.research import save_report
        
        mock_connect.side_effect = sqlite3.Error("Connection failed")
        
        # Should raise the error (no explicit error handling in current implementation)
        with pytest.raises(sqlite3.Error):
            save_report("Test", "Content")
    
    @patch('ollama_workbench.providers.ollama_utils.json.load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('ollama_workbench.providers.ollama_utils.os.path.exists')
    def test_json_loading_error(self, mock_exists, mock_open_fn, mock_json_load):
        """Test JSON loading error handling.

        load_api_keys lives in providers.ollama_utils (research re-exports it),
        so the patches must target that module. The read-through cache must be
        cleared or the cached value short-circuits the file read.
        """
        from ollama_workbench.workflows.research import load_api_keys
        import ollama_workbench.providers.ollama_utils as ou

        ou._api_keys_cache = None
        ou._api_keys_cache_time = 0

        mock_exists.return_value = True
        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        # Should raise the error (no explicit error handling in current implementation)
        with pytest.raises(json.JSONDecodeError):
            load_api_keys()
    
    @patch('ollama_workbench.workflows.research.os.path.join')
    @patch('builtins.open')
    def test_file_writing_error(self, mock_open, mock_join):
        """Test file writing error handling"""
        from ollama_workbench.workflows.research import export_to_txt
        
        mock_join.return_value = "/invalid/path/test.txt"
        mock_open.side_effect = IOError("Permission denied")
        
        # Should raise the error (no explicit error handling in current implementation)
        with pytest.raises(IOError):
            export_to_txt("content", "test.txt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
