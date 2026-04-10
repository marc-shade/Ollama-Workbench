"""
Test suite for file_management.py - File operations functionality
"""

import pytest
import os
import tempfile
import shutil
import time
from unittest.mock import Mock, patch, MagicMock, mock_open, call
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTokenCounting:
    """Test token counting functionality"""
    
    @patch('ollama_workbench.ui.file_management.tiktoken.get_encoding')
    def test_count_tokens_success(self, mock_get_encoding):
        """Test successful token counting"""
        from ollama_workbench.ui.file_management import count_tokens
        
        # Mock tiktoken encoding
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_get_encoding.return_value = mock_encoding
        
        result = count_tokens("Hello world test")
        
        assert result == 5
        mock_get_encoding.assert_called_once_with("cl100k_base")
        mock_encoding.encode.assert_called_once_with("Hello world test")
    
    @patch('ollama_workbench.ui.file_management.tiktoken.get_encoding')
    @patch('ollama_workbench.ui.file_management.logger')
    def test_count_tokens_tiktoken_error(self, mock_logger, mock_get_encoding):
        """Test token counting when tiktoken fails"""
        from ollama_workbench.ui.file_management import count_tokens
        
        mock_get_encoding.side_effect = Exception("tiktoken error")
        
        result = count_tokens("Hello world test")
        
        # Should fall back to rough estimate (length // 4)
        assert result == len("Hello world test") // 4
        mock_logger.error.assert_called_once()
    
    def test_count_tokens_empty_string(self):
        """Test token counting with empty string"""
        from ollama_workbench.ui.file_management import count_tokens
        
        with patch('ollama_workbench.ui.file_management.tiktoken.get_encoding') as mock_get_encoding:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = []
            mock_get_encoding.return_value = mock_encoding
            
            result = count_tokens("")
            
            assert result == 0
    
    @patch('ollama_workbench.ui.file_management.tiktoken.get_encoding')
    def test_count_tokens_long_text(self, mock_get_encoding):
        """Test token counting with long text"""
        from ollama_workbench.ui.file_management import count_tokens
        
        mock_encoding = Mock()
        mock_encoding.encode.return_value = list(range(1000))  # 1000 tokens
        mock_get_encoding.return_value = mock_encoding
        
        long_text = "Lorem ipsum " * 100
        result = count_tokens(long_text)
        
        assert result == 1000


class TestFileSplitting:
    """Test file splitting functionality"""
    
    def test_split_file_success(self):
        """Test successful file splitting"""
        from ollama_workbench.ui.file_management import split_file
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = os.path.join(temp_dir, "test.txt")
            test_content = "A" * 100  # 100 characters
            
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            # Split into 30-byte chunks
            chunk_files = split_file(test_file, 30, include_extension=True)
            
            # Should create 4 chunks (30+30+30+10)
            assert len(chunk_files) == 4
            
            # Verify chunk files exist and have correct content
            total_content = ""
            for chunk_file in chunk_files:
                assert os.path.exists(chunk_file)
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    total_content += f.read()
            
            assert total_content == test_content
    
    def test_split_file_with_extension(self):
        """Test file splitting with extension preservation"""
        from ollama_workbench.ui.file_management import split_file
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("A" * 50)
            
            chunk_files = split_file(test_file, 20, include_extension=True)
            
            # Check that all chunk files have the .txt extension
            for chunk_file in chunk_files:
                assert chunk_file.endswith('.txt')
                assert '.part' in chunk_file
    
    def test_split_file_without_extension(self):
        """Test file splitting without extension preservation"""
        from ollama_workbench.ui.file_management import split_file
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("A" * 50)
            
            chunk_files = split_file(test_file, 20, include_extension=False)
            
            # Check that chunk files don't have the .txt extension
            for chunk_file in chunk_files:
                assert not chunk_file.endswith('.txt')
                assert '.part' in chunk_file
    
    def test_split_file_encoding_fallback(self):
        """Test file splitting with encoding fallback"""
        from ollama_workbench.ui.file_management import split_file
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            
            # Create file with latin-1 encoding
            with open(test_file, 'w', encoding='latin-1') as f:
                f.write("Héllo wörld")
            
            chunk_files = split_file(test_file, 5, include_extension=True)
            
            # Should successfully split even with different encoding
            assert len(chunk_files) > 0
    
    def test_split_file_nonexistent(self):
        """Test splitting non-existent file"""
        from ollama_workbench.ui.file_management import split_file
        
        with patch('ollama_workbench.ui.file_management.logger') as mock_logger:
            result = split_file("/nonexistent/file.txt", 100)
            
            assert result == []
            mock_logger.error.assert_called_once()
    
    def test_split_file_permission_error(self):
        """Test splitting file with permission error"""
        from ollama_workbench.ui.file_management import split_file
        
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with patch('ollama_workbench.ui.file_management.logger') as mock_logger:
                result = split_file("/restricted/file.txt", 100)
                
                assert result == []
                mock_logger.error.assert_called_once()
    
    @patch('ollama_workbench.ui.file_management.split_pdf_file')
    def test_split_pdf_file_fallback(self, mock_split_pdf):
        """Test PDF file splitting fallback"""
        from ollama_workbench.ui.file_management import split_file
        
        mock_split_pdf.return_value = ["/path/chunk1.pdf", "/path/chunk2.pdf"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_file = os.path.join(temp_dir, "test.pdf")
            
            # Create a dummy PDF file (binary content)
            with open(pdf_file, 'wb') as f:
                f.write(b"binary content")
            
            # Mock the encoding failures to trigger PDF fallback
            with patch('builtins.open', side_effect=[
                UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte"),
                UnicodeDecodeError("latin-1", b"", 0, 1, "invalid start byte"),
                UnicodeDecodeError("windows-1252", b"", 0, 1, "invalid start byte"),
                UnicodeDecodeError("ascii", b"", 0, 1, "invalid start byte")
            ]):
                result = split_file(pdf_file, 100)
            
            mock_split_pdf.assert_called_once_with(pdf_file, 100, True)
            assert result == ["/path/chunk1.pdf", "/path/chunk2.pdf"]


class TestPDFSplitting:
    """Test PDF splitting functionality"""
    
    @patch('ollama_workbench.ui.file_management.logger')
    def test_split_pdf_file_stub(self, mock_logger):
        """Test PDF splitting stub implementation"""
        from ollama_workbench.ui.file_management import split_pdf_file
        
        result = split_pdf_file("/path/test.pdf", 1000, True)
        
        assert result == []
        mock_logger.warning.assert_called_once()
        assert "PDF splitting not fully implemented" in mock_logger.warning.call_args[0][0]


class TestFileMetadata:
    """Test file metadata functionality"""
    
    def test_get_file_metadata_success(self):
        """Test successful file metadata retrieval"""
        from ollama_workbench.ui.file_management import get_file_metadata
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("Test content")
            temp_file_path = temp_file.name
        
        try:
            metadata = get_file_metadata(temp_file_path)
            
            assert "name" in metadata
            assert "path" in metadata
            assert "size_bytes" in metadata
            assert "size_human" in metadata
            assert "type" in metadata
            assert "modified" in metadata
            
            assert metadata["path"] == temp_file_path
            assert metadata["name"] == os.path.basename(temp_file_path)
            assert metadata["type"] == "txt"
            assert metadata["size_bytes"] > 0
            assert "B" in metadata["size_human"] or "KB" in metadata["size_human"]
            
        finally:
            os.unlink(temp_file_path)
    
    def test_get_file_metadata_large_file(self):
        """Test metadata for large file size formatting"""
        from ollama_workbench.ui.file_management import get_file_metadata
        
        # Mock a large file
        with patch('ollama_workbench.ui.file_management.os.path.getsize', return_value=1024*1024*5):  # 5MB
            with patch('ollama_workbench.ui.file_management.os.path.getmtime', return_value=time.time()):
                with patch('ollama_workbench.ui.file_management.os.path.splitext', return_value=('/path/large', '.bin')):
                    with patch('ollama_workbench.ui.file_management.os.path.basename', return_value='large.bin'):
                        metadata = get_file_metadata('/path/large.bin')
        
        assert "MB" in metadata["size_human"]
        assert metadata["type"] == "bin"
    
    def test_get_file_metadata_very_large_file(self):
        """Test metadata for very large file (GB)"""
        from ollama_workbench.ui.file_management import get_file_metadata
        
        # Mock a 2GB file
        with patch('ollama_workbench.ui.file_management.os.path.getsize', return_value=1024*1024*1024*2):
            with patch('ollama_workbench.ui.file_management.os.path.getmtime', return_value=time.time()):
                with patch('ollama_workbench.ui.file_management.os.path.splitext', return_value=('/path/huge', '.data')):
                    with patch('ollama_workbench.ui.file_management.os.path.basename', return_value='huge.data'):
                        metadata = get_file_metadata('/path/huge.data')
        
        assert "GB" in metadata["size_human"]
        assert float(metadata["size_human"].split()[0]) >= 2.0
    
    def test_get_file_metadata_zero_size(self):
        """Test metadata for zero-size file"""
        from ollama_workbench.ui.file_management import get_file_metadata
        
        with tempfile.NamedTemporaryFile(suffix='.empty', delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        try:
            metadata = get_file_metadata(temp_file_path)
            
            assert metadata["size_bytes"] == 0
            assert metadata["size_human"] == "0.00 B"
            
        finally:
            os.unlink(temp_file_path)
    
    def test_get_file_metadata_nonexistent_file(self):
        """Test metadata for non-existent file"""
        from ollama_workbench.ui.file_management import get_file_metadata
        
        with patch('ollama_workbench.ui.file_management.logger') as mock_logger:
            metadata = get_file_metadata("/nonexistent/file.txt")
            
            assert "error" in metadata
            assert "name" in metadata
            assert "path" in metadata
            mock_logger.error.assert_called_once()
    
    def test_get_file_metadata_permission_error(self):
        """Test metadata with permission error"""
        from ollama_workbench.ui.file_management import get_file_metadata
        
        with patch('ollama_workbench.ui.file_management.os.path.getsize', side_effect=PermissionError("Access denied")):
            with patch('ollama_workbench.ui.file_management.logger') as mock_logger:
                metadata = get_file_metadata("/restricted/file.txt")
                
                assert "error" in metadata
                mock_logger.error.assert_called_once()
    
    def test_get_file_metadata_time_formatting(self):
        """Test metadata time formatting"""
        from ollama_workbench.ui.file_management import get_file_metadata
        
        # Mock specific timestamp
        test_timestamp = 1640995200  # 2022-01-01 00:00:00 UTC
        
        with patch('ollama_workbench.ui.file_management.os.path.getsize', return_value=100):
            with patch('ollama_workbench.ui.file_management.os.path.getmtime', return_value=test_timestamp):
                with patch('ollama_workbench.ui.file_management.os.path.splitext', return_value=('/path/test', '.txt')):
                    with patch('ollama_workbench.ui.file_management.os.path.basename', return_value='test.txt'):
                        metadata = get_file_metadata('/path/test.txt')
        
        # Should have formatted timestamp
        assert "modified" in metadata
        assert "2022" in metadata["modified"] or "2021" in metadata["modified"]  # Depends on timezone


class TestStreamlitInterface:
    """Test Streamlit interface functionality"""
    
    @pytest.fixture
    def mock_streamlit(self):
        """Mock Streamlit components"""
        with patch('ollama_workbench.ui.file_management.st') as mock_st:
            mock_st.title = Mock()
            mock_st.write = Mock()
            mock_st.button = Mock(return_value=False)
            mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock(), Mock()])
            mock_st.text_area = Mock()
            mock_st.error = Mock()
            mock_st.success = Mock()
            mock_st.warning = Mock()
            mock_st.rerun = Mock()
            mock_st.file_uploader = Mock(return_value=None)
            mock_st.selectbox = Mock(return_value=None)
            mock_st.slider = Mock(return_value=20)
            mock_st.checkbox = Mock(return_value=True)
            mock_st.subheader = Mock()
            mock_st.download_button = Mock()
            mock_st.session_state = {}
            yield mock_st
    
    @patch('ollama_workbench.ui.file_management.os.path.exists')
    @patch('ollama_workbench.ui.file_management.os.makedirs')
    @patch('ollama_workbench.ui.file_management.os.listdir')
    @patch('ollama_workbench.ui.file_management.os.path.isfile')
    def test_files_tab_basic_setup(self, mock_isfile, mock_listdir, mock_makedirs, 
                                  mock_exists, mock_streamlit):
        """Test basic files tab setup"""
        from ollama_workbench.ui.file_management import files_tab
        
        mock_exists.return_value = False  # files folder doesn't exist
        mock_listdir.return_value = []
        mock_isfile.return_value = True
        
        files_tab()
        
        mock_streamlit.title.assert_called_once_with("📂 Files")
        mock_makedirs.assert_called_once_with("files")
    
    @patch('ollama_workbench.ui.file_management.os.path.exists', return_value=True)
    @patch('ollama_workbench.ui.file_management.os.listdir')
    @patch('ollama_workbench.ui.file_management.os.path.isfile')
    def test_files_tab_with_files(self, mock_isfile, mock_listdir, mock_exists, mock_streamlit):
        """Test files tab with existing files"""
        from ollama_workbench.ui.file_management import files_tab
        
        mock_listdir.return_value = ['test.txt', 'document.pdf', 'image.jpg']
        mock_isfile.return_value = True
        
        files_tab()
        
        # Should display files
        assert mock_streamlit.write.call_count >= 3  # At least one call per file
        assert mock_streamlit.button.call_count >= 9  # At least 3 buttons per file
    
    @patch('ollama_workbench.ui.file_management.os.path.exists', return_value=True)
    @patch('ollama_workbench.ui.file_management.os.listdir')
    @patch('ollama_workbench.ui.file_management.os.path.isfile')
    def test_files_tab_file_filtering(self, mock_isfile, mock_listdir, mock_exists, mock_streamlit):
        """Test file filtering by allowed extensions"""
        from ollama_workbench.ui.file_management import files_tab
        
        # Mix of allowed and disallowed files
        mock_listdir.return_value = [
            'allowed.txt', 'allowed.pdf', 'allowed.jpg',
            'disallowed.exe', 'disallowed.dll'
        ]
        mock_isfile.return_value = True
        
        files_tab()
        
        # Should only process allowed files
        write_calls = [call[0][0] for call in mock_streamlit.write.call_args_list]
        assert 'allowed.txt' in write_calls
        assert 'allowed.pdf' in write_calls
        assert 'allowed.jpg' in write_calls
        assert 'disallowed.exe' not in write_calls
        assert 'disallowed.dll' not in write_calls
    
    @patch('ollama_workbench.ui.file_management.os.path.exists', return_value=True)
    @patch('ollama_workbench.ui.file_management.os.listdir', side_effect=PermissionError("Access denied"))
    def test_files_tab_directory_error(self, mock_listdir, mock_exists, mock_streamlit):
        """Test files tab with directory access error"""
        from ollama_workbench.ui.file_management import files_tab
        
        files_tab()
        
        mock_streamlit.error.assert_called_once()
        error_message = mock_streamlit.error.call_args[0][0]
        assert "Error reading files directory" in error_message
    
    @patch('ollama_workbench.ui.file_management.os.path.exists', return_value=True)
    @patch('ollama_workbench.ui.file_management.os.listdir', return_value=['test.txt'])
    @patch('ollama_workbench.ui.file_management.os.path.isfile', return_value=True)
    def test_files_tab_view_file(self, mock_isfile, mock_listdir, mock_exists, mock_streamlit):
        """Test viewing a file"""
        from ollama_workbench.ui.file_management import files_tab
        
        # Mock file view button being clicked
        mock_streamlit.session_state = {'view_test.txt': True}
        
        with patch('builtins.open', mock_open(read_data="File content here")):
            files_tab()
        
        # Should show file content in text area
        mock_streamlit.text_area.assert_called()
        text_area_call = mock_streamlit.text_area.call_args
        assert "File content here" in str(text_area_call)
    
    @patch('ollama_workbench.ui.file_management.os.path.exists', return_value=True)
    @patch('ollama_workbench.ui.file_management.os.listdir', return_value=['test.txt'])
    @patch('ollama_workbench.ui.file_management.os.path.isfile', return_value=True)
    @patch('ollama_workbench.ui.file_management.os.remove')
    def test_files_tab_edit_file(self, mock_remove, mock_isfile, mock_listdir, mock_exists, mock_streamlit):
        """Test editing a file"""
        from ollama_workbench.ui.file_management import files_tab

        # Mock file edit button being clicked
        mock_streamlit.session_state = {'edit_test.txt': True}
        mock_streamlit.text_area.return_value = "Modified content"
        # Only save button should trigger
        def button_side_effect(*args, **kwargs):
            if args and "Save Changes" in str(args[0]):
                return True
            return False
        mock_streamlit.button.side_effect = button_side_effect

        with patch('builtins.open', mock_open(read_data="Original content")) as mock_file:
            files_tab()

        # Should read and write file
        assert mock_file.call_count >= 2  # At least read and write
        mock_streamlit.success.assert_called()
    
    @patch('ollama_workbench.ui.file_management.os.path.exists', return_value=True)
    @patch('ollama_workbench.ui.file_management.os.listdir', return_value=['test.pdf'])
    @patch('ollama_workbench.ui.file_management.os.path.isfile', return_value=True)
    def test_files_tab_download_pdf(self, mock_isfile, mock_listdir, mock_exists, mock_streamlit):
        """Test downloading a PDF file"""
        from ollama_workbench.ui.file_management import files_tab
        
        # Mock PDF download button being clicked
        mock_streamlit.session_state = {'download_test.pdf': True}
        
        with patch('builtins.open', mock_open(read_data=b"PDF content")) as mock_file:
            files_tab()
        
        # Should create download button
        mock_streamlit.download_button.assert_called()
        download_call = mock_streamlit.download_button.call_args
        assert download_call[1]['mime'] == 'application/pdf'
    
    @patch('ollama_workbench.ui.file_management.os.path.exists', return_value=True)
    @patch('ollama_workbench.ui.file_management.os.listdir', return_value=['test.txt'])
    @patch('ollama_workbench.ui.file_management.os.path.isfile', return_value=True)
    @patch('ollama_workbench.ui.file_management.os.remove')
    def test_files_tab_delete_file(self, mock_remove, mock_isfile, mock_listdir, 
                                  mock_exists, mock_streamlit):
        """Test deleting a file"""
        from ollama_workbench.ui.file_management import files_tab
        
        # Mock file delete button being clicked
        mock_streamlit.session_state = {'delete_test.txt': True}
        
        files_tab()
        
        # Should remove file and show success
        mock_remove.assert_called_once_with('files/test.txt')
        mock_streamlit.success.assert_called()
        mock_streamlit.rerun.assert_called()
    
    @patch('ollama_workbench.ui.file_management.os.path.exists', return_value=True)
    @patch('ollama_workbench.ui.file_management.os.listdir', return_value=[])
    @patch('ollama_workbench.ui.file_management.os.path.isfile', return_value=True)
    @patch('ollama_workbench.ui.file_management.os.path.realpath', side_effect=lambda p: os.path.abspath(p))
    def test_files_tab_file_upload(self, mock_realpath, mock_isfile, mock_listdir, mock_exists, mock_streamlit):
        """Test file upload functionality"""
        from ollama_workbench.ui.file_management import files_tab

        # Mock file upload
        mock_uploaded_file = Mock()
        mock_uploaded_file.name = "uploaded.txt"
        mock_uploaded_file.getbuffer.return_value = b"Uploaded content"
        mock_streamlit.file_uploader.return_value = mock_uploaded_file

        with patch('builtins.open', mock_open()) as mock_file:
            files_tab()

        # Should save uploaded file (os.path.basename is applied to the name)
        mock_file.assert_called_with('files/uploaded.txt', 'wb')
        mock_file().write.assert_called_with(b"Uploaded content")
        mock_streamlit.success.assert_called()
        mock_streamlit.rerun.assert_called()
    
    @patch('ollama_workbench.ui.file_management.os.path.exists', return_value=True)
    @patch('ollama_workbench.ui.file_management.os.listdir', return_value=['test.txt', 'doc.md'])
    @patch('ollama_workbench.ui.file_management.os.path.isfile', return_value=True)
    @patch('ollama_workbench.ui.file_management.split_file')
    @patch('ollama_workbench.ui.file_management.os.remove')
    def test_files_tab_file_splitting(self, mock_remove, mock_split_file, mock_isfile, mock_listdir,
                                     mock_exists, mock_streamlit):
        """Test file splitting functionality"""
        from ollama_workbench.ui.file_management import files_tab

        mock_streamlit.selectbox.return_value = "test.txt"
        mock_streamlit.slider.return_value = 5  # 5MB
        mock_streamlit.checkbox.return_value = True
        # Only the split button should trigger; use side_effect to control
        # Buttons are called for: per-file view/edit/delete (2 files x ~3 buttons = ~6),
        # then per-file save (skipped), then split button
        def button_side_effect(*args, **kwargs):
            if args and "Split File" in str(args[0]):
                return True
            return False
        mock_streamlit.button.side_effect = button_side_effect

        mock_split_file.return_value = ["chunk1.txt", "chunk2.txt"]
        mock_streamlit.file_uploader.return_value = None

        files_tab()

        # Should call split_file with correct parameters
        mock_split_file.assert_called_once_with('files/test.txt', 5*1024*1024, True)
        mock_streamlit.success.assert_called()
        mock_streamlit.rerun.assert_called()
    
    @patch('ollama_workbench.ui.file_management.os.path.exists', return_value=True)
    @patch('ollama_workbench.ui.file_management.os.listdir', return_value=[])
    @patch('ollama_workbench.ui.file_management.os.path.isfile', return_value=True)
    def test_files_tab_split_no_file_selected(self, mock_isfile, mock_listdir,
                                             mock_exists, mock_streamlit):
        """Test file splitting with no file selected"""
        from ollama_workbench.ui.file_management import files_tab

        mock_streamlit.selectbox.return_value = None
        # Only the split button should trigger
        def button_side_effect(*args, **kwargs):
            if args and "Split File" in str(args[0]):
                return True
            return False
        mock_streamlit.button.side_effect = button_side_effect

        files_tab()

        # Should show warning
        mock_streamlit.warning.assert_called_with("Please select a file to split.")


class TestFileOperationErrors:
    """Test file operation error handling"""
    
    @patch('ollama_workbench.ui.file_management.os.path.exists', return_value=True)
    @patch('ollama_workbench.ui.file_management.os.listdir', return_value=['corrupted.txt'])
    @patch('ollama_workbench.ui.file_management.os.path.isfile', return_value=True)
    def test_files_tab_view_unicode_error(self, mock_isfile, mock_listdir, mock_exists):
        """Test viewing file with unicode decode error"""
        from ollama_workbench.ui.file_management import files_tab

        with patch('ollama_workbench.ui.file_management.st') as mock_st:
            mock_st.title = Mock()
            mock_st.write = Mock()
            mock_st.button = Mock(return_value=False)
            mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock(), Mock()])
            mock_st.error = Mock()
            mock_st.file_uploader = Mock(return_value=None)
            mock_st.selectbox = Mock(return_value=None)
            mock_st.slider = Mock(return_value=20)
            mock_st.checkbox = Mock(return_value=True)
            mock_st.subheader = Mock()
            mock_st.session_state = {'view_corrupted.txt': True}

            with patch('builtins.open', side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")):
                files_tab()

            mock_st.error.assert_called()
            error_message = mock_st.error.call_args[0][0]
            assert "Unable to decode file" in error_message


class TestIntegration:
    """Test integration scenarios"""
    
    def test_module_imports(self):
        """Test that all required modules can be imported"""
        import ollama_workbench.ui.file_management as file_management

        
        # Test that main functions exist
        assert hasattr(file_management, 'count_tokens')
        assert hasattr(file_management, 'split_file')
        assert hasattr(file_management, 'split_pdf_file')
        assert hasattr(file_management, 'get_file_metadata')
        assert hasattr(file_management, 'files_tab')
    
    def test_file_splitting_integration(self):
        """Test complete file splitting workflow"""
        from ollama_workbench.ui.file_management import split_file, get_file_metadata
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = os.path.join(temp_dir, "integration_test.txt")
            test_content = "This is a test file for integration testing. " * 10
            
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            # Get original metadata
            original_metadata = get_file_metadata(test_file)
            assert original_metadata["size_bytes"] > 0
            
            # Split file
            chunk_files = split_file(test_file, 100, include_extension=True)
            assert len(chunk_files) > 1
            
            # Verify all chunks have metadata
            total_size = 0
            for chunk_file in chunk_files:
                chunk_metadata = get_file_metadata(chunk_file)
                assert "error" not in chunk_metadata
                total_size += chunk_metadata["size_bytes"]
            
            # Total size should approximately match (UTF-8 encoding might differ slightly)
            assert abs(total_size - original_metadata["size_bytes"]) <= len(chunk_files)
    
    def test_logging_integration(self):
        """Test logging integration across functions"""
        from ollama_workbench.ui.file_management import split_file, get_file_metadata
        
        with patch('ollama_workbench.ui.file_management.logger') as mock_logger:
            # Test error logging in split_file
            split_file("/nonexistent/file.txt", 100)
            
            # Test error logging in get_file_metadata
            get_file_metadata("/nonexistent/metadata.txt")
            
            # Should have logged errors
            assert mock_logger.error.call_count >= 2


class TestPerformance:
    """Test performance-related functionality"""
    
    def test_large_file_splitting_efficiency(self):
        """Test splitting of large file"""
        from ollama_workbench.ui.file_management import split_file
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a moderately large test file (1KB)
            test_file = os.path.join(temp_dir, "large_test.txt")
            test_content = "A" * 1024  # 1KB file
            
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            # Split into small chunks
            start_time = time.time()
            chunk_files = split_file(test_file, 50, include_extension=True)  # 50-byte chunks
            end_time = time.time()
            
            # Should complete quickly (less than 1 second)
            assert end_time - start_time < 1.0
            assert len(chunk_files) > 10  # Should create many small chunks
    
    def test_metadata_performance(self):
        """Test metadata retrieval performance"""
        from ollama_workbench.ui.file_management import get_file_metadata
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple test files
            test_files = []
            for i in range(10):
                test_file = os.path.join(temp_dir, f"test_{i}.txt")
                with open(test_file, 'w') as f:
                    f.write(f"Content {i}")
                test_files.append(test_file)
            
            # Get metadata for all files
            start_time = time.time()
            for test_file in test_files:
                metadata = get_file_metadata(test_file)
                assert "error" not in metadata
            end_time = time.time()
            
            # Should complete quickly
            assert end_time - start_time < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
