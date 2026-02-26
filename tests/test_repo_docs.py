"""
Comprehensive tests for repo_docs.py module.

Tests all repository documentation functionality including:
- PDF generation with optional FPDF dependency
- Code analysis with optional radon/flake8 dependencies  
- Repository structure analysis
- Documentation generation with AI models
- File processing and multi-threading
- Streamlit UI components
- Error handling and edge cases
"""

import pytest
import json
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from unittest import TestCase
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock optional dependencies before importing repo_docs
with patch.dict('sys.modules', {
    'fpdf': Mock(),
    'radon.complexity': Mock(),
    'radon.metrics': Mock(), 
    'flake8.api.legacy': Mock()
}):
    from ollama_workbench.knowledge.repo_docs import (
        PDF, call_ollama_endpoint, get_available_models, generate_documentation_stream,
        run_pylint, run_phpstan, run_eslint, get_all_code_files, get_file_info,
        process_file_with_updates, analyze_repository_structure, generate_pdf,
        generate_requirements_file, generate_project_summary, write_file_details,
        load_model_settings, save_model_settings, main
    )


class TestPDFClass(TestCase):
    """Test PDF class functionality"""
    
    @patch('repo_docs.fpdf_available', True)
    def test_pdf_init_with_fpdf_available(self):
        """Test PDF initialization when FPDF is available"""
        with patch('repo_docs.FPDF.__init__', return_value=None):
            pdf = PDF()
            self.assertIsInstance(pdf, PDF)
    
    @patch('repo_docs.fpdf_available', False)
    def test_pdf_init_without_fpdf(self):
        """Test PDF initialization when FPDF is not available"""
        pdf = PDF()
        self.assertIsInstance(pdf, PDF)
    
    @patch('repo_docs.fpdf_available', True)
    def test_pdf_header_with_fpdf(self):
        """Test PDF header method when FPDF is available"""
        with patch('repo_docs.FPDF.__init__', return_value=None):
            pdf = PDF()
            pdf.set_font = Mock()
            pdf.cell = Mock()
            
            pdf.header()
            
            pdf.set_font.assert_called_once_with('Arial', 'B', 12)
            pdf.cell.assert_called_once_with(0, 10, 'Repository Analysis', 0, 1, 'C')
    
    @patch('repo_docs.fpdf_available', False)
    def test_pdf_header_without_fpdf(self):
        """Test PDF header method when FPDF is not available"""
        pdf = PDF()
        # Should not raise an exception
        pdf.header()
    
    @patch('repo_docs.fpdf_available', True)
    def test_pdf_add_chapter_with_fpdf(self):
        """Test PDF add_chapter method when FPDF is available"""
        with patch('repo_docs.FPDF.__init__', return_value=None):
            pdf = PDF()
            pdf.add_page = Mock()
            pdf.chapter_title = Mock()
            pdf.chapter_body = Mock()
            
            pdf.add_chapter("Test Title", "Test Body")
            
            pdf.add_page.assert_called_once()
            pdf.chapter_title.assert_called_once_with("Test Title")
            pdf.chapter_body.assert_called_once_with("Test Body")
    
    @patch('repo_docs.fpdf_available', False)
    def test_pdf_add_chapter_without_fpdf(self):
        """Test PDF add_chapter method when FPDF is not available"""
        pdf = PDF()
        # Should not raise an exception
        pdf.add_chapter("Test Title", "Test Body")


class TestOllamaEndpoint(TestCase):
    """Test Ollama API endpoint functionality"""
    
    @patch('requests.post')
    def test_call_ollama_endpoint_success(self, mock_post):
        """Test successful call to Ollama endpoint"""
        # Mock streaming response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [
            b'{"response": "Hello", "eval_count": 10, "eval_duration": 1000}',
            b'{"response": " World", "done": true}'
        ]
        mock_post.return_value = mock_response
        
        result, context, eval_count, eval_duration = call_ollama_endpoint(
            "test_model", "test prompt"
        )
        
        self.assertEqual(result, "Hello World")
        self.assertEqual(eval_count, 10)
        self.assertEqual(eval_duration, 1000)
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_call_ollama_endpoint_error(self, mock_post):
        """Test error handling in Ollama endpoint call"""
        mock_post.side_effect = Exception("Connection error")
        
        result, context, eval_count, eval_duration = call_ollama_endpoint(
            "test_model", "test prompt"
        )
        
        self.assertIn("Error calling Ollama endpoint", result)
        self.assertEqual(eval_count, 0)
        self.assertEqual(eval_duration, 0)
    
    @patch('requests.post')
    def test_call_ollama_endpoint_invalid_json(self, mock_post):
        """Test handling of invalid JSON in response"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [
            b'{"response": "Hello"}',
            b'invalid json line',
            b'{"response": " World", "done": true}'
        ]
        mock_post.return_value = mock_response
        
        result, context, eval_count, eval_duration = call_ollama_endpoint(
            "test_model", "test prompt"
        )
        
        self.assertEqual(result, "Hello World")


class TestModelFunctions(TestCase):
    """Test model-related functions"""
    
    @patch('repo_docs.get_ollama_models')
    @patch('repo_docs.OPENAI_MODELS', ['gpt-4'])
    @patch('repo_docs.GROQ_MODELS', ['llama-3'])
    def test_get_available_models(self, mock_get_ollama):
        """Test getting available models"""
        mock_get_ollama.return_value = ['mistral:instruct']
        
        models = get_available_models()
        
        expected_models = ['mistral:instruct', 'gpt-4', 'llama-3']
        self.assertEqual(models, expected_models)


class TestDocumentationGeneration(TestCase):
    """Test documentation generation functionality"""
    
    @patch('repo_docs.load_api_keys')
    @patch('requests.post')
    def test_generate_documentation_stream_ollama(self, mock_post, mock_load_keys):
        """Test documentation generation with Ollama model"""
        mock_load_keys.return_value = {}
        
        # Mock streaming response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [
            b'{"response": "Generated "}',
            b'{"response": "documentation", "done": true}'
        ]
        mock_post.return_value.__enter__ = Mock(return_value=mock_response)
        mock_post.return_value.__exit__ = Mock(return_value=None)
        
        # Collect all chunks
        chunks = list(generate_documentation_stream(
            "def hello(): pass", "documentation", "mistral:instruct", 0.7, 1000
        ))
        
        self.assertEqual(chunks, ["Generated ", "documentation"])
    
    @patch('repo_docs.load_api_keys')
    @patch('repo_docs.call_openai_api')
    @patch('repo_docs.OPENAI_MODELS', ['gpt-4'])
    def test_generate_documentation_stream_openai(self, mock_openai_call, mock_load_keys):
        """Test documentation generation with OpenAI model"""
        mock_load_keys.return_value = {'openai_api_key': 'test_key'}
        mock_openai_call.return_value = "OpenAI generated documentation"
        
        chunks = list(generate_documentation_stream(
            "def hello(): pass", "documentation", "gpt-4", 0.7, 1000
        ))
        
        self.assertEqual(chunks, ["OpenAI generated documentation"])
        mock_openai_call.assert_called_once()
    
    @patch('repo_docs.load_api_keys')
    @patch('repo_docs.call_groq_api')
    @patch('repo_docs.GROQ_MODELS', ['llama-3'])
    def test_generate_documentation_stream_groq(self, mock_groq_call, mock_load_keys):
        """Test documentation generation with Groq model"""
        mock_load_keys.return_value = {'groq_api_key': 'test_key'}
        mock_groq_call.return_value = "Groq generated documentation"
        
        chunks = list(generate_documentation_stream(
            "def hello(): pass", "documentation", "llama-3", 0.7, 1000
        ))
        
        self.assertEqual(chunks, ["Groq generated documentation"])
        mock_groq_call.assert_called_once()
    
    def test_generate_documentation_stream_requirements_task(self):
        """Test documentation generation with requirements task type"""
        result = generate_documentation_stream(
            "some content", "requirements", "model", 0.7, 1000
        )
        
        self.assertIsNone(result)


class TestCodeAnalysisTools(TestCase):
    """Test code analysis tool functions"""
    
    @patch('subprocess.run')
    def test_run_pylint(self, mock_run):
        """Test pylint execution"""
        mock_run.return_value.stdout = "pylint output"
        
        result = run_pylint("/path/to/file.py")
        
        self.assertEqual(result, "pylint output")
        mock_run.assert_called_once_with(['pylint', '/path/to/file.py'], capture_output=True, text=True)
    
    @patch('subprocess.run')
    def test_run_phpstan(self, mock_run):
        """Test phpstan execution"""
        mock_run.return_value.stdout = "phpstan output"
        
        result = run_phpstan("/path/to/file.php")
        
        self.assertEqual(result, "phpstan output")
        mock_run.assert_called_once_with(['phpstan', 'analyse', '/path/to/file.php'], capture_output=True, text=True)
    
    @patch('subprocess.run')
    def test_run_eslint(self, mock_run):
        """Test eslint execution"""
        mock_run.return_value.stdout = "eslint output"
        
        result = run_eslint("/path/to/file.js")
        
        self.assertEqual(result, "eslint output")
        mock_run.assert_called_once_with(['eslint', '/path/to/file.js'], capture_output=True, text=True)


class TestFileOperations(TestCase):
    """Test file operation functions"""
    
    def setUp(self):
        """Set up temporary directory for testing"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_get_all_code_files(self):
        """Test getting all code files from directory"""
        # Create test files
        py_file = os.path.join(self.temp_dir, "test.py")
        js_file = os.path.join(self.temp_dir, "test.js")
        txt_file = os.path.join(self.temp_dir, "test.txt")
        
        with open(py_file, 'w') as f:
            f.write("print('hello')")
        with open(js_file, 'w') as f:
            f.write("console.log('hello')")
        with open(txt_file, 'w') as f:
            f.write("hello")
        
        code_files = get_all_code_files(self.temp_dir, [])
        
        # Should include .py and .js but not .txt
        self.assertIn(py_file, code_files)
        self.assertIn(js_file, code_files)
        self.assertNotIn(txt_file, code_files)
    
    def test_get_all_code_files_with_exclusions(self):
        """Test getting code files with exclusion patterns"""
        # Create test files
        py_file = os.path.join(self.temp_dir, "test.py")
        excluded_file = os.path.join(self.temp_dir, "exclude_me.py")
        
        with open(py_file, 'w') as f:
            f.write("print('hello')")
        with open(excluded_file, 'w') as f:
            f.write("print('excluded')")
        
        code_files = get_all_code_files(self.temp_dir, ["exclude_me"])
        
        self.assertIn(py_file, code_files)
        self.assertNotIn(excluded_file, code_files)
    
    @patch('repo_docs.radon_available', True)
    @patch('repo_docs.flake8_available', True)
    @patch('repo_docs.cc_visit')
    @patch('repo_docs.mi_visit')
    @patch('repo_docs.h_visit')
    @patch('repo_docs.flake8.get_style_guide')
    def test_get_file_info_python_with_analysis(self, mock_style_guide, mock_h_visit, 
                                               mock_mi_visit, mock_cc_visit):
        """Test getting file info for Python file with code analysis"""
        # Create test Python file
        py_file = os.path.join(self.temp_dir, "test.py")
        with open(py_file, 'w') as f:
            f.write("def hello():\n    print('hello')\n")
        
        # Mock analysis results
        mock_complexity = Mock()
        mock_complexity.name = "hello"
        mock_complexity.complexity = 1
        mock_cc_visit.return_value = [mock_complexity]
        mock_mi_visit.return_value = 85.5
        mock_h_visit.return_value = {"volume": 10, "difficulty": 2}
        
        mock_report = Mock()
        mock_report.get_statistics.return_value = ["E302 expected 2 blank lines"]
        mock_style_guide.return_value.check_files.return_value = mock_report
        
        file_info = get_file_info(py_file)
        
        self.assertEqual(file_info["Language"], "python")
        self.assertIn("Complexity", file_info)
        self.assertIn("Style Violations", file_info)
        self.assertIn("Code", file_info)
    
    @patch('repo_docs.radon_available', False)
    @patch('repo_docs.flake8_available', False)
    def test_get_file_info_python_without_analysis(self):
        """Test getting file info for Python file without analysis tools"""
        # Create test Python file
        py_file = os.path.join(self.temp_dir, "test.py")
        with open(py_file, 'w') as f:
            f.write("def hello():\n    print('hello')\n")
        
        file_info = get_file_info(py_file)
        
        self.assertEqual(file_info["Language"], "python")
        self.assertIn("not available", file_info["Complexity"]["Note"])
        self.assertIn("not available", file_info["Style Violations"][0])
    
    def test_get_file_info_nonexistent_file(self):
        """Test getting file info for nonexistent file"""
        file_info = get_file_info("/nonexistent/file.py")
        
        self.assertIsNone(file_info)
    
    def test_get_file_info_javascript(self):
        """Test getting file info for JavaScript file"""
        js_file = os.path.join(self.temp_dir, "test.js")
        with open(js_file, 'w') as f:
            f.write("console.log('hello');")
        
        with patch('repo_docs.run_eslint') as mock_eslint:
            mock_eslint.return_value = "eslint output"
            
            file_info = get_file_info(js_file)
            
            self.assertEqual(file_info["Language"], "javascript")
            self.assertEqual(file_info["ESLint Report"], "eslint output")


class TestRepositoryAnalysis(TestCase):
    """Test repository analysis functions"""
    
    def setUp(self):
        """Set up temporary repository for testing"""
        self.repo_dir = tempfile.mkdtemp()
        self.code_files = []
    
    def tearDown(self):
        """Clean up temporary repository"""
        shutil.rmtree(self.repo_dir)
    
    def test_analyze_repository_structure_with_readme(self):
        """Test analyzing repository structure with existing README"""
        # Create README.md
        readme_path = os.path.join(self.repo_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write("# Test Repository\n\nThis is a test.")
        
        # Create requirements.txt
        req_path = os.path.join(self.repo_dir, "requirements.txt")
        with open(req_path, 'w') as f:
            f.write("numpy==1.21.0\npandas==1.3.0")
        
        # Create main.py
        main_path = os.path.join(self.repo_dir, "main.py")
        with open(main_path, 'w') as f:
            f.write("def main():\n    print('hello')")
        
        code_files = [readme_path, req_path, main_path]
        repo_info = analyze_repository_structure(self.repo_dir, code_files)
        
        self.assertEqual(repo_info['existing_readme'], "# Test Repository\n\nThis is a test.")
        self.assertEqual(repo_info['existing_readme_path'], readme_path)
        self.assertIn("numpy==1.21.0", repo_info['requirements'])
        self.assertIn(main_path, repo_info['entry_points'])
        self.assertIn('main.py', repo_info['key_files'])
    
    def test_analyze_repository_structure_no_readme(self):
        """Test analyzing repository structure without README"""
        # Create only main.py
        main_path = os.path.join(self.repo_dir, "main.py")
        with open(main_path, 'w') as f:
            f.write("def main():\n    print('hello')")
        
        code_files = [main_path]
        repo_info = analyze_repository_structure(self.repo_dir, code_files)
        
        self.assertIsNone(repo_info['existing_readme'])
        self.assertIsNone(repo_info['existing_readme_path'])
        self.assertIn(main_path, repo_info['entry_points'])


class TestPDFGeneration(TestCase):
    """Test PDF generation functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.test_results = [
            ("/test/file1.py", "Documentation for file1", "Pylint report", "def hello(): pass"),
            ("/test/file2.py", "Documentation for file2", "", "def world(): pass")
        ]
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir)
    
    @patch('repo_docs.fpdf_available', False)
    def test_generate_pdf_without_fpdf(self):
        """Test PDF generation when FPDF is not available"""
        output_path = os.path.join(self.temp_dir, "test.pdf")
        
        result_path = generate_pdf(self.test_results, output_path, "documentation")
        
        # Should create a text file instead
        self.assertTrue(result_path.endswith('.txt'))
        self.assertTrue(os.path.exists(result_path))
        
        with open(result_path, 'r') as f:
            content = f.read()
            self.assertIn("Repository Analysis Report", content)
            self.assertIn("file1.py", content)
    
    @patch('repo_docs.fpdf_available', True)
    def test_generate_pdf_with_fpdf(self):
        """Test PDF generation when FPDF is available"""
        output_path = os.path.join(self.temp_dir, "test.pdf")
        
        with patch('repo_docs.PDF') as mock_pdf_class:
            mock_pdf = Mock()
            mock_pdf_class.return_value = mock_pdf
            
            result_path = generate_pdf(self.test_results, output_path, "documentation")
            
            self.assertEqual(result_path, output_path)
            mock_pdf.add_chapter.assert_called()
            mock_pdf.output.assert_called_once_with(output_path, 'F')
    
    @patch('repo_docs.fpdf_available', True)
    def test_generate_pdf_error_handling(self):
        """Test PDF generation error handling"""
        output_path = os.path.join(self.temp_dir, "test.pdf")
        
        with patch('repo_docs.PDF') as mock_pdf_class:
            mock_pdf_class.side_effect = Exception("PDF generation failed")
            
            result_path = generate_pdf(self.test_results, output_path, "documentation")
            
            self.assertIsNone(result_path)


class TestRequirementsGeneration(TestCase):
    """Test requirements.txt generation"""
    
    def setUp(self):
        """Set up temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    @patch('subprocess.run')
    @patch('pkg_resources.working_set')
    @patch('pkg_resources.get_distribution')
    def test_generate_requirements_file(self, mock_get_dist, mock_working_set, mock_subprocess):
        """Test requirements file generation"""
        # Mock pipreqs output
        mock_subprocess.return_value = None
        
        # Mock installed packages
        mock_pkg = Mock()
        mock_pkg.key = 'numpy'
        mock_pkg.version = '1.21.0'
        mock_working_set.__iter__.return_value = [mock_pkg]
        
        mock_dist = Mock()
        mock_dist.version = '1.21.0'
        mock_get_dist.return_value = mock_dist
        
        # Create temporary requirements file that pipreqs would create
        temp_req = os.path.join(self.temp_dir, 'temp_requirements.txt')
        with open(temp_req, 'w') as f:
            f.write("numpy==1.21.0\n")
        
        with patch('builtins.open', mock_open(read_data="numpy==1.21.0\n")) as mock_file:
            result_path = generate_requirements_file(self.temp_dir, [])
            
            expected_path = os.path.join(self.temp_dir, 'requirements.txt')
            self.assertEqual(result_path, expected_path)


class TestProjectSummaryGeneration(TestCase):
    """Test project summary generation"""
    
    def setUp(self):
        """Set up temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_generate_project_summary(self):
        """Test project summary generation"""
        # Create test files
        py_file = os.path.join(self.temp_dir, "test.py")
        with open(py_file, 'w') as f:
            f.write("def hello(): pass")
        
        with patch('repo_docs.get_file_info') as mock_get_info:
            mock_get_info.return_value = {
                'Full Path': py_file,
                'Extension': '.py',
                'Language': 'python',
                'Size': 100,
                'Created': '2024-01-01 10:00:00',
                'Modified': '2024-01-01 10:00:00'
            }
            
            result_path = generate_project_summary(self.temp_dir, [])
            
            expected_path = os.path.join(self.temp_dir, 'project_summary.md')
            self.assertEqual(result_path, expected_path)
            self.assertTrue(os.path.exists(result_path))
    
    def test_write_file_details(self):
        """Test writing file details to summary"""
        file_info = {
            'Full Path': '/test/file.py',
            'Extension': '.py',
            'Language': 'python',
            'Size': 100,
            'Code': 'def hello(): pass',
            'Complexity': {
                'Cyclomatic Complexity': ['hello: 1 (A)'],
                'Maintainability Index': 85.5
            }
        }
        
        mock_file = Mock()
        write_file_details(mock_file, file_info)
        
        # Verify various writes were called
        mock_file.write.assert_called()
        calls = [call[0][0] for call in mock_file.write.call_args_list]
        
        # Should include file path, complexity info, and code
        file_header_found = any("## File:" in call for call in calls)
        complexity_found = any("Complexity" in call for call in calls)
        code_found = any("```python" in call for call in calls)
        
        self.assertTrue(file_header_found)
        self.assertTrue(complexity_found)
        self.assertTrue(code_found)


class TestModelSettings(TestCase):
    """Test model settings functions"""
    
    def setUp(self):
        """Set up temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up and restore directory"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_load_model_settings_no_file(self):
        """Test loading model settings when file doesn't exist"""
        settings = load_model_settings()
        
        # Should return default settings
        expected_defaults = {
            "model": "mistral:instruct",
            "temperature": 0.7,
            "max_tokens": 4000,
            "api_key": ""
        }
        
        for key, value in expected_defaults.items():
            self.assertEqual(settings[key], value)
    
    def test_load_model_settings_existing_file(self):
        """Test loading model settings from existing file"""
        test_settings = {
            "model": "gpt-4",
            "temperature": 0.5,
            "max_tokens": 2000,
            "api_key": "test_key"
        }
        
        with open("model_settings.json", 'w') as f:
            json.dump(test_settings, f)
        
        settings = load_model_settings()
        
        self.assertEqual(settings["model"], "gpt-4")
        self.assertEqual(settings["temperature"], 0.5)
    
    def test_load_model_settings_invalid_json(self):
        """Test loading model settings with invalid JSON"""
        with open("model_settings.json", 'w') as f:
            f.write("invalid json")
        
        with patch('streamlit.warning') as mock_warning:
            settings = load_model_settings()
            
            mock_warning.assert_called_once()
            # Should still return defaults
            self.assertEqual(settings["model"], "mistral:instruct")
    
    def test_save_model_settings(self):
        """Test saving model settings"""
        test_settings = {
            "model": "gpt-4",
            "temperature": 0.5,
            "max_tokens": 2000,
            "api_key": "test_key"
        }
        
        save_model_settings(test_settings)
        
        # Verify file was created and contains correct data
        self.assertTrue(os.path.exists("model_settings.json"))
        
        with open("model_settings.json", 'r') as f:
            saved_settings = json.load(f)
        
        self.assertEqual(saved_settings, test_settings)


class TestProcessFileWithUpdates(TestCase):
    """Test file processing with updates"""
    
    def setUp(self):
        """Set up test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.py")
        with open(self.test_file, 'w') as f:
            f.write("def hello(): pass")
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)
    
    @patch('repo_docs.generate_documentation_stream')
    @patch('repo_docs.run_pylint')
    def test_process_file_with_updates_success(self, mock_pylint, mock_generate):
        """Test successful file processing"""
        mock_generate.return_value = ["Generated ", "documentation"]
        mock_pylint.return_value = "Pylint output"
        
        update_queue = Mock()
        
        result = process_file_with_updates(
            self.test_file, "debug", "mistral", 0.7, 1000, "api_key",
            update_queue, None, None, None
        )
        
        file_path, documentation, pylint_report, file_content = result
        
        self.assertEqual(file_path, self.test_file)
        self.assertEqual(documentation, "Generated documentation")
        self.assertEqual(pylint_report, "Pylint output")
        self.assertEqual(file_content, "def hello(): pass")
    
    @patch('builtins.open', side_effect=UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid'))
    def test_process_file_with_updates_unicode_error(self, mock_open):
        """Test file processing with Unicode decode error"""
        update_queue = Mock()
        
        result = process_file_with_updates(
            self.test_file, "documentation", "mistral", 0.7, 1000, "api_key",
            update_queue, None, None, None
        )
        
        file_path, documentation, pylint_report, file_content = result
        
        self.assertEqual(file_path, self.test_file)
        self.assertIn("UnicodeDecodeError", documentation)


class TestMainFunction(TestCase):
    """Test main Streamlit application function"""
    
    @patch('repo_docs.load_model_settings')
    @patch('repo_docs.load_api_keys')
    @patch('streamlit.title')
    @patch('streamlit.write')
    @patch('streamlit.columns')
    @patch('streamlit.selectbox')
    @patch('streamlit.text_input')
    @patch('streamlit.sidebar')
    @patch('streamlit.button')
    def test_main_function_basic_ui(self, mock_button, mock_sidebar, mock_text_input,
                                   mock_selectbox, mock_columns, mock_write, mock_title,
                                   mock_load_api_keys, mock_load_model_settings):
        """Test main function creates basic UI components"""
        mock_load_model_settings.return_value = {
            "model": "mistral:instruct",
            "temperature": 0.7,
            "max_tokens": 4000,
            "api_key": ""
        }
        mock_load_api_keys.return_value = {}
        
        # Mock Streamlit components
        mock_columns.return_value = [Mock(), Mock()]
        mock_selectbox.return_value = "documentation"
        mock_text_input.return_value = ""
        mock_button.return_value = False
        
        # Mock sidebar context manager
        mock_sidebar.__enter__ = Mock(return_value=mock_sidebar)
        mock_sidebar.__exit__ = Mock(return_value=None)
        
        with patch('repo_docs.get_available_models', return_value=['mistral:instruct']):
            main()
        
        # Verify UI components were created
        mock_title.assert_called_with("🔍 Repository Analyzer")
        mock_selectbox.assert_called()
        mock_text_input.assert_called()


class TestIntegrationScenarios(TestCase):
    """Test integration scenarios and edge cases"""
    
    def test_optional_dependencies_handling(self):
        """Test that the module handles missing optional dependencies gracefully"""
        # Test that module can import even without optional dependencies
        with patch.dict('sys.modules', {
            'fpdf': None,
            'radon.complexity': None,
            'radon.metrics': None,
            'flake8.api.legacy': None
        }):
            # Should not raise import errors
            import ollama_workbench.knowledge.repo_docs as repo_docs

            
            # Should have dummy implementations
            self.assertIsNotNone(repo_docs.PDF)
            self.assertIsNotNone(repo_docs.cc_visit)
    
    def test_file_processing_with_various_languages(self):
        """Test file processing with different programming languages"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create files in different languages
            files = {
                "test.py": "def hello(): pass",
                "test.js": "console.log('hello');",
                "test.php": "<?php echo 'hello'; ?>",
                "test.css": "body { color: red; }",
                "test.html": "<html><body>Hello</body></html>"
            }
            
            for filename, content in files.items():
                filepath = os.path.join(temp_dir, filename)
                with open(filepath, 'w') as f:
                    f.write(content)
            
            code_files = get_all_code_files(temp_dir, [])
            
            # Should find all code files
            self.assertEqual(len(code_files), 5)
            
            # Test file info for each
            for filepath in code_files:
                file_info = get_file_info(filepath)
                self.assertIsNotNone(file_info)
                self.assertIn("Language", file_info)
                self.assertNotEqual(file_info["Language"], "unknown")
        
        finally:
            shutil.rmtree(temp_dir)
    
    def test_large_repository_simulation(self):
        """Test handling of a large repository structure"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create a complex directory structure
            dirs = ["src", "tests", "docs", "config"]
            for dir_name in dirs:
                os.makedirs(os.path.join(temp_dir, dir_name))
            
            # Create many files
            for i in range(10):
                for dir_name in dirs:
                    filepath = os.path.join(temp_dir, dir_name, f"file_{i}.py")
                    with open(filepath, 'w') as f:
                        f.write(f"# File {i} in {dir_name}\ndef function_{i}(): pass")
            
            code_files = get_all_code_files(temp_dir, [])
            
            # Should find all Python files
            self.assertEqual(len(code_files), 40)
            
            # Test repository analysis
            repo_info = analyze_repository_structure(temp_dir, code_files)
            
            self.assertIsInstance(repo_info, dict)
            self.assertIn('file_structure', repo_info)
            self.assertEqual(len(repo_info['file_structure']), 40)
        
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
