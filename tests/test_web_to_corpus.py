"""
Comprehensive tests for web_to_corpus.py module.

Tests all web crawling functionality including:
- WebsiteCrawler class with multiple output formats
- Web page fetching with requests and Selenium
- Content extraction and link discovery
- PDF/JSON/TXT output generation
- User agent rotation and rate limiting
- File management and cleanup
- Streamlit UI components
- Error handling and edge cases
"""

import pytest
import os
import tempfile
import shutil
import json
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from unittest import TestCase
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock external dependencies before importing
with patch.dict('sys.modules', {
    'selenium': Mock(),
    'selenium.webdriver': Mock(),
    'selenium.webdriver.chrome.service': Mock(),
    'selenium.webdriver.chrome.options': Mock(),
    'webdriver_manager.chrome': Mock(),
    'bs4': Mock(),
    'pdfkit': Mock(),
    'PyPDF2': Mock(),
    'fake_useragent': Mock(),
    'streamlit': Mock()
}):
    # Import after mocking
    import ollama_workbench.knowledge.web_to_corpus as web_to_corpus

    from ollama_workbench.knowledge.web_to_corpus import WebsiteCrawler, get_random_user_agent, main


class TestUserAgentUtils(TestCase):
    """Test user agent utility functions"""
    
    @patch('web_to_corpus.UserAgent')
    def test_get_random_user_agent_success(self, mock_ua_class):
        """Test successful user agent generation"""
        mock_ua = Mock()
        mock_ua.random = "Mozilla/5.0 (Test Browser)"
        mock_ua_class.return_value = mock_ua
        
        result = get_random_user_agent()
        
        self.assertEqual(result, "Mozilla/5.0 (Test Browser)")
        mock_ua_class.assert_called_once()
    
    @patch('web_to_corpus.UserAgent')
    def test_get_random_user_agent_fallback(self, mock_ua_class):
        """Test fallback when UserAgent fails"""
        mock_ua_class.side_effect = Exception("UserAgent failed")
        
        with patch('random.choice') as mock_choice:
            mock_choice.return_value = "Fallback User Agent"
            
            result = get_random_user_agent()
            
            self.assertEqual(result, "Fallback User Agent")
            mock_choice.assert_called_once()


class TestWebsiteCrawlerInit(TestCase):
    """Test WebsiteCrawler initialization"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_url = "https://example.com"
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('web_to_corpus.tempfile.mkdtemp')
    @patch('web_to_corpus.webdriver.Chrome')
    @patch('web_to_corpus.ChromeDriverManager')
    @patch('web_to_corpus.Service')
    @patch('web_to_corpus.Options')
    @patch('web_to_corpus.get_random_user_agent')
    def test_crawler_init_success(self, mock_ua, mock_options, mock_service, 
                                 mock_driver_manager, mock_chrome, mock_mkdtemp):
        """Test successful crawler initialization"""
        mock_mkdtemp.return_value = self.temp_dir
        mock_ua.return_value = "Test User Agent"
        mock_chrome_instance = Mock()
        mock_chrome.return_value = mock_chrome_instance
        mock_driver_manager.return_value.install.return_value = "/path/to/chromedriver"
        
        crawler = WebsiteCrawler(self.test_url, "TXT")
        
        self.assertEqual(crawler.root_url, self.test_url)
        self.assertEqual(crawler.output_format, "TXT")
        self.assertEqual(crawler.visited_links, set())
        self.assertEqual(crawler.to_visit_links, {self.test_url})
        self.assertEqual(crawler.domain_name, "example.com")
        self.assertEqual(crawler.driver, mock_chrome_instance)
        self.assertEqual(crawler.crawled_data, [])
    
    @patch('web_to_corpus.tempfile.mkdtemp')
    @patch('web_to_corpus.webdriver.Chrome')
    @patch('streamlit.warning')
    def test_crawler_init_chrome_failure(self, mock_warning, mock_chrome, mock_mkdtemp):
        """Test crawler initialization when Chrome driver fails"""
        mock_mkdtemp.return_value = self.temp_dir
        mock_chrome.side_effect = Exception("Chrome driver failed")
        
        crawler = WebsiteCrawler(self.test_url, "PDF")
        
        self.assertIsNone(crawler.driver)
        mock_warning.assert_called()
        self.assertIn("PDF", crawler.output_format)
        self.assertIn("pdf_options", crawler.__dict__)
    
    @patch('web_to_corpus.tempfile.mkdtemp')
    @patch('web_to_corpus.Options')
    @patch('streamlit.warning')
    def test_crawler_init_options_failure(self, mock_warning, mock_options, mock_mkdtemp):
        """Test crawler initialization when Chrome options setup fails"""
        mock_mkdtemp.return_value = self.temp_dir
        mock_options.side_effect = Exception("Chrome setup failed")
        
        crawler = WebsiteCrawler(self.test_url, "JSON")
        
        self.assertIsNone(crawler.driver)
        mock_warning.assert_called()


class TestWebsiteCrawlerCleanup(TestCase):
    """Test WebsiteCrawler cleanup functionality"""
    
    @patch('web_to_corpus.shutil.rmtree')
    def test_crawler_del_with_driver(self, mock_rmtree):
        """Test crawler cleanup with active driver"""
        mock_driver = Mock()
        
        # Create a minimal crawler instance
        with patch('web_to_corpus.tempfile.mkdtemp') as mock_mkdtemp:
            mock_mkdtemp.return_value = "/temp/path"
            crawler = WebsiteCrawler("https://example.com", "TXT")
            crawler.driver = mock_driver
            
            # Trigger cleanup
            del crawler
            
            mock_driver.quit.assert_called_once()
            mock_rmtree.assert_called_with("/temp/path", ignore_errors=True)
    
    @patch('web_to_corpus.shutil.rmtree')
    def test_crawler_del_without_driver(self, mock_rmtree):
        """Test crawler cleanup without driver"""
        with patch('web_to_corpus.tempfile.mkdtemp') as mock_mkdtemp:
            mock_mkdtemp.return_value = "/temp/path"
            crawler = WebsiteCrawler("https://example.com", "TXT")
            crawler.driver = None
            
            # Trigger cleanup
            del crawler
            
            mock_rmtree.assert_called_with("/temp/path", ignore_errors=True)
    
    @patch('web_to_corpus.shutil.rmtree')
    @patch('builtins.print')
    def test_crawler_del_with_errors(self, mock_print, mock_rmtree):
        """Test crawler cleanup with errors"""
        mock_driver = Mock()
        mock_driver.quit.side_effect = Exception("Driver quit failed")
        mock_rmtree.side_effect = Exception("Rmtree failed")
        
        with patch('web_to_corpus.tempfile.mkdtemp') as mock_mkdtemp:
            mock_mkdtemp.return_value = "/temp/path"
            crawler = WebsiteCrawler("https://example.com", "TXT")
            crawler.driver = mock_driver
            
            # Should not raise exceptions
            del crawler
            
            # Should print error messages
            self.assertEqual(mock_print.call_count, 2)


class TestWebsiteCrawlerFetching(TestCase):
    """Test web page fetching functionality"""
    
    def setUp(self):
        """Set up test crawler"""
        with patch('web_to_corpus.tempfile.mkdtemp'):
            self.crawler = WebsiteCrawler("https://example.com", "TXT")
            self.crawler.driver = None
    
    @patch('web_to_corpus.requests.get')
    @patch('web_to_corpus.get_random_user_agent')
    def test_fetch_page_success(self, mock_ua, mock_get):
        """Test successful page fetching with requests"""
        mock_ua.return_value = "Test User Agent"
        mock_response = Mock()
        mock_response.text = "<html><body>Test content</body></html>"
        mock_get.return_value = mock_response
        
        result = self.crawler.fetch_page("https://example.com/page")
        
        self.assertEqual(result, "<html><body>Test content</body></html>")
        mock_get.assert_called_once()
        
        # Verify headers
        call_args = mock_get.call_args
        headers = call_args[1]['headers']
        self.assertEqual(headers['User-Agent'], "Test User Agent")
        self.assertIn('Accept', headers)
        self.assertEqual(call_args[1]['timeout'], 10)
    
    @patch('web_to_corpus.requests.get')
    @patch('streamlit.error')
    def test_fetch_page_request_exception(self, mock_error, mock_get):
        """Test page fetching with request exception"""
        mock_get.side_effect = requests.RequestException("Network error")
        
        result = self.crawler.fetch_page("https://example.com/page")
        
        self.assertIsNone(result)
        mock_error.assert_called_once()
        self.assertIn("Failed to fetch", mock_error.call_args[0][0])
    
    @patch('web_to_corpus.requests.get')
    @patch('streamlit.error')
    def test_fetch_page_http_error(self, mock_error, mock_get):
        """Test page fetching with HTTP error"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        result = self.crawler.fetch_page("https://example.com/page")
        
        self.assertIsNone(result)
        mock_error.assert_called_once()
    
    @patch('web_to_corpus.time.sleep')
    @patch('web_to_corpus.random.uniform')
    def test_fetch_page_selenium_success(self, mock_uniform, mock_sleep):
        """Test successful page fetching with Selenium"""
        mock_uniform.return_value = 3.5
        mock_driver = Mock()
        mock_driver.page_source = "<html><body>Selenium content</body></html>"
        self.crawler.driver = mock_driver
        
        result = self.crawler.fetch_page_selenium("https://example.com/page")
        
        self.assertEqual(result, "<html><body>Selenium content</body></html>")
        mock_driver.get.assert_called_once_with("https://example.com/page")
        mock_sleep.assert_called_once_with(3.5)
    
    @patch('streamlit.warning')
    def test_fetch_page_selenium_no_driver(self, mock_warning):
        """Test Selenium fetching without driver"""
        self.crawler.driver = None
        
        result = self.crawler.fetch_page_selenium("https://example.com/page")
        
        self.assertIsNone(result)
        mock_warning.assert_called_once()
        self.assertIn("Selenium driver not available", mock_warning.call_args[0][0])
    
    @patch('streamlit.error')
    def test_fetch_page_selenium_exception(self, mock_error):
        """Test Selenium fetching with exception"""
        mock_driver = Mock()
        mock_driver.get.side_effect = Exception("Selenium error")
        self.crawler.driver = mock_driver
        
        result = self.crawler.fetch_page_selenium("https://example.com/page")
        
        self.assertIsNone(result)
        mock_error.assert_called_once()


class TestWebsiteCrawlerContentExtraction(TestCase):
    """Test content extraction functionality"""
    
    def setUp(self):
        """Set up test crawler"""
        with patch('web_to_corpus.tempfile.mkdtemp'):
            self.crawler = WebsiteCrawler("https://example.com", "TXT")
    
    @patch('web_to_corpus.BeautifulSoup')
    def test_extract_main_content_basic(self, mock_soup_class):
        """Test basic content extraction"""
        # Mock BeautifulSoup behavior
        mock_soup = Mock()
        mock_soup_class.return_value = mock_soup
        
        # Mock script/style removal
        mock_soup.return_value = []
        
        # Mock text extraction
        mock_soup.get_text.return_value = "Line 1\n  Line 2  \n\nLine 3\n"
        
        result = self.crawler.extract_main_content("<html><body>Test</body></html>")
        
        expected = "Line 1\nLine 2\nLine 3"
        self.assertEqual(result, expected)
        mock_soup_class.assert_called_once_with("<html><body>Test</body></html>", 'html.parser')
    
    @patch('web_to_corpus.BeautifulSoup')
    def test_extract_main_content_with_scripts(self, mock_soup_class):
        """Test content extraction with script removal"""
        mock_soup = Mock()
        mock_soup_class.return_value = mock_soup
        
        # Mock script elements
        mock_script = Mock()
        mock_style = Mock()
        mock_soup.return_value = [mock_script, mock_style]
        
        mock_soup.get_text.return_value = "Clean content"
        
        result = self.crawler.extract_main_content("<html><script>js</script><body>content</body></html>")
        
        self.assertEqual(result, "Clean content")
        mock_script.decompose.assert_called_once()
        mock_style.decompose.assert_called_once()
    
    @patch('web_to_corpus.BeautifulSoup')
    def test_extract_main_content_complex_whitespace(self, mock_soup_class):
        """Test content extraction with complex whitespace"""
        mock_soup = Mock()
        mock_soup_class.return_value = mock_soup
        mock_soup.return_value = []
        
        # Test complex whitespace handling
        mock_soup.get_text.return_value = "  Title  \n\n  Paragraph 1  \n  \n  Paragraph 2  \n\n\n"
        
        result = self.crawler.extract_main_content("<html><body>complex</body></html>")
        
        expected = "Title\nParagraph 1\nParagraph 2"
        self.assertEqual(result, expected)


class TestWebsiteCrawlerLinkDiscovery(TestCase):
    """Test link discovery functionality"""
    
    def setUp(self):
        """Set up test crawler"""
        with patch('web_to_corpus.tempfile.mkdtemp'):
            self.crawler = WebsiteCrawler("https://example.com", "TXT")
    
    @patch('web_to_corpus.BeautifulSoup')
    @patch('web_to_corpus.urljoin')
    @patch('web_to_corpus.urldefrag')
    def test_find_links_on_page_valid_links(self, mock_urldefrag, mock_urljoin, mock_soup_class):
        """Test finding valid links on page"""
        # Mock BeautifulSoup
        mock_soup = Mock()
        mock_soup_class.return_value = mock_soup
        
        # Mock link elements
        mock_link1 = Mock()
        mock_link1.__getitem__.return_value = "/page1"
        mock_link2 = Mock()
        mock_link2.__getitem__.return_value = "/page2"
        mock_soup.find_all.return_value = [mock_link1, mock_link2]
        
        # Mock URL processing
        mock_urljoin.side_effect = [
            "https://example.com/page1",
            "https://example.com/page2"
        ]
        mock_urldefrag.side_effect = [
            ("https://example.com/page1", ""),
            ("https://example.com/page2", "")
        ]
        
        # Mock URL validation
        with patch.object(self.crawler, 'is_valid_url') as mock_is_valid:
            mock_is_valid.return_value = True
            
            self.crawler.find_links_on_page("https://example.com", "<html>content</html>")
            
            expected_links = {"https://example.com/page1", "https://example.com/page2"}
            self.assertEqual(self.crawler.to_visit_links, expected_links)
    
    @patch('web_to_corpus.BeautifulSoup')
    def test_find_links_on_page_invalid_links(self, mock_soup_class):
        """Test finding invalid links on page"""
        mock_soup = Mock()
        mock_soup_class.return_value = mock_soup
        
        mock_link = Mock()
        mock_link.__getitem__.return_value = "/invalid"
        mock_soup.find_all.return_value = [mock_link]
        
        with patch('web_to_corpus.urljoin') as mock_urljoin:
            mock_urljoin.return_value = "https://other-domain.com/page"
            
            with patch('web_to_corpus.urldefrag') as mock_urldefrag:
                mock_urldefrag.return_value = ("https://other-domain.com/page", "")
                
                with patch.object(self.crawler, 'is_valid_url') as mock_is_valid:
                    mock_is_valid.return_value = False
                    
                    self.crawler.find_links_on_page("https://example.com", "<html>content</html>")
                    
                    # Should not add invalid links
                    self.assertEqual(len(self.crawler.to_visit_links), 1)  # Only root URL
    
    @patch('web_to_corpus.urlparse')
    def test_is_valid_url_same_domain(self, mock_urlparse):
        """Test URL validation for same domain"""
        mock_parsed = Mock()
        mock_parsed.scheme = "https"
        mock_parsed.netloc = "example.com"
        mock_parsed.path = "/valid-page"
        mock_parsed.query = ""
        mock_urlparse.return_value = mock_parsed
        
        result = self.crawler.is_valid_url("https://example.com/valid-page")
        
        self.assertTrue(result)
    
    @patch('web_to_corpus.urlparse')
    def test_is_valid_url_different_domain(self, mock_urlparse):
        """Test URL validation for different domain"""
        mock_parsed = Mock()
        mock_parsed.scheme = "https"
        mock_parsed.netloc = "other-domain.com"
        mock_parsed.path = "/page"
        mock_parsed.query = ""
        mock_urlparse.return_value = mock_parsed
        
        result = self.crawler.is_valid_url("https://other-domain.com/page")
        
        self.assertFalse(result)
    
    @patch('web_to_corpus.urlparse')
    def test_is_valid_url_invalid_scheme(self, mock_urlparse):
        """Test URL validation for invalid scheme"""
        mock_parsed = Mock()
        mock_parsed.scheme = "ftp"
        mock_parsed.netloc = "example.com"
        mock_parsed.path = "/page"
        mock_parsed.query = ""
        mock_urlparse.return_value = mock_parsed
        
        result = self.crawler.is_valid_url("ftp://example.com/page")
        
        self.assertFalse(result)
    
    @patch('web_to_corpus.urlparse')
    def test_is_valid_url_excluded_extensions(self, mock_urlparse):
        """Test URL validation for excluded file extensions"""
        mock_parsed = Mock()
        mock_parsed.scheme = "https"
        mock_parsed.netloc = "example.com"
        mock_parsed.path = "/image.jpg"
        mock_parsed.query = ""
        mock_urlparse.return_value = mock_parsed
        
        result = self.crawler.is_valid_url("https://example.com/image.jpg")
        
        self.assertFalse(result)
    
    @patch('web_to_corpus.urlparse')
    def test_is_valid_url_excluded_query_params(self, mock_urlparse):
        """Test URL validation for excluded query parameters"""
        mock_parsed = Mock()
        mock_parsed.scheme = "https"
        mock_parsed.netloc = "example.com"
        mock_parsed.path = "/page"
        mock_parsed.query = "utm_source=test"
        mock_urlparse.return_value = mock_parsed
        
        result = self.crawler.is_valid_url("https://example.com/page?utm_source=test")
        
        self.assertFalse(result)


class TestWebsiteCrawlerFileSaving(TestCase):
    """Test file saving functionality"""
    
    def setUp(self):
        """Set up test crawler and temp directory"""
        self.temp_dir = tempfile.mkdtemp()
        
        with patch('web_to_corpus.tempfile.mkdtemp'):
            self.crawler = WebsiteCrawler("https://example.com", "PDF")
            self.crawler.temp_dir = self.temp_dir
    
    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('web_to_corpus.os.path.exists')
    @patch('web_to_corpus.os.makedirs')
    @patch('web_to_corpus.urlparse')
    def test_get_filename_basic(self, mock_urlparse, mock_makedirs, mock_exists):
        """Test basic filename generation"""
        mock_exists.return_value = False
        mock_parsed = Mock()
        mock_parsed.path = "/path/to/page"
        mock_urlparse.return_value = mock_parsed
        
        with patch('web_to_corpus.SCRIPT_DIR', '/script'):
            result = self.crawler.get_filename("https://example.com/path/to/page", "txt")
            
            expected = "/script/files/path_to_page.txt"
            self.assertEqual(result, expected)
            mock_makedirs.assert_called_once_with("/script/files")
    
    @patch('web_to_corpus.os.path.exists')
    @patch('web_to_corpus.urlparse')
    def test_get_filename_root_path(self, mock_urlparse, mock_exists):
        """Test filename generation for root path"""
        mock_exists.return_value = True
        mock_parsed = Mock()
        mock_parsed.path = "/"
        mock_urlparse.return_value = mock_parsed
        
        with patch('web_to_corpus.SCRIPT_DIR', '/script'):
            result = self.crawler.get_filename("https://example.com/", "pdf")
            
            expected = "/script/files/index.pdf"
            self.assertEqual(result, expected)
    
    @patch('streamlit.info')
    @patch('web_to_corpus.pdfkit.from_string')
    def test_save_page_as_pdf_success(self, mock_pdfkit, mock_info):
        """Test successful PDF saving"""
        mock_pdfkit.return_value = None
        
        with patch.object(self.crawler, 'get_filename') as mock_get_filename:
            mock_get_filename.return_value = "/test/file.pdf"
            
            result = self.crawler.save_page_as_pdf("https://example.com/page", "content")
            
            self.assertEqual(result, "/test/file.pdf")
            mock_pdfkit.assert_called_once()
            mock_info.assert_called_once()
    
    @patch('streamlit.info')
    @patch('streamlit.warning')
    @patch('web_to_corpus.pdfkit.from_string')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_page_as_pdf_fallback_to_text(self, mock_file, mock_pdfkit, mock_warning, mock_info):
        """Test PDF saving fallback to text file"""
        mock_pdfkit.side_effect = ImportError("pdfkit not available")
        
        with patch.object(self.crawler, 'get_filename') as mock_get_filename:
            mock_get_filename.side_effect = ["/test/file.pdf", "/test/file.txt"]
            
            result = self.crawler.save_page_as_pdf("https://example.com/page", "content")
            
            self.assertEqual(result, "/test/file.txt")
            mock_warning.assert_called_once()
            mock_file.assert_called_once_with("/test/file.txt", "w", encoding="utf-8")
    
    @patch('streamlit.info')
    @patch('streamlit.error')
    @patch('web_to_corpus.pdfkit.from_string')
    def test_save_page_as_pdf_general_exception(self, mock_pdfkit, mock_error, mock_info):
        """Test PDF saving with general exception"""
        mock_pdfkit.side_effect = Exception("General error")
        
        with patch.object(self.crawler, 'get_filename') as mock_get_filename:
            mock_get_filename.return_value = "/test/file.pdf"
            
            result = self.crawler.save_page_as_pdf("https://example.com/page", "content")
            
            self.assertIsNone(result)
            mock_error.assert_called_once()


class TestWebsiteCrawlerOutputGeneration(TestCase):
    """Test output generation functionality"""
    
    def setUp(self):
        """Set up test crawler and data"""
        self.temp_dir = tempfile.mkdtemp()
        
        with patch('web_to_corpus.tempfile.mkdtemp'):
            self.crawler = WebsiteCrawler("https://example.com", "TXT")
            self.crawler.temp_dir = self.temp_dir
            
        self.crawler.crawled_data = [
            {"url": "https://example.com/page1", "content": "Content 1"},
            {"url": "https://example.com/page2", "content": "Content 2"}
        ]
    
    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('web_to_corpus.os.path.join')
    def test_generate_output_txt(self, mock_join):
        """Test TXT output generation"""
        mock_join.return_value = "/test/output.txt"
        self.crawler.output_format = "TXT"
        
        with patch.object(self.crawler, 'save_as_txt') as mock_save_txt:
            self.crawler.generate_output("output.txt")
            
            mock_save_txt.assert_called_once_with("/test/output.txt")
    
    @patch('web_to_corpus.os.path.join')
    def test_generate_output_json(self, mock_join):
        """Test JSON output generation"""
        mock_join.return_value = "/test/output.json"
        self.crawler.output_format = "JSON"
        
        with patch.object(self.crawler, 'save_as_json') as mock_save_json:
            self.crawler.generate_output("output.json")
            
            mock_save_json.assert_called_once_with("/test/output.json")
    
    @patch('web_to_corpus.os.path.join')
    def test_generate_output_pdf(self, mock_join):
        """Test PDF output generation"""
        mock_join.return_value = "/test/output.pdf"
        self.crawler.output_format = "PDF"
        
        with patch.object(self.crawler, 'merge_pdfs') as mock_merge_pdfs:
            self.crawler.generate_output("output.pdf")
            
            mock_merge_pdfs.assert_called_once_with("/test/output.pdf")
    
    @patch('builtins.open', new_callable=mock_open)
    def test_save_as_txt(self, mock_file):
        """Test saving as TXT file"""
        self.crawler.save_as_txt("/test/output.txt")
        
        mock_file.assert_called_once_with("/test/output.txt", 'w', encoding='utf-8')
        
        # Check that content was written
        handle = mock_file()
        written_content = ''.join(call[0][0] for call in handle.write.call_args_list)
        
        self.assertIn("URL: https://example.com/page1", written_content)
        self.assertIn("Content 1", written_content)
        self.assertIn("URL: https://example.com/page2", written_content)
        self.assertIn("Content 2", written_content)
        self.assertIn("-" * 80, written_content)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_as_json(self, mock_json_dump, mock_file):
        """Test saving as JSON file"""
        self.crawler.save_as_json("/test/output.json")
        
        mock_file.assert_called_once_with("/test/output.json", 'w', encoding='utf-8')
        mock_json_dump.assert_called_once_with(
            self.crawler.crawled_data, 
            mock_file(), 
            ensure_ascii=False, 
            indent=4
        )
    
    @patch('web_to_corpus.PdfMerger')
    def test_merge_pdfs(self, mock_pdf_merger_class):
        """Test PDF merging"""
        mock_merger = Mock()
        mock_pdf_merger_class.return_value = mock_merger
        
        # Set up crawler data with file paths
        self.crawler.crawled_data = [
            {"url": "https://example.com/page1", "file": "/path/file1.pdf"},
            {"url": "https://example.com/page2", "file": "/path/file2.pdf"}
        ]
        
        self.crawler.merge_pdfs("/test/merged.pdf")
        
        mock_merger.append.assert_has_calls([
            call("/path/file1.pdf"),
            call("/path/file2.pdf")
        ])
        mock_merger.write.assert_called_once_with("/test/merged.pdf")
        mock_merger.close.assert_called_once()


class TestWebsiteCrawlerMainCrawl(TestCase):
    """Test main crawling functionality"""
    
    def setUp(self):
        """Set up test crawler"""
        with patch('web_to_corpus.tempfile.mkdtemp'):
            self.crawler = WebsiteCrawler("https://example.com", "TXT")
            
        # Mock Streamlit components
        self.mock_progress = Mock()
        self.mock_status = Mock()
    
    @patch('streamlit.progress')
    @patch('streamlit.empty')
    @patch('web_to_corpus.time.sleep')
    @patch('web_to_corpus.random.uniform')
    def test_crawl_basic_flow(self, mock_uniform, mock_sleep, mock_empty, mock_progress):
        """Test basic crawling flow"""
        mock_progress.return_value = self.mock_progress
        mock_empty.return_value = self.mock_status
        mock_uniform.return_value = 2.0
        
        # Set up limited crawling
        self.crawler.to_visit_links = {"https://example.com"}
        
        with patch.object(self.crawler, 'fetch_page') as mock_fetch:
            mock_fetch.return_value = "<html><body>Test content</body></html>"
            
            with patch.object(self.crawler, 'extract_main_content') as mock_extract:
                mock_extract.return_value = "Extracted content"
                
                with patch.object(self.crawler, 'find_links_on_page') as mock_find_links:
                    # Prevent infinite crawling
                    mock_find_links.return_value = None
                    
                    self.crawler.crawl()
                    
                    # Verify crawling progression
                    self.assertIn("https://example.com", self.crawler.visited_links)
                    self.assertEqual(len(self.crawler.crawled_data), 1)
                    self.assertEqual(self.crawler.crawled_data[0]['url'], "https://example.com")
                    self.assertEqual(self.crawler.crawled_data[0]['content'], "Extracted content")
    
    @patch('streamlit.progress')
    @patch('streamlit.empty')
    def test_crawl_fetch_fallback(self, mock_empty, mock_progress):
        """Test crawling with fetch fallback to Selenium"""
        mock_progress.return_value = self.mock_progress
        mock_empty.return_value = self.mock_status
        
        self.crawler.to_visit_links = {"https://example.com"}
        
        with patch.object(self.crawler, 'fetch_page') as mock_fetch:
            mock_fetch.return_value = None  # First fetch fails
            
            with patch.object(self.crawler, 'fetch_page_selenium') as mock_selenium:
                mock_selenium.return_value = "<html><body>Selenium content</body></html>"
                
                with patch.object(self.crawler, 'extract_main_content') as mock_extract:
                    mock_extract.return_value = "Selenium extracted"
                    
                    with patch.object(self.crawler, 'find_links_on_page'):
                        with patch('web_to_corpus.time.sleep'):
                            with patch('web_to_corpus.random.uniform'):
                                self.crawler.crawl()
                                
                                # Should try Selenium after requests fails
                                mock_selenium.assert_called_once()
                                self.assertEqual(len(self.crawler.crawled_data), 1)
    
    @patch('streamlit.progress')
    @patch('streamlit.empty')
    def test_crawl_both_fetch_fail(self, mock_empty, mock_progress):
        """Test crawling when both fetch methods fail"""
        mock_progress.return_value = self.mock_progress
        mock_empty.return_value = self.mock_status
        
        self.crawler.to_visit_links = {"https://example.com"}
        
        with patch.object(self.crawler, 'fetch_page') as mock_fetch:
            mock_fetch.return_value = None
            
            with patch.object(self.crawler, 'fetch_page_selenium') as mock_selenium:
                mock_selenium.return_value = None
                
                with patch('web_to_corpus.time.sleep'):
                    with patch('web_to_corpus.random.uniform'):
                        self.crawler.crawl()
                        
                        # Should visit the URL but not add to crawled_data
                        self.assertIn("https://example.com", self.crawler.visited_links)
                        self.assertEqual(len(self.crawler.crawled_data), 0)
    
    @patch('streamlit.progress')
    @patch('streamlit.empty')
    def test_crawl_pdf_output(self, mock_empty, mock_progress):
        """Test crawling with PDF output format"""
        mock_progress.return_value = self.mock_progress
        mock_empty.return_value = self.mock_status
        
        self.crawler.output_format = "PDF"
        self.crawler.to_visit_links = {"https://example.com"}
        
        with patch.object(self.crawler, 'fetch_page') as mock_fetch:
            mock_fetch.return_value = "<html><body>Test</body></html>"
            
            with patch.object(self.crawler, 'extract_main_content') as mock_extract:
                mock_extract.return_value = "Content"
                
                with patch.object(self.crawler, 'save_page_as_pdf') as mock_save_pdf:
                    mock_save_pdf.return_value = "/path/test.pdf"
                    
                    with patch.object(self.crawler, 'find_links_on_page'):
                        with patch('web_to_corpus.time.sleep'):
                            with patch('web_to_corpus.random.uniform'):
                                self.crawler.crawl()
                                
                                mock_save_pdf.assert_called_once_with("https://example.com", "Content")
                                self.assertEqual(self.crawler.crawled_data[0]['file'], "/path/test.pdf")


class TestMainStreamlitApp(TestCase):
    """Test main Streamlit application"""
    
    @patch('streamlit.title')
    @patch('streamlit.write')
    @patch('streamlit.columns')
    @patch('streamlit.text_input')
    @patch('streamlit.selectbox')
    @patch('streamlit.button')
    def test_main_app_ui_creation(self, mock_button, mock_selectbox, mock_text_input, 
                                 mock_columns, mock_write, mock_title):
        """Test main app UI component creation"""
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.return_value = ""
        mock_selectbox.return_value = "TXT"
        mock_button.return_value = False
        
        main()
        
        mock_title.assert_called_once_with("🕸️ Web Crawler")
        mock_write.assert_called_once()
        mock_columns.assert_called_once()
        mock_text_input.assert_called_once()
        mock_selectbox.assert_called_once()
        mock_button.assert_called_once()
    
    @patch('streamlit.title')
    @patch('streamlit.write')
    @patch('streamlit.columns')
    @patch('streamlit.text_input')
    @patch('streamlit.selectbox')
    @patch('streamlit.button')
    @patch('streamlit.success')
    @patch('streamlit.download_button')
    @patch('builtins.open', new_callable=mock_open, read_data=b"test content")
    @patch('web_to_corpus.urlparse')
    def test_main_app_crawling_flow(self, mock_urlparse, mock_file, mock_download, 
                                   mock_success, mock_button, mock_selectbox, 
                                   mock_text_input, mock_columns, mock_write, mock_title):
        """Test main app crawling flow"""
        # Set up UI returns
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.return_value = "https://example.com"
        mock_selectbox.return_value = "JSON"
        mock_button.return_value = True
        
        # Mock URL parsing
        mock_parsed = Mock()
        mock_parsed.netloc = "example.com"
        mock_urlparse.return_value = mock_parsed
        
        # Mock crawler
        with patch('web_to_corpus.WebsiteCrawler') as mock_crawler_class:
            mock_crawler = Mock()
            mock_crawler_class.return_value = mock_crawler
            
            with patch('web_to_corpus.os.path.join') as mock_join:
                mock_join.return_value = "/test/example.com.json"
                
                with patch('web_to_corpus.os.path.exists') as mock_exists:
                    mock_exists.return_value = True
                    
                    main()
                    
                    # Verify crawler was created and used
                    mock_crawler_class.assert_called_once_with("https://example.com", "JSON")
                    mock_crawler.crawl.assert_called_once()
                    mock_crawler.generate_output.assert_called_once_with("example.com.json")
                    
                    # Verify success messages
                    self.assertEqual(mock_success.call_count, 2)
                    mock_download.assert_called_once()
    
    @patch('streamlit.title')
    @patch('streamlit.write')
    @patch('streamlit.columns')
    @patch('streamlit.text_input')
    @patch('streamlit.selectbox')
    @patch('streamlit.button')
    @patch('streamlit.error')
    def test_main_app_no_url_error(self, mock_error, mock_button, mock_selectbox, 
                                  mock_text_input, mock_columns, mock_write, mock_title):
        """Test main app error when no URL provided"""
        mock_columns.return_value = [Mock(), Mock()]
        mock_text_input.return_value = ""  # Empty URL
        mock_selectbox.return_value = "TXT"
        mock_button.return_value = True  # Button clicked
        
        main()
        
        mock_error.assert_called_once_with("Please enter a valid URL.")


class TestIntegrationScenarios(TestCase):
    """Test integration scenarios and edge cases"""
    
    def test_crawler_full_workflow_txt(self):
        """Test complete crawler workflow for TXT output"""
        with patch('web_to_corpus.tempfile.mkdtemp') as mock_mkdtemp:
            mock_mkdtemp.return_value = "/temp"
            
            with patch('web_to_corpus.webdriver.Chrome'):
                crawler = WebsiteCrawler("https://example.com", "TXT")
                
                # Mock all external dependencies
                with patch.object(crawler, 'fetch_page') as mock_fetch:
                    mock_fetch.return_value = "<html><body><a href='/page1'>Link</a>Content</body></html>"
                    
                    with patch.object(crawler, 'extract_main_content') as mock_extract:
                        mock_extract.return_value = "Extracted content"
                        
                        with patch('web_to_corpus.BeautifulSoup') as mock_soup_class:
                            # Mock link discovery
                            mock_soup = Mock()
                            mock_link = Mock()
                            mock_link.__getitem__.return_value = "/page1"
                            mock_soup.find_all.return_value = [mock_link]
                            mock_soup_class.return_value = mock_soup
                            
                            with patch('web_to_corpus.urljoin') as mock_urljoin:
                                mock_urljoin.return_value = "https://example.com/page1"
                                
                                with patch('web_to_corpus.urldefrag') as mock_urldefrag:
                                    mock_urldefrag.return_value = ("https://example.com/page1", "")
                                    
                                    with patch('streamlit.progress') as mock_progress:
                                        with patch('streamlit.empty') as mock_empty:
                                            with patch('web_to_corpus.time.sleep'):
                                                with patch('web_to_corpus.random.uniform'):
                                                    # Limit to one iteration
                                                    original_to_visit = crawler.to_visit_links.copy()
                                                    crawler.to_visit_links = {"https://example.com"}
                                                    
                                                    crawler.crawl()
                                                    
                                                    # Verify results
                                                    self.assertEqual(len(crawler.visited_links), 1)
                                                    self.assertEqual(len(crawler.crawled_data), 1)
                                                    self.assertEqual(crawler.crawled_data[0]['url'], "https://example.com")
    
    def test_error_handling_resilience(self):
        """Test crawler resilience to various errors"""
        with patch('web_to_corpus.tempfile.mkdtemp'):
            crawler = WebsiteCrawler("https://example.com", "PDF")
            
            # Test with various exception scenarios
            error_scenarios = [
                requests.RequestException("Network error"),
                requests.HTTPError("HTTP error"), 
                Exception("General error"),
                UnicodeDecodeError('utf-8', b'', 0, 1, 'decode error')
            ]
            
            for error in error_scenarios:
                with patch.object(crawler, 'fetch_page') as mock_fetch:
                    mock_fetch.side_effect = error
                    
                    with patch.object(crawler, 'fetch_page_selenium') as mock_selenium:
                        mock_selenium.return_value = None
                        
                        with patch('streamlit.progress'):
                            with patch('streamlit.empty'):
                                with patch('web_to_corpus.time.sleep'):
                                    with patch('web_to_corpus.random.uniform'):
                                        # Should not crash
                                        crawler.to_visit_links = {"https://example.com/test"}
                                        crawler.visited_links = set()
                                        crawler.crawled_data = []
                                        
                                        try:
                                            crawler.crawl()
                                        except Exception as e:
                                            self.fail(f"Crawler crashed with {type(error).__name__}: {e}")
    
    def test_memory_and_resource_management(self):
        """Test memory management for large crawls"""
        with patch('web_to_corpus.tempfile.mkdtemp'):
            crawler = WebsiteCrawler("https://example.com", "TXT")
            
            # Simulate large number of URLs
            large_url_set = {f"https://example.com/page{i}" for i in range(1000)}
            crawler.to_visit_links = large_url_set
            
            # Mock to prevent actual crawling
            with patch.object(crawler, 'fetch_page') as mock_fetch:
                mock_fetch.return_value = None
                
                with patch('streamlit.progress'):
                    with patch('streamlit.empty'):
                        with patch('web_to_corpus.time.sleep'):
                            with patch('web_to_corpus.random.uniform'):
                                # Should handle large sets without memory issues
                                crawler.crawl()
                                
                                # Verify all URLs were processed (even if content fetch failed)
                                self.assertEqual(len(crawler.visited_links), 1000)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
