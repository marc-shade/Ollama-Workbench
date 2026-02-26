"""
Comprehensive tests for performance_metrics.py module.

Tests all performance metrics functionality including:
- Performance metrics interface and displays
- Data loading and saving functionality
- Chart generation and data processing
- Error handling and edge cases
"""

import pytest
import json
import os
import tempfile
import shutil
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from unittest import TestCase
from datetime import datetime, timedelta
import streamlit as st
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ollama_workbench.server.performance_metrics import (
    performance_metrics_interface,
    display_response_time_metrics,
    display_token_usage_metrics,
    display_model_comparison,
    load_metrics_data,
    record_metrics
)


class TestPerformanceMetricsInterface(TestCase):
    """Test the main interface function"""
    
    @patch('streamlit.title')
    @patch('streamlit.tabs')
    @patch('ollama_workbench.server.performance_metrics.display_response_time_metrics')
    @patch('ollama_workbench.server.performance_metrics.display_token_usage_metrics')
    @patch('ollama_workbench.server.performance_metrics.display_model_comparison')
    def test_performance_metrics_interface(self, mock_model_comp, mock_token_usage, 
                                         mock_response_time, mock_tabs, mock_title):
        """Test main interface creates tabs and calls display functions"""
        # Mock the tab context managers
        tab1, tab2, tab3 = Mock(), Mock(), Mock()
        tab1.__enter__ = Mock(return_value=tab1)
        tab1.__exit__ = Mock(return_value=None)
        tab2.__enter__ = Mock(return_value=tab2)
        tab2.__exit__ = Mock(return_value=None)
        tab3.__enter__ = Mock(return_value=tab3)
        tab3.__exit__ = Mock(return_value=None)
        
        mock_tabs.return_value = [tab1, tab2, tab3]
        
        performance_metrics_interface()
        
        # Verify interface setup
        mock_title.assert_called_once_with("Model Performance Metrics")
        mock_tabs.assert_called_once_with(["Response Time", "Token Usage", "Model Comparison"])
        
        # Verify display functions are called
        mock_response_time.assert_called_once()
        mock_token_usage.assert_called_once()
        mock_model_comp.assert_called_once()


class TestDisplayResponseTimeMetrics(TestCase):
    """Test response time metrics display function"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_metrics = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "model": "llama3",
                "response_time": 2.5,
                "prompt_length": 100,
                "input_tokens": 50,
                "output_tokens": 75
            },
            {
                "timestamp": "2024-01-01T11:00:00",
                "model": "gpt-4",
                "response_time": 1.8,
                "prompt_length": 120,
                "input_tokens": 60,
                "output_tokens": 80
            },
            {
                "timestamp": "2024-01-01T12:00:00",
                "model": "llama3",
                "response_time": 3.2,
                "prompt_length": 150,
                "input_tokens": 75,
                "output_tokens": 90
            }
        ]
    
    @patch('streamlit.subheader')
    @patch('streamlit.info')
    @patch('ollama_workbench.server.performance_metrics.load_metrics_data')
    def test_display_response_time_metrics_no_data(self, mock_load, mock_info, mock_subheader):
        """Test display when no metrics data available"""
        mock_load.return_value = []
        
        display_response_time_metrics()
        
        mock_subheader.assert_called_with("Response Time Metrics")
        mock_info.assert_called_with("No performance metrics data available yet. Start using the chat to generate metrics.")
    
    @patch('streamlit.subheader')
    @patch('streamlit.info')
    @patch('ollama_workbench.server.performance_metrics.load_metrics_data')
    def test_display_response_time_metrics_no_response_time_data(self, mock_load, mock_info, mock_subheader):
        """Test display when metrics exist but no response time data"""
        mock_load.return_value = [{"timestamp": "2024-01-01", "model": "test"}]
        
        display_response_time_metrics()
        
        mock_info.assert_called_with("No response time data available yet.")
    
    @patch('streamlit.plotly_chart')
    @patch('streamlit.subheader')
    @patch('ollama_workbench.server.performance_metrics.load_metrics_data')
    def test_display_response_time_metrics_with_data(self, mock_load, mock_subheader, mock_plotly):
        """Test display with valid response time data"""
        mock_load.return_value = self.sample_metrics
        
        display_response_time_metrics()
        
        # Should call subheader multiple times for different sections
        expected_calls = [
            call("Response Time Metrics"),
            call("Response Time Over Time"),
            call("Average Response Time by Model"),
            call("Response Time vs Prompt Length")
        ]
        mock_subheader.assert_has_calls(expected_calls)
        
        # Should create 3 plotly charts
        self.assertEqual(mock_plotly.call_count, 3)
    
    @patch('streamlit.plotly_chart')
    @patch('streamlit.subheader')
    @patch('ollama_workbench.server.performance_metrics.load_metrics_data')
    def test_display_response_time_metrics_missing_fields(self, mock_load, mock_subheader, mock_plotly):
        """Test display with data missing some fields"""
        incomplete_metrics = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "model": "llama3",
                "response_time": 2.5
                # Missing prompt_length
            }
        ]
        mock_load.return_value = incomplete_metrics
        
        display_response_time_metrics()
        
        # Should still create charts with default values
        self.assertEqual(mock_plotly.call_count, 3)


class TestDisplayTokenUsageMetrics(TestCase):
    """Test token usage metrics display function"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_metrics = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "model": "llama3",
                "input_tokens": 50,
                "output_tokens": 75
            },
            {
                "timestamp": "2024-01-01T11:00:00",
                "model": "gpt-4",
                "input_tokens": 60,
                "output_tokens": 80
            }
        ]
    
    @patch('streamlit.subheader')
    @patch('streamlit.info')
    @patch('ollama_workbench.server.performance_metrics.load_metrics_data')
    def test_display_token_usage_metrics_no_data(self, mock_load, mock_info, mock_subheader):
        """Test display when no metrics data available"""
        mock_load.return_value = []
        
        display_token_usage_metrics()
        
        mock_subheader.assert_called_with("Token Usage Metrics")
        mock_info.assert_called_with("No performance metrics data available yet. Start using the chat to generate metrics.")
    
    @patch('streamlit.subheader')
    @patch('streamlit.info')
    @patch('ollama_workbench.server.performance_metrics.load_metrics_data')
    def test_display_token_usage_metrics_no_token_data(self, mock_load, mock_info, mock_subheader):
        """Test display when metrics exist but no token data"""
        mock_load.return_value = [{"timestamp": "2024-01-01", "model": "test"}]
        
        display_token_usage_metrics()
        
        mock_info.assert_called_with("No token usage data available yet.")
    
    @patch('streamlit.plotly_chart')
    @patch('streamlit.subheader')
    @patch('ollama_workbench.server.performance_metrics.load_metrics_data')
    def test_display_token_usage_metrics_with_data(self, mock_load, mock_subheader, mock_plotly):
        """Test display with valid token usage data"""
        mock_load.return_value = self.sample_metrics
        
        display_token_usage_metrics()
        
        # Should call subheader multiple times for different sections
        expected_calls = [
            call("Token Usage Metrics"),
            call("Token Usage Over Time"),
            call("Input vs Output Tokens by Model"),
            call("Total Token Usage by Model")
        ]
        mock_subheader.assert_has_calls(expected_calls)
        
        # Should create 3 plotly charts
        self.assertEqual(mock_plotly.call_count, 3)
    
    @patch('streamlit.plotly_chart')
    @patch('streamlit.subheader')
    @patch('ollama_workbench.server.performance_metrics.load_metrics_data')
    def test_display_token_usage_metrics_missing_fields(self, mock_load, mock_subheader, mock_plotly):
        """Test display with data missing some token fields"""
        incomplete_metrics = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "model": "llama3",
                "input_tokens": 50
                # Missing output_tokens
            }
        ]
        mock_load.return_value = incomplete_metrics
        
        display_token_usage_metrics()
        
        # Should still create charts with default values
        self.assertEqual(mock_plotly.call_count, 3)


class TestDisplayModelComparison(TestCase):
    """Test model comparison display function"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_metrics = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "model": "llama3",
                "response_time": 2.5,
                "input_tokens": 50,
                "output_tokens": 75
            },
            {
                "timestamp": "2024-01-01T11:00:00",
                "model": "gpt-4",
                "response_time": 1.8,
                "input_tokens": 60,
                "output_tokens": 80
            }
        ]
    
    @patch('streamlit.subheader')
    @patch('streamlit.info')
    @patch('ollama_workbench.server.performance_metrics.load_metrics_data')
    def test_display_model_comparison_no_data(self, mock_load, mock_info, mock_subheader):
        """Test display when no metrics data available"""
        mock_load.return_value = []
        
        display_model_comparison()
        
        mock_subheader.assert_called_with("Model Comparison Dashboard")
        mock_info.assert_called_with("No performance metrics data available yet. Start using the chat to generate metrics.")
    
    @patch('streamlit.subheader')
    @patch('streamlit.info')
    @patch('ollama_workbench.server.performance_metrics.load_metrics_data')
    def test_display_model_comparison_insufficient_data(self, mock_load, mock_info, mock_subheader):
        """Test display when data exists but is insufficient for comparison"""
        incomplete_metrics = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "model": "llama3",
                "response_time": 2.5
                # Missing required fields for comparison
            }
        ]
        mock_load.return_value = incomplete_metrics
        
        display_model_comparison()
        
        mock_info.assert_called_with("Not enough data for model comparison yet.")
    
    @patch('streamlit.plotly_chart')
    @patch('streamlit.dataframe')
    @patch('streamlit.subheader')
    @patch('ollama_workbench.server.performance_metrics.load_metrics_data')
    def test_display_model_comparison_with_data(self, mock_load, mock_subheader, mock_dataframe, mock_plotly):
        """Test display with sufficient comparison data"""
        mock_load.return_value = self.sample_metrics
        
        display_model_comparison()
        
        # Should call subheader multiple times for different sections
        expected_calls = [
            call("Model Comparison Dashboard"),
            call("Average Performance Metrics by Model"),
            call("Model Throughput (Tokens per Second)"),
            call("Model Performance Radar Chart")
        ]
        mock_subheader.assert_has_calls(expected_calls)
        
        # Should display dataframe and create 2 plotly charts
        mock_dataframe.assert_called_once()
        self.assertEqual(mock_plotly.call_count, 2)
    
    @patch('streamlit.plotly_chart')
    @patch('streamlit.dataframe')
    @patch('streamlit.subheader')
    @patch('ollama_workbench.server.performance_metrics.load_metrics_data')
    def test_display_model_comparison_division_by_zero_handling(self, mock_load, mock_subheader, mock_dataframe, mock_plotly):
        """Test handling of division by zero in tokens per second calculation"""
        zero_time_metrics = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "model": "llama3",
                "response_time": 0,  # Zero response time
                "input_tokens": 50,
                "output_tokens": 75
            }
        ]
        mock_load.return_value = zero_time_metrics
        
        # Should not raise an exception
        display_model_comparison()
        
        # Should still create charts
        self.assertEqual(mock_plotly.call_count, 2)


class TestLoadMetricsData(TestCase):
    """Test load_metrics_data function"""
    
    def setUp(self):
        """Set up temporary directory for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.metrics_file = os.path.join(self.temp_dir, "data", "performance_metrics.json")
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    @patch('os.path.join')
    def test_load_metrics_data_file_not_exists(self, mock_join):
        """Test loading when metrics file doesn't exist"""
        mock_join.return_value = "nonexistent_file.json"
        
        result = load_metrics_data()
        
        self.assertEqual(result, [])
    
    def test_load_metrics_data_valid_file(self):
        """Test loading valid metrics file"""
        test_data = [
            {"model": "llama3", "response_time": 2.5},
            {"model": "gpt-4", "response_time": 1.8}
        ]
        
        with open(self.metrics_file, "w") as f:
            json.dump(test_data, f)
        
        with patch('os.path.join', return_value=self.metrics_file):
            result = load_metrics_data()
        
        self.assertEqual(result, test_data)
    
    def test_load_metrics_data_empty_file(self):
        """Test loading empty metrics file"""
        with open(self.metrics_file, "w") as f:
            f.write("")
        
        with patch('os.path.join', return_value=self.metrics_file):
            with patch('streamlit.error') as mock_error:
                result = load_metrics_data()
        
        self.assertEqual(result, [])
        mock_error.assert_called_once()
    
    def test_load_metrics_data_invalid_json(self):
        """Test loading file with invalid JSON"""
        with open(self.metrics_file, "w") as f:
            f.write("invalid json content")
        
        with patch('os.path.join', return_value=self.metrics_file):
            with patch('streamlit.error') as mock_error:
                result = load_metrics_data()
        
        self.assertEqual(result, [])
        mock_error.assert_called_once()


class TestRecordMetrics(TestCase):
    """Test record_metrics function"""
    
    def setUp(self):
        """Set up temporary directory for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.metrics_dir = os.path.join(self.temp_dir, "data")
        self.metrics_file = os.path.join(self.metrics_dir, "performance_metrics.json")
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    @patch('os.path.join')
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_record_metrics_new_file(self, mock_json_dump, mock_file, mock_makedirs, mock_join):
        """Test recording metrics when file doesn't exist"""
        mock_join.side_effect = lambda *args: "/".join(args)
        
        # Mock file not existing
        with patch('os.path.exists', return_value=False):
            record_metrics("llama3", 2.5, prompt_length=100, input_tokens=50, output_tokens=75)
        
        # Should create directory
        mock_makedirs.assert_called_once()
        
        # Should write new file
        mock_file.assert_called_once()
        mock_json_dump.assert_called_once()
        
        # Check the data structure passed to json.dump
        call_args = mock_json_dump.call_args[0][0]
        self.assertEqual(len(call_args), 1)  # One entry
        self.assertEqual(call_args[0]["model"], "llama3")
        self.assertEqual(call_args[0]["response_time"], 2.5)
        self.assertEqual(call_args[0]["prompt_length"], 100)
        self.assertEqual(call_args[0]["input_tokens"], 50)
        self.assertEqual(call_args[0]["output_tokens"], 75)
        self.assertIn("timestamp", call_args[0])
    
    def test_record_metrics_existing_file(self):
        """Test recording metrics when file already exists"""
        # Create initial file with existing data
        existing_data = [{"model": "gpt-4", "response_time": 1.5}]
        os.makedirs(self.metrics_dir, exist_ok=True)
        with open(self.metrics_file, "w") as f:
            json.dump(existing_data, f)
        
        # Patch to return our test file
        def mock_path_join(*args):
            if len(args) == 2 and args[0] == "data" and args[1] == "performance_metrics.json":
                return self.metrics_file
            return os.path.join(*args)
        
        with patch('os.path.join', side_effect=mock_path_join):
            record_metrics("llama3", 2.5)
        
        # Check file contents
        with open(self.metrics_file, "r") as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["model"], "gpt-4")
        self.assertEqual(data[1]["model"], "llama3")
    
    @patch('os.path.join')
    @patch('os.makedirs')
    @patch('os.path.exists')
    def test_record_metrics_corrupted_existing_file(self, mock_exists, mock_makedirs, mock_join):
        """Test recording metrics when existing file is corrupted"""
        mock_join.side_effect = lambda *args: "/".join(args)
        mock_exists.return_value = True
        
        # Mock corrupted file
        with patch('builtins.open', mock_open(read_data="corrupted json")):
            with patch('json.dump') as mock_json_dump:
                record_metrics("llama3", 2.5)
        
        # Should still work and start with empty list
        mock_json_dump.assert_called_once()
        call_args = mock_json_dump.call_args[0][0]
        self.assertEqual(len(call_args), 1)  # Only new entry
    
    def test_record_metrics_minimal_data(self):
        """Test recording metrics with minimal required data"""
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Patch to return our test file
        def mock_path_join(*args):
            if len(args) == 2 and args[0] == "data" and args[1] == "performance_metrics.json":
                return self.metrics_file
            return os.path.join(*args)
        
        with patch('os.path.join', side_effect=mock_path_join):
            record_metrics("test_model", 1.0)
        
        with open(self.metrics_file, "r") as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 1)
        entry = data[0]
        self.assertEqual(entry["model"], "test_model")
        self.assertEqual(entry["response_time"], 1.0)
        self.assertIn("timestamp", entry)
        self.assertNotIn("prompt_length", entry)
        self.assertNotIn("input_tokens", entry)
        self.assertNotIn("output_tokens", entry)
    
    def test_record_metrics_full_data(self):
        """Test recording metrics with all optional data"""
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Patch to return our test file
        def mock_path_join(*args):
            if len(args) == 2 and args[0] == "data" and args[1] == "performance_metrics.json":
                return self.metrics_file
            return os.path.join(*args)
        
        with patch('os.path.join', side_effect=mock_path_join):
            record_metrics("test_model", 1.0, prompt_length=200, input_tokens=100, output_tokens=150)
        
        with open(self.metrics_file, "r") as f:
            data = json.load(f)
        
        entry = data[0]
        self.assertEqual(entry["model"], "test_model")
        self.assertEqual(entry["response_time"], 1.0)
        self.assertEqual(entry["prompt_length"], 200)
        self.assertEqual(entry["input_tokens"], 100)
        self.assertEqual(entry["output_tokens"], 150)
        self.assertIn("timestamp", entry)


class TestDataProcessingEdgeCases(TestCase):
    """Test edge cases in data processing"""
    
    @patch('streamlit.plotly_chart')
    @patch('streamlit.subheader')
    @patch('ollama_workbench.server.performance_metrics.load_metrics_data')
    def test_response_time_metrics_with_invalid_timestamps(self, mock_load, mock_subheader, mock_plotly):
        """Test handling of invalid timestamp formats"""
        invalid_timestamp_metrics = [
            {
                "timestamp": "invalid-timestamp",
                "model": "llama3",
                "response_time": 2.5,
                "prompt_length": 100
            },
            {
                "timestamp": "2024-01-01T10:00:00",  # Valid timestamp
                "model": "gpt-4",
                "response_time": 1.8,
                "prompt_length": 120
            }
        ]
        mock_load.return_value = invalid_timestamp_metrics
        
        # Should handle invalid timestamps gracefully
        try:
            display_response_time_metrics()
            # If no exception, test passes
        except Exception as e:
            self.fail(f"Function should handle invalid timestamps gracefully, but raised: {e}")
    
    @patch('streamlit.plotly_chart')
    @patch('streamlit.subheader')
    @patch('ollama_workbench.server.performance_metrics.load_metrics_data')
    def test_token_usage_metrics_with_negative_values(self, mock_load, mock_subheader, mock_plotly):
        """Test handling of negative token values"""
        negative_token_metrics = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "model": "llama3",
                "input_tokens": -10,  # Negative value
                "output_tokens": 75
            },
            {
                "timestamp": "2024-01-01T11:00:00",
                "model": "gpt-4",
                "input_tokens": 60,
                "output_tokens": -5  # Negative value
            }
        ]
        mock_load.return_value = negative_token_metrics
        
        # Should handle negative values gracefully
        try:
            display_token_usage_metrics()
            # If no exception, test passes
        except Exception as e:
            self.fail(f"Function should handle negative values gracefully, but raised: {e}")
    
    @patch('streamlit.plotly_chart')
    @patch('streamlit.dataframe')
    @patch('streamlit.subheader')
    @patch('ollama_workbench.server.performance_metrics.load_metrics_data')
    def test_model_comparison_with_single_model(self, mock_load, mock_subheader, mock_dataframe, mock_plotly):
        """Test model comparison with only one model"""
        single_model_metrics = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "model": "llama3",
                "response_time": 2.5,
                "input_tokens": 50,
                "output_tokens": 75
            }
        ]
        mock_load.return_value = single_model_metrics
        
        display_model_comparison()
        
        # Should still work with single model
        mock_dataframe.assert_called_once()
        self.assertEqual(mock_plotly.call_count, 2)


class TestIntegrationScenarios(TestCase):
    """Test integration scenarios and complex use cases"""
    
    def test_timestamp_handling_consistency(self):
        """Test that timestamp handling is consistent across functions"""
        test_time = datetime.now()
        timestamp_str = test_time.isoformat()
        
        # Test recording with specific timestamp format
        temp_dir = tempfile.mkdtemp()
        metrics_dir = os.path.join(temp_dir, "data")
        metrics_file = os.path.join(metrics_dir, "performance_metrics.json")
        
        try:
            # Patch to return our test file
            def mock_path_join(*args):
                if len(args) == 2 and args[0] == "data" and args[1] == "performance_metrics.json":
                    return metrics_file
                return os.path.join(*args)
            
            # Store original makedirs function
            original_makedirs = os.makedirs
            
            def mock_makedirs(path, exist_ok=False):
                if path == "data":
                    original_makedirs(metrics_dir, exist_ok=True)
                else:
                    original_makedirs(path, exist_ok=exist_ok)
            
            with patch('os.path.join', side_effect=mock_path_join), \
                 patch('os.makedirs', side_effect=mock_makedirs):
                record_metrics("test_model", 1.0)
            
            # Load and verify timestamp can be processed
            with open(metrics_file, "r") as f:
                data = json.load(f)
            
            # Should be able to parse timestamp
            parsed_time = datetime.fromisoformat(data[0]["timestamp"])
            self.assertIsInstance(parsed_time, datetime)
            
        finally:
            shutil.rmtree(temp_dir)
    
    @patch('ollama_workbench.server.performance_metrics.load_metrics_data')
    def test_large_dataset_handling(self, mock_load):
        """Test handling of large datasets"""
        # Create large dataset
        large_dataset = []
        for i in range(1000):
            large_dataset.append({
                "timestamp": f"2024-01-{(i % 30) + 1:02d}T10:00:00",
                "model": f"model_{i % 3}",
                "response_time": 1.0 + (i % 10) * 0.5,
                "input_tokens": 50 + (i % 100),
                "output_tokens": 75 + (i % 150),
                "prompt_length": 100 + (i % 200)
            })
        
        mock_load.return_value = large_dataset
        
        # Should handle large datasets without errors
        with patch('streamlit.plotly_chart'), patch('streamlit.subheader'):
            try:
                display_response_time_metrics()
                display_token_usage_metrics()
                display_model_comparison()
            except Exception as e:
                self.fail(f"Functions should handle large datasets gracefully, but raised: {e}")
    
    def test_concurrent_file_access_simulation(self):
        """Test file access patterns that might occur with concurrent usage"""
        temp_dir = tempfile.mkdtemp()
        metrics_dir = os.path.join(temp_dir, "data")
        metrics_file = os.path.join(metrics_dir, "performance_metrics.json")
        
        try:
            # Patch to return our test file
            def mock_path_join(*args):
                if len(args) == 2 and args[0] == "data" and args[1] == "performance_metrics.json":
                    return metrics_file
                return os.path.join(*args)
            
            # Store original makedirs function
            original_makedirs = os.makedirs
            
            def mock_makedirs(path, exist_ok=False):
                if path == "data":
                    original_makedirs(metrics_dir, exist_ok=True)
                else:
                    original_makedirs(path, exist_ok=exist_ok)
            
            # Simulate multiple concurrent writes
            with patch('os.path.join', side_effect=mock_path_join), \
                 patch('os.makedirs', side_effect=mock_makedirs):
                record_metrics("model1", 1.0)
                record_metrics("model2", 2.0)
                record_metrics("model3", 3.0)
            
            # Verify all data was recorded
            with open(metrics_file, "r") as f:
                data = json.load(f)
            
            self.assertEqual(len(data), 3)
            models = [entry["model"] for entry in data]
            self.assertIn("model1", models)
            self.assertIn("model2", models)
            self.assertIn("model3", models)
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
