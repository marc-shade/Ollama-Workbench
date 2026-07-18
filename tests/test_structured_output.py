"""
Test suite for structured_output.py - Structured output generation functionality
"""

import os
import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock, mock_open, call
import sys
import pandas as pd

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _AttrDict(dict):
    """Attribute-access dict mirroring Streamlit's session_state (same shape as conftest.AttrDict)."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


class TestDefaultSchemas:
    """Test default JSON schemas"""
    
    def test_default_schemas_exist(self):
        """Test that default schemas are properly defined"""
        from ollama_workbench.ui.structured_output import DEFAULT_SCHEMAS
        
        expected_schemas = [
            "person_details", "product_info", "event", "article_summary",
            "data_analysis", "recipe", "custom"
        ]
        
        for schema_name in expected_schemas:
            assert schema_name in DEFAULT_SCHEMAS
            assert isinstance(DEFAULT_SCHEMAS[schema_name], dict)
            assert "type" in DEFAULT_SCHEMAS[schema_name]
            assert "properties" in DEFAULT_SCHEMAS[schema_name]
    
    def test_person_details_schema_structure(self):
        """Test person details schema structure"""
        from ollama_workbench.ui.structured_output import DEFAULT_SCHEMAS
        
        schema = DEFAULT_SCHEMAS["person_details"]
        
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert "email" in schema["properties"]
        assert "address" in schema["properties"]
        
        # Check required fields
        assert "required" in schema
        assert "name" in schema["required"]
        assert "age" in schema["required"]
        assert "email" in schema["required"]
        
        # Check nested address object
        address = schema["properties"]["address"]
        assert address["type"] == "object"
        assert "street" in address["properties"]
        assert "city" in address["properties"]
    
    def test_product_info_schema_structure(self):
        """Test product info schema structure"""
        from ollama_workbench.ui.structured_output import DEFAULT_SCHEMAS
        
        schema = DEFAULT_SCHEMAS["product_info"]
        
        assert schema["type"] == "object"
        assert "id" in schema["properties"]
        assert "name" in schema["properties"]
        assert "price" in schema["properties"]
        assert "currency" in schema["properties"]
        
        # Check enum constraint on currency
        currency = schema["properties"]["currency"]
        assert "enum" in currency
        assert "USD" in currency["enum"]
        assert "EUR" in currency["enum"]
        
        # Check array of attributes
        attributes = schema["properties"]["attributes"]
        assert attributes["type"] == "array"
        assert "items" in attributes
    
    def test_event_schema_structure(self):
        """Test event schema structure"""
        from ollama_workbench.ui.structured_output import DEFAULT_SCHEMAS
        
        schema = DEFAULT_SCHEMAS["event"]
        
        assert schema["type"] == "object"
        assert "title" in schema["properties"]
        assert "start_date" in schema["properties"]
        assert "location" in schema["properties"]
        
        # Check date format
        start_date = schema["properties"]["start_date"]
        assert start_date["format"] == "date-time"
        
        # Check nested location object with coordinates
        location = schema["properties"]["location"]
        assert "coordinates" in location["properties"]
        coordinates = location["properties"]["coordinates"]
        assert "latitude" in coordinates["properties"]
        assert "longitude" in coordinates["properties"]
    
    def test_recipe_schema_complexity(self):
        """Test complex recipe schema"""
        from ollama_workbench.ui.structured_output import DEFAULT_SCHEMAS
        
        schema = DEFAULT_SCHEMAS["recipe"]
        
        # Check ingredients array structure
        ingredients = schema["properties"]["ingredients"]
        assert ingredients["type"] == "array"
        ingredient_item = ingredients["items"]
        assert "name" in ingredient_item["properties"]
        assert "quantity" in ingredient_item["properties"]
        assert "unit" in ingredient_item["properties"]
        
        # Check instructions array structure
        instructions = schema["properties"]["instructions"]
        assert instructions["type"] == "array"
        instruction_item = instructions["items"]
        assert "step" in instruction_item["properties"]
        assert "description" in instruction_item["properties"]
        
        # Check nutrition facts object
        nutrition = schema["properties"]["nutrition_facts"]
        assert nutrition["type"] == "object"
        assert "calories" in nutrition["properties"]
        assert "protein" in nutrition["properties"]
    
    def test_data_analysis_schema_complexity(self):
        """Test complex data analysis schema"""
        from ollama_workbench.ui.structured_output import DEFAULT_SCHEMAS
        
        schema = DEFAULT_SCHEMAS["data_analysis"]
        
        # Check summary statistics structure
        summary_stats = schema["properties"]["summary_statistics"]
        assert "variables" in summary_stats["properties"]
        
        variables = summary_stats["properties"]["variables"]
        assert variables["type"] == "array"
        
        variable_item = variables["items"]
        assert "type" in variable_item["properties"]
        assert "metrics" in variable_item["properties"]
        
        # Check variable type enum
        var_type = variable_item["properties"]["type"]
        assert "enum" in var_type
        assert "numerical" in var_type["enum"]
        assert "categorical" in var_type["enum"]


class TestJSONEditor:
    """Test JSON schema editor functionality"""
    
    @pytest.fixture
    def mock_streamlit(self):
        """Mock Streamlit components"""
        with patch('ollama_workbench.ui.structured_output.st') as mock_st:
            mock_st.text_area = Mock()
            mock_st.error = Mock()
            yield mock_st
    
    def test_json_editor_valid_json(self, mock_streamlit):
        """Test JSON editor with valid JSON"""
        from ollama_workbench.ui.structured_output import json_editor
        
        schema_data = {"type": "object", "properties": {"name": {"type": "string"}}}
        mock_streamlit.text_area.return_value = '{"type": "object", "properties": {"name": {"type": "string"}}}'
        
        result = json_editor(schema_data, "test_key")
        
        assert result == schema_data
        mock_streamlit.text_area.assert_called_once()
    
    def test_json_editor_invalid_json(self, mock_streamlit):
        """Test JSON editor with invalid JSON"""
        from ollama_workbench.ui.structured_output import json_editor
        
        schema_data = {"type": "object"}
        mock_streamlit.text_area.return_value = '{"type": "object", invalid json'
        
        result = json_editor(schema_data, "test_key")
        
        # Should return original data and show error
        assert result == schema_data
        mock_streamlit.error.assert_called_once()
        assert "Invalid JSON" in mock_streamlit.error.call_args[0][0]
    
    def test_json_editor_modified_schema(self, mock_streamlit):
        """Test JSON editor with modified schema"""
        from ollama_workbench.ui.structured_output import json_editor
        
        original_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        modified_schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
        
        mock_streamlit.text_area.return_value = json.dumps(modified_schema)
        
        result = json_editor(original_schema, "test_key")
        
        assert result == modified_schema
        assert "age" in result["properties"]


class TestSchemaVisualization:
    """Test schema visualization functionality"""
    
    @pytest.fixture
    def mock_streamlit(self):
        """Mock Streamlit components"""
        with patch('ollama_workbench.ui.structured_output.st') as mock_st:
            mock_st.json = Mock()
            mock_st.markdown = Mock()
            mock_st.warning = Mock()
            mock_st.info = Mock()
            yield mock_st
    
    @patch('ollama_workbench.ui.structured_output.HAS_JSF', True)
    @patch('ollama_workbench.ui.structured_output.jsf')
    def test_visualize_schema_with_jsf(self, mock_jsf, mock_streamlit):
        """Test schema visualization with json-schema-for-humans"""
        from ollama_workbench.ui.structured_output import visualize_schema
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        mock_jsf.generate_from_schema.return_value = "<h1>Schema HTML</h1>"
        
        visualize_schema(schema)
        
        mock_jsf.generate_from_schema.assert_called_once_with(schema, "md")
        mock_streamlit.markdown.assert_called_once_with("<h1>Schema HTML</h1>", unsafe_allow_html=True)
    
    @patch('ollama_workbench.ui.structured_output.HAS_JSF', True)
    @patch('ollama_workbench.ui.structured_output.jsf')
    def test_visualize_schema_jsf_error(self, mock_jsf, mock_streamlit):
        """Test schema visualization with JSF error"""
        from ollama_workbench.ui.structured_output import visualize_schema
        
        schema = {"type": "object"}
        mock_jsf.generate_from_schema.side_effect = Exception("JSF error")
        
        visualize_schema(schema)
        
        # Should fall back to basic visualization
        mock_streamlit.json.assert_called_once_with(schema)
        mock_streamlit.warning.assert_called_once()
    
    @patch('ollama_workbench.ui.structured_output.HAS_JSF', False)
    def test_visualize_schema_no_jsf(self, mock_streamlit):
        """Test schema visualization without json-schema-for-humans"""
        from ollama_workbench.ui.structured_output import visualize_schema
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        
        visualize_schema(schema)
        
        # Should use basic fallback
        mock_streamlit.json.assert_called_once_with(schema)
        mock_streamlit.info.assert_called_once()


class TestJSONGeneration:
    """Test JSON generation from text"""
    
    @patch('ollama_workbench.ui.structured_output.logger')
    @patch('subprocess.run')
    def test_generate_json_cli_success(self, mock_subprocess, mock_logger):
        """Test successful JSON generation via CLI"""
        from ollama_workbench.ui.structured_output import generate_json_from_text
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = '{"name": "John Doe"}'
        mock_subprocess.return_value = mock_result
        
        result = generate_json_from_text("John Doe is a person", schema, "llama3")
        
        assert result == {"name": "John Doe"}
        mock_subprocess.assert_called_once()
    
    @patch('ollama_workbench.ui.structured_output.logger')
    @patch('subprocess.run')
    @patch('ollama_workbench.providers.ollama_utils.get_ollama_client')
    def test_generate_json_cli_fallback_to_api(self, mock_get_client, mock_subprocess, mock_logger):
        """Test JSON generation falling back from CLI to API"""
        from ollama_workbench.ui.structured_output import generate_json_from_text
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        
        # CLI fails
        mock_subprocess.side_effect = Exception("CLI failed")
        
        # API succeeds
        mock_client = Mock()
        mock_client.generate.return_value = {"response": '{"name": "Jane Smith"}'}
        mock_get_client.return_value = mock_client
        
        result = generate_json_from_text("Jane Smith is a person", schema, "llama3")
        
        assert result == {"name": "Jane Smith"}
        mock_client.generate.assert_called_once()
    
    @patch('ollama_workbench.ui.structured_output.logger')
    @patch('subprocess.run')
    @patch('ollama_workbench.providers.ollama_utils.get_ollama_client')
    @patch('ollama_workbench.providers.ollama_utils.call_ollama_endpoint')
    def test_generate_json_final_fallback(self, mock_call_endpoint, mock_get_client, 
                                         mock_subprocess, mock_logger):
        """Test JSON generation with final fallback"""
        from ollama_workbench.ui.structured_output import generate_json_from_text
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        
        # CLI fails
        mock_subprocess.side_effect = Exception("CLI failed")
        
        # Client API fails
        mock_get_client.return_value = None
        
        # Final fallback succeeds (call_ollama_endpoint returns a 5-tuple:
        # response_text, context, eval_count, eval_duration, metrics_dict)
        mock_call_endpoint.return_value = ('{"name": "Bob Wilson"}', None, None, None, None)
        
        result = generate_json_from_text("Bob Wilson is a person", schema, "llama3")
        
        assert result == {"name": "Bob Wilson"}
        mock_call_endpoint.assert_called_once()
    
    @patch('ollama_workbench.ui.structured_output.logger')
    def test_generate_json_no_model(self, mock_logger):
        """Test JSON generation with no model"""
        from ollama_workbench.ui.structured_output import generate_json_from_text
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        
        result = generate_json_from_text("Test text", schema, "")
        
        assert result == {}
    
    @patch('ollama_workbench.ui.structured_output.logger')
    @patch('subprocess.run')
    def test_generate_json_malformed_response(self, mock_subprocess, mock_logger):
        """Test JSON generation with malformed response"""
        from ollama_workbench.ui.structured_output import generate_json_from_text
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = 'This is not JSON at all'
        mock_subprocess.return_value = mock_result
        
        result = generate_json_from_text("Test text", schema, "llama3")
        
        assert result == {}
    
    @patch('ollama_workbench.ui.structured_output.logger')
    @patch('subprocess.run')
    def test_generate_json_partial_json(self, mock_subprocess, mock_logger):
        """Test JSON generation with partial JSON in response"""
        from ollama_workbench.ui.structured_output import generate_json_from_text
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = 'Here is the JSON: {"name": "Alice"} with some extra text'
        mock_subprocess.return_value = mock_result
        
        result = generate_json_from_text("Alice is a person", schema, "llama3")
        
        assert result == {"name": "Alice"}
    
    @patch('ollama_workbench.ui.structured_output.logger')
    @patch('subprocess.run')
    def test_generate_json_single_quotes_fix(self, mock_subprocess, mock_logger):
        """Test JSON generation with single quotes that get fixed"""
        from ollama_workbench.ui.structured_output import generate_json_from_text
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "{'name': 'Charlie'}"
        mock_subprocess.return_value = mock_result
        
        result = generate_json_from_text("Charlie is a person", schema, "llama3")
        
        assert result == {"name": "Charlie"}


class TestStreamlitUI:
    """Test Streamlit UI functionality"""
    
    @pytest.fixture
    def mock_streamlit(self):
        """Mock Streamlit components"""
        with patch('ollama_workbench.ui.structured_output.st') as mock_st:
            mock_st.title = Mock()
            mock_st.write = Mock()
            mock_st.selectbox = Mock()
            mock_st.text_area = Mock()
            mock_st.button = Mock()
            mock_st.columns = Mock()
            mock_st.subheader = Mock()
            mock_st.slider = Mock()
            mock_st.radio = Mock()
            mock_st.info = Mock()
            mock_st.error = Mock()
            mock_st.success = Mock()
            mock_st.spinner = Mock()
            mock_st.json = Mock()
            mock_st.table = Mock()
            mock_st.code = Mock()
            mock_st.markdown = Mock()
            mock_st.expander = Mock()
            mock_st.rerun = Mock()
            mock_st.session_state = _AttrDict()
            yield mock_st
    
    @patch('ollama_workbench.ui.structured_output.get_available_models')
    def test_structured_output_ui_basic_setup(self, mock_get_models, mock_streamlit):
        """Test basic UI setup"""
        from ollama_workbench.ui.structured_output import structured_output_ui
        
        mock_get_models.return_value = ["llama3", "mistral"]
        # Columns are used as context managers; MagicMock supports the protocol.
        mock_streamlit.columns.return_value = [MagicMock(), MagicMock()]
        mock_streamlit.expander.return_value.__enter__ = Mock()
        mock_streamlit.expander.return_value.__exit__ = Mock()
        mock_streamlit.selectbox.side_effect = ["person_details", "llama3"]
        mock_streamlit.text_area.side_effect = [
            '{"title": "Person Details", "type": "object", "properties": {"name": {"type": "string"}}}',
            ""  # No input text: generation stays untriggered
        ]
        mock_streamlit.button.return_value = False

        structured_output_ui()

        mock_streamlit.title.assert_called_once_with("🔍 Structured Output Generator")
        mock_get_models.assert_called_once()
    
    @patch('ollama_workbench.ui.structured_output.get_available_models', return_value=[])
    @patch('subprocess.run')
    def test_structured_output_ui_no_models_cli_fallback(self, mock_subprocess, mock_get_models, mock_streamlit):
        """Test UI with no models from API, using CLI fallback"""
        from ollama_workbench.ui.structured_output import structured_output_ui
        
        # Mock CLI command returning models
        mock_result = Mock()
        mock_result.stdout = "NAME\nllama3:latest\nmistral:latest\n"
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        mock_streamlit.columns.return_value = [MagicMock(), MagicMock()]
        mock_streamlit.expander.return_value.__enter__ = Mock()
        mock_streamlit.expander.return_value.__exit__ = Mock()
        mock_streamlit.selectbox.side_effect = ["person_details", "llama3:latest"]
        mock_streamlit.text_area.side_effect = [
            '{"title": "Person Details", "type": "object", "properties": {"name": {"type": "string"}}}',
            ""
        ]
        mock_streamlit.button.return_value = False

        structured_output_ui()
        
        # Should continue with CLI-discovered models
        mock_subprocess.assert_called_once()
        mock_streamlit.error.assert_not_called()
    
    @patch('ollama_workbench.ui.structured_output.get_available_models', return_value=[])
    @patch('subprocess.run')
    def test_structured_output_ui_no_models_available(self, mock_subprocess, mock_get_models, mock_streamlit):
        """Test UI with no models available"""
        from ollama_workbench.ui.structured_output import structured_output_ui
        
        # Both API and CLI return no models
        mock_result = Mock()
        mock_result.stdout = "NAME\n"  # Empty except header
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        structured_output_ui()
        
        mock_streamlit.error.assert_called_once()
        error_message = mock_streamlit.error.call_args[0][0]
        assert "No Ollama models found" in error_message
    
    @patch('ollama_workbench.ui.structured_output.get_available_models')
    @patch('ollama_workbench.ui.structured_output.generate_json_from_text')
    def test_structured_output_ui_generation(self, mock_generate, mock_get_models, mock_streamlit):
        """Test output generation in UI"""
        from ollama_workbench.ui.structured_output import structured_output_ui
        
        mock_get_models.return_value = ["llama3"]
        mock_generate.return_value = {"name": "Test User", "age": 30}
        
        # Mock UI interactions
        mock_streamlit.columns.return_value = [MagicMock(), MagicMock()]
        mock_streamlit.selectbox.side_effect = ["person_details", "llama3"]
        mock_streamlit.text_area.side_effect = [
            '{"title": "Person Details", "type": "object", "properties": {"name": {"type": "string"}}}',  # Schema editor
            "Test user information"  # Input text
        ]
        mock_streamlit.slider.return_value = 0.7
        # Click only the Generate button; other buttons (Save as Custom Schema,
        # Copy JSON, View Result) must stay unclicked.
        mock_streamlit.button.side_effect = lambda label, *a, **k: label == "Generate Structured Output"
        mock_streamlit.radio.return_value = "JSON"
        
        # Mock expander context manager
        mock_expander = Mock()
        mock_expander.__enter__ = Mock(return_value=mock_expander)
        mock_expander.__exit__ = Mock(return_value=None)
        mock_streamlit.expander.return_value = mock_expander
        
        # Mock spinner context manager
        mock_spinner = Mock()
        mock_spinner.__enter__ = Mock(return_value=mock_spinner)
        mock_spinner.__exit__ = Mock(return_value=None)
        mock_streamlit.spinner.return_value = mock_spinner
        
        structured_output_ui()
        
        # Should call generation function
        mock_generate.assert_called_once()
        
        # Should display result
        mock_streamlit.json.assert_called()
    
    @patch('ollama_workbench.ui.structured_output.get_available_models')
    @patch('ollama_workbench.ui.structured_output.generate_json_from_text')
    def test_structured_output_ui_table_format(self, mock_generate, mock_get_models, mock_streamlit):
        """Test table format output"""
        from ollama_workbench.ui.structured_output import structured_output_ui
        
        mock_get_models.return_value = ["llama3"]
        mock_generate.return_value = {"name": "Test User", "age": 30, "address": {"city": "New York"}}
        
        # Mock UI interactions for table format
        mock_streamlit.columns.return_value = [MagicMock(), MagicMock()]
        mock_streamlit.selectbox.side_effect = ["person_details", "llama3"]
        mock_streamlit.text_area.side_effect = [
            '{"title": "Person Details", "type": "object", "properties": {"name": {"type": "string"}}}',
            "Test user"
        ]
        mock_streamlit.slider.return_value = 0.7
        mock_streamlit.button.side_effect = lambda label, *a, **k: label == "Generate Structured Output"
        mock_streamlit.radio.return_value = "Table"

        # Mock session state with result (attribute-access dict like st.session_state)
        mock_streamlit.session_state = _AttrDict({
            "structured_output_result": {"name": "Test User", "age": 30, "address": {"city": "New York"}},
            "structured_output_history": []
        })
        
        # Mock expander and spinner
        mock_expander = Mock()
        mock_expander.__enter__ = Mock(return_value=mock_expander)
        mock_expander.__exit__ = Mock(return_value=None)
        mock_streamlit.expander.return_value = mock_expander
        
        mock_spinner = Mock()
        mock_spinner.__enter__ = Mock(return_value=mock_spinner)
        mock_spinner.__exit__ = Mock(return_value=None)
        mock_streamlit.spinner.return_value = mock_spinner
        
        with patch('ollama_workbench.ui.structured_output.pd.DataFrame') as mock_df:
            mock_df.return_value = Mock()
            structured_output_ui()
            
            # Should create DataFrame and show table
            mock_df.assert_called_once()
            mock_streamlit.table.assert_called_once()
    
    @patch('ollama_workbench.ui.structured_output.get_available_models')
    def test_structured_output_ui_history_functionality(self, mock_get_models, mock_streamlit):
        """Test history functionality in UI"""
        from ollama_workbench.ui.structured_output import structured_output_ui
        
        mock_get_models.return_value = ["llama3"]
        
        # Mock session state with history (attribute-access dict like st.session_state)
        mock_streamlit.session_state = _AttrDict({
            "structured_output_result": {"name": "Current User"},
            "structured_output_history": [
                {
                    "input": "Test input text",
                    "schema": "Person Details",
                    "result": {"name": "Historical User"},
                    "timestamp": "2024-01-01 12:00:00"
                }
            ]
        })

        mock_streamlit.columns.return_value = [MagicMock(), MagicMock()]
        mock_streamlit.selectbox.side_effect = ["person_details", "llama3"]
        mock_streamlit.text_area.side_effect = [
            '{"title": "Person Details", "type": "object", "properties": {"name": {"type": "string"}}}',
            ""
        ]
        mock_streamlit.slider.return_value = 0.7
        # Click only the history View button; Generate and the other buttons stay unclicked.
        mock_streamlit.button.side_effect = lambda label, *a, **k: label == "View Result"
        mock_streamlit.radio.return_value = "JSON"
        
        # Mock expander context manager
        mock_expander = Mock()
        mock_expander.__enter__ = Mock(return_value=mock_expander)
        mock_expander.__exit__ = Mock(return_value=None)
        mock_streamlit.expander.return_value = mock_expander
        
        structured_output_ui()
        
        # Should show history and handle view button
        mock_streamlit.markdown.assert_called()
        mock_streamlit.rerun.assert_called_once()


class TestErrorHandling:
    """Test error handling scenarios"""
    
    @patch('ollama_workbench.ui.structured_output.st')
    @patch('ollama_workbench.ui.structured_output.logger')
    @patch('ollama_workbench.ui.structured_output.get_available_models')
    def test_ui_model_fetch_error(self, mock_get_models, mock_logger, mock_st):
        """Test UI error handling when model fetch fails"""
        from ollama_workbench.ui.structured_output import structured_output_ui
        
        mock_get_models.side_effect = Exception("API error")
        
        # Mock subprocess to also fail
        with patch('subprocess.run', side_effect=Exception("CLI error")):
            structured_output_ui()
        
        mock_st.error.assert_called()
        error_message = mock_st.error.call_args[0][0]
        assert "Error fetching models" in error_message
    
    @patch('ollama_workbench.ui.structured_output.st')
    @patch('ollama_workbench.ui.structured_output.logger')
    @patch('ollama_workbench.ui.structured_output.get_available_models')
    @patch('ollama_workbench.ui.structured_output.generate_json_from_text')
    def test_ui_generation_error(self, mock_generate, mock_get_models, mock_logger, mock_st):
        """Test UI error handling during generation"""
        from ollama_workbench.ui.structured_output import structured_output_ui
        
        mock_get_models.return_value = ["llama3"]
        mock_generate.side_effect = Exception("Generation failed")
        
        # Mock UI state for generation
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.side_effect = ["person_details", "llama3"]
        mock_st.text_area.side_effect = [
            '{"title": "Person Details", "type": "object", "properties": {"name": {"type": "string"}}}',
            "Test text"
        ]
        mock_st.slider.return_value = 0.7
        mock_st.button.side_effect = lambda label, *a, **k: label == "Generate Structured Output"
        mock_st.session_state = _AttrDict()
        
        # Mock spinner context manager
        mock_spinner = Mock()
        mock_spinner.__enter__ = Mock(return_value=mock_spinner)
        mock_spinner.__exit__ = Mock(return_value=None)
        mock_st.spinner.return_value = mock_spinner
        
        # Mock expander context manager
        mock_expander = Mock()
        mock_expander.__enter__ = Mock(return_value=mock_expander)
        mock_expander.__exit__ = Mock(return_value=None)
        mock_st.expander.return_value = mock_expander
        
        structured_output_ui()
        
        mock_st.error.assert_called()
        mock_logger.error.assert_called()


class TestIntegration:
    """Test integration scenarios"""
    
    def test_module_imports(self):
        """Test that all required modules can be imported"""
        import ollama_workbench.ui.structured_output as structured_output

        
        # Test that main functions exist
        assert hasattr(structured_output, 'DEFAULT_SCHEMAS')
        assert hasattr(structured_output, 'json_editor')
        assert hasattr(structured_output, 'visualize_schema')
        assert hasattr(structured_output, 'generate_json_from_text')
        assert hasattr(structured_output, 'structured_output_ui')
    
    def test_schema_to_generation_workflow(self):
        """Test complete workflow from schema to generation"""
        from ollama_workbench.ui.structured_output import DEFAULT_SCHEMAS, generate_json_from_text
        
        schema = DEFAULT_SCHEMAS["person_details"]
        
        # Mock the generation process
        with patch('subprocess.run') as mock_subprocess:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = '{"name": "Integration Test", "age": 25, "email": "test@example.com"}'
            mock_subprocess.return_value = mock_result
            
            result = generate_json_from_text(
                "Integration Test is 25 years old with email test@example.com",
                schema,
                "llama3"
            )
        
        # Should return properly structured result
        assert isinstance(result, dict)
        assert "name" in result
        assert "age" in result
        assert "email" in result
    
    def test_json_editor_to_visualization_workflow(self):
        """Test workflow from editing to visualization"""
        from ollama_workbench.ui.structured_output import json_editor, visualize_schema
        
        original_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        
        # Mock editing
        with patch('ollama_workbench.ui.structured_output.st') as mock_st:
            mock_st.text_area.return_value = json.dumps({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                }
            })
            mock_st.json = Mock()
            mock_st.info = Mock()
            
            # Edit schema
            edited_schema = json_editor(original_schema, "test")
            
            # Visualize edited schema
            with patch('ollama_workbench.ui.structured_output.HAS_JSF', False):
                visualize_schema(edited_schema)
        
        # Should have added age property
        assert "age" in edited_schema["properties"]
        mock_st.json.assert_called_once_with(edited_schema)


class TestPerformance:
    """Test performance-related functionality"""
    
    def test_large_schema_handling(self):
        """Test handling of large, complex schemas"""
        from ollama_workbench.ui.structured_output import json_editor, DEFAULT_SCHEMAS
        
        # Use the most complex schema
        complex_schema = DEFAULT_SCHEMAS["data_analysis"]
        
        with patch('ollama_workbench.ui.structured_output.st') as mock_st:
            mock_st.text_area.return_value = json.dumps(complex_schema)
            
            # Should handle large schema without issues
            result = json_editor(complex_schema, "performance_test")
            
            assert result == complex_schema
            # Should not timeout or cause memory issues
    
    def test_history_memory_management(self):
        """Test memory management with large history"""
        from ollama_workbench.ui.structured_output import structured_output_ui
        
        # Create large history
        large_history = []
        for i in range(100):
            large_history.append({
                "input": f"Test input {i}" * 10,  # Large input
                "schema": "Test Schema",
                "result": {"data": f"result_{i}" * 20},  # Large result
                "timestamp": f"2024-01-01 12:{i:02d}:00"
            })
        
        with patch('ollama_workbench.ui.structured_output.st') as mock_st:
            with patch('ollama_workbench.ui.structured_output.get_available_models', return_value=["llama3"]):
                mock_st.session_state = _AttrDict({
                    "structured_output_history": large_history,
                    "structured_output_result": {}
                })
                mock_st.columns.return_value = [MagicMock(), MagicMock()]
                mock_st.selectbox.side_effect = ["person_details", "llama3"]
                mock_st.text_area.side_effect = ['{"type": "object"}', ""]
                mock_st.slider.return_value = 0.7
                mock_st.button.return_value = False
                mock_st.radio.return_value = "JSON"
                
                # Mock expander context manager
                mock_expander = Mock()
                mock_expander.__enter__ = Mock(return_value=mock_expander)
                mock_expander.__exit__ = Mock(return_value=None)
                mock_st.expander.return_value = mock_expander
                
                # Should handle large history without performance issues
                try:
                    structured_output_ui()
                    # If we get here, it handled the large history appropriately
                    assert True
                except Exception as e:
                    pytest.fail(f"Failed to handle large history: {e}")


class TestSchemaValidation:
    """Test schema validation functionality"""
    
    def test_schema_property_types(self):
        """Test that all default schemas have valid property types"""
        from ollama_workbench.ui.structured_output import DEFAULT_SCHEMAS
        
        valid_types = ["string", "number", "integer", "boolean", "array", "object"]
        
        for schema_name, schema in DEFAULT_SCHEMAS.items():
            if schema_name == "custom":
                continue  # Skip custom schema
                
            assert "properties" in schema
            
            def check_properties(props):
                for prop_name, prop_def in props.items():
                    assert "type" in prop_def
                    assert prop_def["type"] in valid_types
                    
                    # Recursively check nested objects
                    if prop_def["type"] == "object" and "properties" in prop_def:
                        check_properties(prop_def["properties"])
                    
                    # Check array items
                    if prop_def["type"] == "array" and "items" in prop_def:
                        items = prop_def["items"]
                        if "type" in items:
                            assert items["type"] in valid_types
                            if items["type"] == "object" and "properties" in items:
                                check_properties(items["properties"])
            
            check_properties(schema["properties"])
    
    def test_required_fields_exist(self):
        """Test that required fields exist in properties"""
        from ollama_workbench.ui.structured_output import DEFAULT_SCHEMAS
        
        for schema_name, schema in DEFAULT_SCHEMAS.items():
            if schema_name == "custom":
                continue
                
            if "required" in schema:
                properties = schema["properties"]
                for required_field in schema["required"]:
                    assert required_field in properties, f"Required field '{required_field}' not in properties for schema '{schema_name}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
