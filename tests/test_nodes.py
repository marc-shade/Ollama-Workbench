"""
Test suite for nodes.py - Visual workflow builder
"""

import pytest
import json
import os
import tempfile
import base64
from unittest.mock import Mock, patch, MagicMock, call, mock_open
from pathlib import Path
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNode:
    """Test Node class"""
    
    def test_node_creation(self):
        """Test Node creation"""
        from nodes import Node
        
        data = {"content": "Test Node", "input_type": "Text"}
        node = Node("1", "Input", data)
        
        assert node.id == "1"
        assert node.type == "Input"
        assert node.data == data
    
    def test_node_to_dict(self):
        """Test Node to_dict method"""
        from nodes import Node
        
        data = {"content": "Test Node", "model": "gpt-4"}
        node = Node("2", "LLM", data)
        
        result = node.to_dict()
        
        expected = {
            "id": "2",
            "type": "LLM",
            "data": data
        }
        assert result == expected
    
    @patch('nodes.create_node')
    def test_node_from_dict(self, mock_create_node):
        """Test Node from_dict method"""
        from nodes import Node
        
        # Setup mock
        default_node = Mock()
        default_node.data = {"default": "value", "content": "Default"}
        mock_create_node.return_value = default_node
        
        # Test data
        node_dict = {
            "id": "3",
            "type": "Processing",
            "data": {"content": "Custom Node", "processing_type": "Logic"}
        }
        
        result = Node.from_dict(node_dict)
        
        assert result.id == "3"
        assert result.type == "Processing"
        assert result.data["content"] == "Custom Node"
        assert result.data["processing_type"] == "Logic"
        assert result.data["default"] == "value"  # Merged from default
        
        mock_create_node.assert_called_once_with("temp", "Processing")
    
    @patch('nodes.create_node')
    def test_node_from_dict_no_data(self, mock_create_node):
        """Test Node from_dict with no data field"""
        from nodes import Node
        
        # Setup mock
        default_node = Mock()
        default_node.data = {"default": "value"}
        mock_create_node.return_value = default_node
        
        node_dict = {"id": "4", "type": "Output"}
        
        result = Node.from_dict(node_dict)
        
        assert result.id == "4"
        assert result.type == "Output"
        assert result.data == {"default": "value"}


class TestEdge:
    """Test Edge class"""
    
    def test_edge_creation(self):
        """Test Edge creation"""
        from nodes import Edge
        
        edge = Edge("1-2", "1", "2")
        
        assert edge.id == "1-2"
        assert edge.source == "1"
        assert edge.target == "2"
    
    def test_edge_to_dict(self):
        """Test Edge to_dict method"""
        from nodes import Edge
        
        edge = Edge("2-3", "2", "3")
        
        result = edge.to_dict()
        
        expected = {
            "id": "2-3",
            "source": "2",
            "target": "3"
        }
        assert result == expected
    
    def test_edge_from_dict(self):
        """Test Edge from_dict method"""
        from nodes import Edge
        
        edge_dict = {
            "id": "3-4",
            "source": "3",
            "target": "4"
        }
        
        result = Edge.from_dict(edge_dict)
        
        assert result.id == "3-4"
        assert result.source == "3"
        assert result.target == "4"
    
    def test_edge_from_dict_missing_key(self):
        """Test Edge from_dict with missing key"""
        from nodes import Edge
        
        edge_dict = {"id": "5-6", "source": "5"}  # Missing target
        
        with pytest.raises(KeyError, match="Missing key 'target'"):
            Edge.from_dict(edge_dict)


class TestUtilityFunctions:
    """Test utility functions"""
    
    @patch('nodes.get_available_models')
    def test_get_all_models(self, mock_available):
        """Test get_all_models function"""
        from nodes import get_all_models, OPENAI_MODELS, GROQ_MODELS
        
        mock_available.return_value = ["llama3", "mistral"]
        
        result = get_all_models()
        
        expected = ["llama3", "mistral"] + OPENAI_MODELS + GROQ_MODELS
        assert result == expected
    
    def test_call_api_and_decode_response_success(self):
        """Test successful API call and response decoding"""
        from nodes import call_api_and_decode_response
        
        # Mock API function
        def mock_api(*args, **kwargs):
            return '{"nodes": [{"id": "1", "type": "Input"}], "edges": [{"id": "1-2", "source": "1", "target": "2"}]}'
        
        result = call_api_and_decode_response(mock_api, "test_arg")
        
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) == 1
        assert len(result["edges"]) == 1
    
    def test_call_api_and_decode_response_empty_response(self):
        """Test API call with empty response"""
        from nodes import call_api_and_decode_response
        
        def mock_api(*args, **kwargs):
            return ""
        
        with pytest.raises(ValueError, match="Received empty or invalid response"):
            call_api_and_decode_response(mock_api)
    
    def test_call_api_and_decode_response_invalid_json(self):
        """Test API call with invalid JSON"""
        from nodes import call_api_and_decode_response
        
        def mock_api(*args, **kwargs):
            return "invalid json"
        
        with pytest.raises(ValueError, match="Invalid JSON response"):
            call_api_and_decode_response(mock_api)
    
    def test_call_api_and_decode_response_missing_structure(self):
        """Test API call with missing nodes/edges"""
        from nodes import call_api_and_decode_response
        
        def mock_api(*args, **kwargs):
            return '{"nodes": []}'  # Missing edges
        
        with pytest.raises(ValueError, match="Missing 'nodes' or 'edges'"):
            call_api_and_decode_response(mock_api)
    
    def test_call_api_and_decode_response_invalid_edge(self):
        """Test API call with invalid edge structure"""
        from nodes import call_api_and_decode_response
        
        def mock_api(*args, **kwargs):
            return '{"nodes": [], "edges": [{"id": "1-2", "source": "1"}]}'  # Missing target
        
        with pytest.raises(ValueError, match="Missing key 'source' or 'target'"):
            call_api_and_decode_response(mock_api)


class TestWorkflowGeneration:
    """Test workflow generation functionality"""
    
    @patch('nodes.load_api_keys')
    @patch('nodes.call_openai_api')
    @patch('nodes.parse_openai_response')
    def test_generate_workflow_openai(self, mock_parse, mock_openai, mock_keys):
        """Test workflow generation with OpenAI"""
        from nodes import generate_workflow, OPENAI_MODELS
        
        # Setup mocks
        mock_keys.return_value = {"openai_api_key": "test_key"}
        mock_openai.return_value = '{"nodes": [], "edges": []}'
        mock_parse.return_value = {
            "nodes": [
                {"id": "1", "type": "Input", "data": {"content": "Input"}}
            ],
            "edges": [
                {"id": "1-2", "source": "1", "target": "2"}
            ]
        }
        
        with patch('nodes.OPENAI_MODELS', ["gpt-4"]):
            with patch('nodes.Node.from_dict') as mock_node:
                with patch('nodes.Edge.from_dict') as mock_edge:
                    mock_node.return_value = Mock()
                    mock_edge.return_value = Mock()
                    
                    nodes, edges = generate_workflow("Create a simple workflow", "gpt-4")
        
        # Verify API was called
        mock_openai.assert_called_once()
        mock_parse.assert_called_once()
        
        # Verify nodes and edges were created
        assert len(nodes) == 1
        assert len(edges) == 1
    
    @patch('nodes.load_api_keys')
    @patch('nodes.call_groq_api')
    @patch('nodes.parse_openai_response')
    def test_generate_workflow_groq(self, mock_parse, mock_groq, mock_keys):
        """Test workflow generation with Groq"""
        from nodes import generate_workflow, GROQ_MODELS
        
        # Setup mocks
        mock_keys.return_value = {"groq_api_key": "test_key"}
        mock_groq.return_value = '{"nodes": [], "edges": []}'
        mock_parse.return_value = {
            "nodes": [{"id": "1", "type": "LLM", "data": {"content": "LLM"}}],
            "edges": []
        }
        
        with patch('nodes.GROQ_MODELS', ["mixtral-8x7b"]):
            with patch('nodes.Node.from_dict') as mock_node:
                with patch('nodes.Edge.from_dict') as mock_edge:
                    mock_node.return_value = Mock()
                    
                    nodes, edges = generate_workflow("Generate text", "mixtral-8x7b")
        
        # Verify API was called
        mock_groq.assert_called_once()
        mock_parse.assert_called_once()
    
    @patch('nodes.load_api_keys')
    @patch('nodes.call_ollama_endpoint')
    @patch('nodes.parse_ollama_response')
    def test_generate_workflow_ollama(self, mock_parse, mock_ollama, mock_keys):
        """Test workflow generation with Ollama"""
        from nodes import generate_workflow
        
        # Setup mocks
        mock_keys.return_value = {}
        mock_ollama.return_value = ('{"nodes": [], "edges": []}', None, None, None)
        mock_parse.return_value = {
            "nodes": [{"id": "1", "type": "Processing", "data": {"content": "Process"}}],
            "edges": []
        }
        
        with patch('nodes.OPENAI_MODELS', []):
            with patch('nodes.GROQ_MODELS', []):
                with patch('nodes.Node.from_dict') as mock_node:
                    mock_node.return_value = Mock()
                    
                    nodes, edges = generate_workflow("Process data", "llama3")
        
        # Verify API was called
        mock_ollama.assert_called_once()
        mock_parse.assert_called_once()
    
    @patch('nodes.st')
    @patch('nodes.load_api_keys')
    @patch('nodes.call_ollama_endpoint')
    def test_generate_workflow_exception(self, mock_ollama, mock_keys, mock_st):
        """Test workflow generation with exception"""
        from nodes import generate_workflow
        
        # Setup mocks
        mock_keys.return_value = {}
        mock_ollama.side_effect = Exception("API Error")
        mock_st.error = Mock()
        
        with patch('nodes.OPENAI_MODELS', []):
            with patch('nodes.GROQ_MODELS', []):
                nodes, edges = generate_workflow("Test request", "llama3")
        
        # Should return empty lists and show error
        assert nodes == []
        assert edges == []
        mock_st.error.assert_called_once()


class TestResponseParsing:
    """Test response parsing functions"""
    
    def test_parse_openai_response_success(self):
        """Test successful OpenAI response parsing"""
        from nodes import parse_openai_response
        
        response = 'Some text before {"nodes": [{"id": "1"}], "edges": [{"id": "1-2", "source": "1", "target": "2"}]} some text after'
        
        result = parse_openai_response(response)
        
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) == 1
        assert len(result["edges"]) == 1
    
    def test_parse_openai_response_no_json(self):
        """Test OpenAI response with no JSON"""
        from nodes import parse_openai_response
        
        response = "No JSON here at all"
        
        with pytest.raises(ValueError, match="No valid JSON object found"):
            parse_openai_response(response)
    
    def test_parse_openai_response_invalid_structure(self):
        """Test OpenAI response with invalid structure"""
        from nodes import parse_openai_response
        
        response = '{"nodes": []}'  # Missing edges
        
        with pytest.raises(ValueError, match="Missing 'nodes' or 'edges'"):
            parse_openai_response(response)
    
    def test_parse_ollama_response_success(self):
        """Test successful Ollama response parsing"""
        from nodes import parse_ollama_response
        
        response = '''Some conversation
        
        Here is the final result: {"nodes": [{"id": "1"}], "edges": [{"id": "1-2", "source": "1", "target": "2"}]}'''
        
        result = parse_ollama_response(response)
        
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) == 1
        assert len(result["edges"]) == 1
    
    def test_parse_ollama_response_missing_edge_fields(self):
        """Test Ollama response with missing edge fields"""
        from nodes import parse_ollama_response
        
        response = '{"nodes": [], "edges": [{"id": "1-2"}]}'  # Missing source/target
        
        result = parse_ollama_response(response)
        
        # Should auto-populate missing fields
        assert result["edges"][0]["source"] == "1"
        assert result["edges"][0]["target"] == "2"
    
    def test_parse_ollama_response_no_json(self):
        """Test Ollama response with no JSON"""
        from nodes import parse_ollama_response
        
        response = "No JSON in this response"
        
        with pytest.raises(ValueError, match="No valid JSON object found"):
            parse_ollama_response(response)


class TestWorkflowExecution:
    """Test workflow execution functionality"""
    
    def test_execute_workflow_basic(self):
        """Test basic workflow execution"""
        from nodes import execute_workflow, Node, Edge
        
        # Create simple workflow: Input -> Output
        nodes = [
            Node("1", "Input", {"content": "Input", "input_type": "Text", "input_text": "Test input"}),
            Node("2", "Output", {"content": "Output", "output_type": "Text", "output_label": "Result"})
        ]
        edges = [Edge("1-2", "1", "2")]
        
        with patch('nodes.load_api_keys', return_value={}):
            results = execute_workflow(nodes, edges)
        
        assert "1" in results
        assert "2" in results
        assert results["1"] == "Test input"
        assert "Result: Test input" in results["2"]
    
    def test_execute_workflow_node_not_found(self):
        """Test workflow execution with missing node"""
        from nodes import execute_workflow, Node, Edge
        
        nodes = [Node("1", "Input", {"input_type": "Text", "input_text": "Test"})]
        edges = [Edge("1-2", "1", "2")]  # Edge points to non-existent node
        
        with patch('nodes.load_api_keys', return_value={}):
            results = execute_workflow(nodes, edges)
        
        assert "Error: Node 2 not found" in results["2"]
    
    def test_execute_workflow_with_processing_cycle(self):
        """Test workflow execution processes all nodes"""
        from nodes import execute_workflow, Node, Edge
        
        nodes = [
            Node("1", "Input", {"input_type": "Text", "input_text": "Hello"}),
            Node("2", "Processing", {"processing_type": "Preprocessing", "preprocessing_steps": ["Lowercasing"]}),
            Node("3", "Output", {"output_type": "Text", "output_label": "Final"})
        ]
        edges = [
            Edge("1-2", "1", "2"),
            Edge("2-3", "2", "3")
        ]
        
        with patch('nodes.load_api_keys', return_value={}):
            results = execute_workflow(nodes, edges)
        
        assert results["1"] == "Hello"
        assert results["2"] == "hello"  # Lowercased
        assert "Final: hello" in results["3"]


class TestNodeHandlers:
    """Test individual node handler functions"""
    
    def test_handle_input_node_text(self):
        """Test input node with text type"""
        from nodes import handle_input_node, Node
        
        node = Node("1", "Input", {
            "input_type": "Text",
            "input_text": "Test input text"
        })
        
        result = handle_input_node(node)
        assert result == "Test input text"
    
    def test_handle_input_node_file(self):
        """Test input node with file type"""
        from nodes import handle_input_node, Node
        
        mock_file = Mock()
        mock_file.getvalue.return_value = b"File content"
        
        node = Node("1", "Input", {
            "input_type": "File",
            "file_upload": mock_file
        })
        
        result = handle_input_node(node)
        assert result == "File content"
    
    def test_handle_input_node_file_no_upload(self):
        """Test input node with file type but no file"""
        from nodes import handle_input_node, Node
        
        node = Node("1", "Input", {
            "input_type": "File",
            "file_upload": None
        })
        
        result = handle_input_node(node)
        assert "Error: No file uploaded" in result
    
    @patch('nodes.requests.get')
    def test_handle_input_node_api_success(self, mock_get):
        """Test input node with API type"""
        from nodes import handle_input_node, Node
        
        mock_response = Mock()
        mock_response.text = "API response data"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        node = Node("1", "Input", {
            "input_type": "API",
            "api_endpoint": "https://api.example.com/data"
        })
        
        result = handle_input_node(node)
        assert result == "API response data"
        mock_get.assert_called_once_with("https://api.example.com/data")
    
    @patch('nodes.requests.get')
    def test_handle_input_node_api_error(self, mock_get):
        """Test input node with API error"""
        from nodes import handle_input_node, Node
        
        mock_get.side_effect = Exception("Connection error")
        
        node = Node("1", "Input", {
            "input_type": "API",
            "api_endpoint": "https://api.example.com/data"
        })
        
        result = handle_input_node(node)
        assert "Error fetching API" in result
    
    def test_handle_processing_node_preprocessing(self):
        """Test processing node with preprocessing"""
        from nodes import handle_processing_node, Node, Edge
        
        node = Node("2", "Processing", {
            "processing_type": "Preprocessing",
            "preprocessing_steps": ["Lowercasing", "Remove Punctuation"]
        })
        
        def mock_process_node(node_id):
            return "Hello, World!"
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        result = handle_processing_node(node, incoming_edges, mock_process_node)
        assert result == "hello world"  # Lowercased and punctuation removed
    
    def test_handle_processing_node_no_input(self):
        """Test processing node without input"""
        from nodes import handle_processing_node, Node
        
        node = Node("2", "Processing", {"processing_type": "Preprocessing"})
        
        result = handle_processing_node(node, [], lambda x: "")
        assert "Error: Processing node requires input" in result
    
    @patch('nodes.load_api_keys')
    @patch('nodes.call_ollama_endpoint')
    @patch('nodes.construct_prompt')
    def test_handle_llm_node_ollama(self, mock_construct, mock_ollama, mock_keys):
        """Test LLM node with Ollama"""
        from nodes import handle_llm_node, Node, Edge
        
        # Setup mocks
        mock_keys.return_value = {}
        mock_ollama.return_value = ("LLM response", None, None, None)
        mock_construct.return_value = "Complete prompt"
        
        node = Node("2", "LLM", {
            "model_name": "llama3",
            "prompt": "Analyze this text:",
            "temperature": 0.7,
            "max_tokens": 1000,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "conversation_history": []
        })
        
        def mock_process_node(node_id):
            return "Input text"
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        with patch('nodes.OPENAI_MODELS', []):
            with patch('nodes.GROQ_MODELS', []):
                result = handle_llm_node(node, incoming_edges, mock_process_node, {})
        
        assert result == "LLM response"
        mock_ollama.assert_called_once()
        
        # Verify conversation history was updated
        assert len(node.data["conversation_history"]) == 2
        assert node.data["conversation_history"][0]["role"] == "user"
        assert node.data["conversation_history"][1]["role"] == "assistant"
    
    @patch('nodes.load_api_keys')
    @patch('nodes.call_openai_api')
    @patch('nodes.construct_prompt')
    def test_handle_llm_node_openai(self, mock_construct, mock_openai, mock_keys):
        """Test LLM node with OpenAI"""
        from nodes import handle_llm_node, Node, Edge
        
        # Setup mocks
        mock_keys.return_value = {"openai_api_key": "test_key"}
        mock_openai.return_value = "OpenAI response"
        mock_construct.return_value = "Complete prompt"
        
        node = Node("2", "LLM", {
            "model_name": "gpt-4",
            "prompt": "Generate text:",
            "temperature": 0.8,
            "max_tokens": 2000,
            "conversation_history": []
        })
        
        def mock_process_node(node_id):
            return "Input for GPT"
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        with patch('nodes.OPENAI_MODELS', ["gpt-4"]):
            result = handle_llm_node(node, incoming_edges, mock_process_node, {"openai_api_key": "test_key"})
        
        assert result == "OpenAI response"
        mock_openai.assert_called_once()
    
    def test_handle_output_node_text(self):
        """Test output node with text type"""
        from nodes import handle_output_node, Node, Edge
        
        node = Node("3", "Output", {
            "output_type": "Text",
            "output_label": "Final Result"
        })
        
        def mock_process_node(node_id):
            return "Processed data"
        
        incoming_edges = [Edge("2-3", "2", "3")]
        
        result = handle_output_node(node, incoming_edges, mock_process_node)
        assert result == "Final Result: Processed data"
    
    def test_handle_output_node_file(self):
        """Test output node with file type"""
        from nodes import handle_output_node, Node, Edge
        
        node = Node("3", "Output", {"output_type": "File"})
        
        def mock_process_node(node_id):
            return "File content data"
        
        incoming_edges = [Edge("2-3", "2", "3")]
        
        result = handle_output_node(node, incoming_edges, mock_process_node)
        assert result == "File content data"
    
    @patch('nodes.plt')
    @patch('nodes.base64.b64encode')
    def test_handle_output_node_visualization(self, mock_b64, mock_plt):
        """Test output node with visualization type"""
        from nodes import handle_output_node, Node, Edge
        
        # Setup mocks
        mock_b64.return_value.decode.return_value = "base64_image_data"
        
        node = Node("3", "Output", {"output_type": "Visualization"})
        
        def mock_process_node(node_id):
            return "Data for visualization"
        
        incoming_edges = [Edge("2-3", "2", "3")]
        
        result = handle_output_node(node, incoming_edges, mock_process_node)
        
        assert result == "base64_image_data"
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called_once()


class TestDataRetrievalNode:
    """Test data retrieval node functionality"""
    
    def test_handle_data_retrieval_node_database(self):
        """Test data retrieval node with database type"""
        from nodes import handle_data_retrieval_node, Node, Edge
        
        node = Node("2", "DataRetrieval", {
            "retrieval_type": "Database",
            "database_type": "SQL",
            "query": "SELECT * FROM users WHERE name = '{input}'"
        })
        
        def mock_process_node(node_id):
            return "John"
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        result = handle_data_retrieval_node(node, incoming_edges, mock_process_node)
        
        assert "SQL query result" in result
        assert "John" in result
    
    @patch('nodes.requests.get')
    def test_handle_data_retrieval_node_api(self, mock_get):
        """Test data retrieval node with API type"""
        from nodes import handle_data_retrieval_node, Node, Edge
        
        # Setup mock
        mock_response = Mock()
        mock_response.text = "API data response"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        node = Node("2", "DataRetrieval", {
            "retrieval_type": "API",
            "api_endpoint": "https://api.example.com/search",
            "api_method": "GET",
            "api_params": {"q": "{input}"}
        })
        
        def mock_process_node(node_id):
            return "search_term"
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        result = handle_data_retrieval_node(node, incoming_edges, mock_process_node)
        
        assert result == "API data response"
        mock_get.assert_called_once_with(
            "https://api.example.com/search",
            params={"q": "search_term"},
            timeout=10
        )
    
    @patch('nodes.requests.get')
    def test_handle_data_retrieval_node_api_error(self, mock_get):
        """Test data retrieval node with API error"""
        from nodes import handle_data_retrieval_node, Node, Edge
        
        mock_get.side_effect = Exception("Network error")
        
        node = Node("2", "DataRetrieval", {
            "retrieval_type": "API",
            "api_endpoint": "https://api.example.com/data",
            "api_method": "GET",
            "api_params": {}
        })
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        result = handle_data_retrieval_node(node, incoming_edges, lambda x: "input")
        
        assert "API request error" in result
    
    @patch('builtins.open', mock_open(read_data="File content here"))
    def test_handle_data_retrieval_node_file(self):
        """Test data retrieval node with file type"""
        from nodes import handle_data_retrieval_node, Node, Edge
        
        node = Node("2", "DataRetrieval", {
            "retrieval_type": "File",
            "file_path": "/path/to/file.txt"
        })
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        result = handle_data_retrieval_node(node, incoming_edges, lambda x: "input")
        
        assert result == "File content here"
    
    def test_handle_data_retrieval_node_file_error(self):
        """Test data retrieval node with file error"""
        from nodes import handle_data_retrieval_node, Node, Edge
        
        node = Node("2", "DataRetrieval", {
            "retrieval_type": "File",
            "file_path": "/nonexistent/file.txt"
        })
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
            result = handle_data_retrieval_node(node, incoming_edges, lambda x: "input")
        
        assert "File read error" in result


class TestControlNode:
    """Test control node functionality"""
    
    def test_handle_control_node_conditional_true(self):
        """Test control node with conditional (true case)"""
        from nodes import handle_control_node, Node, Edge
        
        node = Node("2", "Control", {
            "control_type": "Conditional",
            "condition": "len('{input}') > 5",
            "true_branch": "Input is long",
            "false_branch": "Input is short"
        })
        
        def mock_process_node(node_id):
            return "This is a long input"
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        result = handle_control_node(node, incoming_edges, mock_process_node)
        
        assert "TRUE: Input is long" in result
    
    def test_handle_control_node_conditional_false(self):
        """Test control node with conditional (false case)"""
        from nodes import handle_control_node, Node, Edge
        
        node = Node("2", "Control", {
            "control_type": "Conditional",
            "condition": "len('{input}') > 10",
            "true_branch": "Input is long",
            "false_branch": "Input is short"
        })
        
        def mock_process_node(node_id):
            return "Short"
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        result = handle_control_node(node, incoming_edges, mock_process_node)
        
        assert "FALSE: Input is short" in result
    
    def test_handle_control_node_loop(self):
        """Test control node with loop type"""
        from nodes import handle_control_node, Node, Edge
        
        node = Node("2", "Control", {
            "control_type": "Loop",
            "iterations": 3,
            "loop_body": "Processing {input} iteration {i}"
        })
        
        def mock_process_node(node_id):
            return "test_data"
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        result = handle_control_node(node, incoming_edges, mock_process_node)
        
        assert "Iteration 1: Processing test_data iteration 0" in result
        assert "Iteration 2: Processing test_data iteration 1" in result
        assert "Iteration 3: Processing test_data iteration 2" in result
    
    def test_handle_control_node_switch_match(self):
        """Test control node with switch (matching case)"""
        from nodes import handle_control_node, Node, Edge
        
        node = Node("2", "Control", {
            "control_type": "Switch",
            "cases": {
                "option1": "Selected option 1",
                "option2": "Selected option 2"
            },
            "default_case": "Unknown option"
        })
        
        def mock_process_node(node_id):
            return "option1"
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        result = handle_control_node(node, incoming_edges, mock_process_node)
        
        assert "CASE 'option1': Selected option 1" in result
    
    def test_handle_control_node_switch_default(self):
        """Test control node with switch (default case)"""
        from nodes import handle_control_node, Node, Edge
        
        node = Node("2", "Control", {
            "control_type": "Switch",
            "cases": {"option1": "Result 1"},
            "default_case": "Default result"
        })
        
        def mock_process_node(node_id):
            return "unknown_option"
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        result = handle_control_node(node, incoming_edges, mock_process_node)
        
        assert "DEFAULT: Default result" in result


class TestIntegrationNode:
    """Test integration node functionality"""
    
    def test_handle_integration_node_email(self):
        """Test integration node with email type"""
        from nodes import handle_integration_node, Node, Edge
        
        node = Node("2", "Integration", {
            "integration_type": "Email",
            "email_recipient": "test@example.com",
            "email_subject": "Test Subject",
            "email_body": "Email content: {input}"
        })
        
        def mock_process_node(node_id):
            return "test data"
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        result = handle_integration_node(node, incoming_edges, mock_process_node)
        
        assert "Email would be sent to test@example.com" in result
        assert "Test Subject" in result
    
    @patch('nodes.requests.post')
    def test_handle_integration_node_webhook_success(self, mock_post):
        """Test integration node with webhook type"""
        from nodes import handle_integration_node, Node, Edge
        
        # Setup mock
        mock_response = Mock()
        mock_response.text = "Webhook success"
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        node = Node("2", "Integration", {
            "integration_type": "Webhook",
            "webhook_url": "https://webhook.example.com",
            "webhook_method": "POST",
            "webhook_data": {"message": "{input}"}
        })
        
        def mock_process_node(node_id):
            return "webhook test"
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        result = handle_integration_node(node, incoming_edges, mock_process_node)
        
        assert "Webhook call successful" in result
        mock_post.assert_called_once_with(
            "https://webhook.example.com",
            json={"message": "webhook test"},
            timeout=10
        )
    
    @patch('nodes.requests.post')
    def test_handle_integration_node_webhook_error(self, mock_post):
        """Test integration node with webhook error"""
        from nodes import handle_integration_node, Node, Edge
        
        mock_post.side_effect = Exception("Connection failed")
        
        node = Node("2", "Integration", {
            "integration_type": "Webhook",
            "webhook_url": "https://webhook.example.com",
            "webhook_method": "POST",
            "webhook_data": {}
        })
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        result = handle_integration_node(node, incoming_edges, lambda x: "input")
        
        assert "Webhook call error" in result
    
    def test_handle_integration_node_database(self):
        """Test integration node with database type"""
        from nodes import handle_integration_node, Node, Edge
        
        node = Node("2", "Integration", {
            "integration_type": "Database",
            "database_action": "insert",
            "database_data": "INSERT INTO table VALUES ('{input}')"
        })
        
        def mock_process_node(node_id):
            return "new_value"
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        result = handle_integration_node(node, incoming_edges, mock_process_node)
        
        assert "Database insert with data" in result
        assert "new_value" in result


class TestUtilityNode:
    """Test utility node functionality"""
    
    def test_handle_utility_node_format_json(self):
        """Test utility node with JSON formatting"""
        from nodes import handle_utility_node, Node, Edge
        
        node = Node("2", "Utility", {
            "utility_type": "Format",
            "format_type": "JSON"
        })
        
        def mock_process_node(node_id):
            return '{"key": "value"}'
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        result = handle_utility_node(node, incoming_edges, mock_process_node)
        
        # Should parse and re-format JSON
        assert '"key": "value"' in result
    
    def test_handle_utility_node_format_html(self):
        """Test utility node with HTML formatting"""
        from nodes import handle_utility_node, Node, Edge
        
        node = Node("2", "Utility", {
            "utility_type": "Format",
            "format_type": "HTML"
        })
        
        def mock_process_node(node_id):
            return "Plain text"
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        result = handle_utility_node(node, incoming_edges, mock_process_node)
        
        assert "<html><body><p>Plain text</p></body></html>" == result
    
    def test_handle_utility_node_transform_uppercase(self):
        """Test utility node with uppercase transform"""
        from nodes import handle_utility_node, Node, Edge
        
        node = Node("2", "Utility", {
            "utility_type": "Transform",
            "transform_type": "Uppercase"
        })
        
        def mock_process_node(node_id):
            return "lowercase text"
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        result = handle_utility_node(node, incoming_edges, mock_process_node)
        
        assert result == "LOWERCASE TEXT"
    
    def test_handle_utility_node_calculate_basic(self):
        """Test utility node with basic calculation"""
        from nodes import handle_utility_node, Node, Edge
        
        node = Node("2", "Utility", {
            "utility_type": "Calculate",
            "calculation_type": "Basic Math",
            "formula": "{input} * 2"
        })
        
        def mock_process_node(node_id):
            return "5"
        
        incoming_edges = [Edge("1-2", "1", "2")]
        
        result = handle_utility_node(node, incoming_edges, mock_process_node)
        
        assert "Calculation result: 10" in result


class TestWorkflowValidation:
    """Test workflow validation functionality"""
    
    def test_validate_workflow_valid(self):
        """Test validation of valid workflow"""
        from nodes import validate_workflow, Node, Edge
        
        nodes = [
            Node("1", "Input", {}),
            Node("2", "LLM", {}),
            Node("3", "Output", {})
        ]
        edges = [
            Edge("1-2", "1", "2"),
            Edge("2-3", "2", "3")
        ]
        
        result = validate_workflow(nodes, edges)
        
        assert result["valid"] is True
        assert result["error"] is None
    
    def test_validate_workflow_no_input(self):
        """Test validation without input node"""
        from nodes import validate_workflow, Node, Edge
        
        nodes = [
            Node("1", "LLM", {}),
            Node("2", "Output", {})
        ]
        edges = [Edge("1-2", "1", "2")]
        
        result = validate_workflow(nodes, edges)
        
        assert result["valid"] is False
        assert "must have at least one Input node" in result["error"]
    
    def test_validate_workflow_no_output(self):
        """Test validation without output node"""
        from nodes import validate_workflow, Node, Edge
        
        nodes = [
            Node("1", "Input", {}),
            Node("2", "LLM", {})
        ]
        edges = [Edge("1-2", "1", "2")]
        
        result = validate_workflow(nodes, edges)
        
        assert result["valid"] is False
        assert "must have at least one Output node" in result["error"]
    
    def test_validate_workflow_invalid_edge(self):
        """Test validation with invalid edge"""
        from nodes import validate_workflow, Node, Edge
        
        nodes = [
            Node("1", "Input", {}),
            Node("2", "Output", {})
        ]
        edges = [Edge("1-3", "1", "3")]  # Node 3 doesn't exist
        
        result = validate_workflow(nodes, edges)
        
        assert result["valid"] is False
        assert "connects to non-existent nodes" in result["error"]
    
    def test_validate_workflow_disconnected_nodes(self):
        """Test validation with disconnected nodes"""
        from nodes import validate_workflow, Node, Edge
        
        nodes = [
            Node("1", "Input", {}),
            Node("2", "LLM", {}),
            Node("3", "Output", {})
        ]
        edges = [Edge("1-2", "1", "2")]  # Node 3 is disconnected
        
        result = validate_workflow(nodes, edges)
        
        assert result["valid"] is False
        assert "All nodes must be connected" in result["error"]
    
    def test_has_cycle_true(self):
        """Test cycle detection with cycle"""
        from nodes import has_cycle, Node, Edge
        
        nodes = [
            Node("1", "Input", {}),
            Node("2", "LLM", {}),
            Node("3", "Processing", {})
        ]
        edges = [
            Edge("1-2", "1", "2"),
            Edge("2-3", "2", "3"),
            Edge("3-2", "3", "2")  # Creates cycle
        ]
        
        result = has_cycle(nodes, edges)
        assert result is True
    
    def test_has_cycle_false(self):
        """Test cycle detection without cycle"""
        from nodes import has_cycle, Node, Edge
        
        nodes = [
            Node("1", "Input", {}),
            Node("2", "LLM", {}),
            Node("3", "Output", {})
        ]
        edges = [
            Edge("1-2", "1", "2"),
            Edge("2-3", "2", "3")
        ]
        
        result = has_cycle(nodes, edges)
        assert result is False
    
    def test_path_exists_input_to_output_true(self):
        """Test path existence with valid path"""
        from nodes import path_exists_input_to_output, Node, Edge
        
        nodes = [
            Node("1", "Input", {}),
            Node("2", "LLM", {}),
            Node("3", "Output", {})
        ]
        edges = [
            Edge("1-2", "1", "2"),
            Edge("2-3", "2", "3")
        ]
        
        result = path_exists_input_to_output(nodes, edges)
        assert result is True
    
    def test_path_exists_input_to_output_false(self):
        """Test path existence without valid path"""
        from nodes import path_exists_input_to_output, Node, Edge
        
        nodes = [
            Node("1", "Input", {}),
            Node("2", "LLM", {}),
            Node("3", "Output", {})
        ]
        edges = [
            Edge("2-3", "2", "3")  # No path from input to output
        ]
        
        result = path_exists_input_to_output(nodes, edges)
        assert result is False


class TestPromptConstruction:
    """Test prompt construction functionality"""
    
    @patch('nodes.get_agent_prompt')
    @patch('nodes.get_metacognitive_prompt')
    @patch('nodes.get_voice_prompt')
    @patch('nodes.get_identity_prompt')
    def test_construct_prompt_full(self, mock_identity, mock_voice, mock_metacog, mock_agent):
        """Test prompt construction with all components"""
        from nodes import construct_prompt, Node
        
        # Setup mocks
        mock_agent.return_value = {"Analyst": "You are an analyst"}
        mock_metacog.return_value = {"Logical": "Think logically"}
        mock_voice.return_value = {"Formal": "Speak formally"}
        mock_identity.return_value = {"Professional": "Be professional"}
        
        node = Node("1", "LLM", {
            "agent_type": "Analyst",
            "metacognitive_type": "Logical",
            "voice_type": "Formal",
            "identity_type": "Professional",
            "conversation_history": [
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"}
            ]
        })
        
        result = construct_prompt(node, "Base prompt", "User input text")
        
        assert "You are an analyst" in result
        assert "Think logically" in result
        assert "Speak formally" in result
        assert "Be professional" in result
        assert "Base prompt" in result
        assert "User Input: User input text" in result
        assert "Recent Conversation:" in result
        assert "Previous question" in result
    
    @patch('nodes.get_agent_prompt')
    def test_construct_prompt_minimal(self, mock_agent):
        """Test prompt construction with minimal components"""
        from nodes import construct_prompt, Node
        
        mock_agent.return_value = {}
        
        node = Node("1", "LLM", {
            "agent_type": "None",
            "metacognitive_type": "None",
            "voice_type": "None",
            "identity_type": "None",
            "conversation_history": []
        })
        
        result = construct_prompt(node, "Simple prompt", "Input")
        
        assert "Simple prompt" in result
        assert "User Input: Input" in result
        assert "Recent Conversation:" not in result


class TestFileOperations:
    """Test file operations (save/load workflows)"""
    
    @patch('nodes.time.time')
    def test_save_workflow(self, mock_time):
        """Test saving workflow"""
        from nodes import save_workflow, Node, Edge
        
        # Setup mock
        mock_time.return_value = 1234567890
        
        nodes = [Node("1", "Input", {"content": "Test"})]
        edges = [Edge("1-2", "1", "2")]
        
        mock_file = Mock()
        with patch('builtins.open', mock_file):
            with patch('nodes.json.dump') as mock_dump:
                result = save_workflow(nodes, edges)
        
        # Should save to both files
        assert mock_file.call_count == 2
        assert mock_dump.call_count == 2
        assert result == "workflow_1234567890.json"
    
    def test_load_workflow_success(self):
        """Test successful workflow loading"""
        from nodes import load_workflow
        
        workflow_data = {
            "nodes": [{"id": "1", "type": "Input", "data": {"content": "Test"}}],
            "edges": [{"id": "1-2", "source": "1", "target": "2"}]
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(workflow_data))):
            with patch('nodes.Node.from_dict') as mock_node:
                with patch('nodes.Edge.from_dict') as mock_edge:
                    mock_node.return_value = Mock()
                    mock_edge.return_value = Mock()
                    
                    nodes, edges = load_workflow()
        
        assert len(nodes) == 1
        assert len(edges) == 1
        mock_node.assert_called_once()
        mock_edge.assert_called_once()
    
    @patch('nodes.st')
    def test_load_workflow_file_not_found(self, mock_st):
        """Test workflow loading with missing file"""
        from nodes import load_workflow
        
        mock_st.warning = Mock()
        
        with patch('builtins.open', side_effect=FileNotFoundError):
            nodes, edges = load_workflow()
        
        assert nodes == []
        assert edges == []
        mock_st.warning.assert_called_with("No saved workflow found.")
    
    @patch('nodes.st')
    def test_load_workflow_json_error(self, mock_st):
        """Test workflow loading with JSON error"""
        from nodes import load_workflow
        
        mock_st.error = Mock()
        
        with patch('builtins.open', mock_open(read_data="invalid json")):
            nodes, edges = load_workflow()
        
        assert nodes == []
        assert edges == []
        mock_st.error.assert_called_with("Error decoding the saved workflow. The file may be corrupted.")


class TestNodeCreation:
    """Test node creation functionality"""
    
    @patch('nodes.get_all_models')
    def test_create_node_input(self, mock_models):
        """Test creating input node"""
        from nodes import create_node
        
        mock_models.return_value = ["llama3", "gpt-4"]
        
        node = create_node("1", "Input")
        
        assert node.id == "1"
        assert node.type == "Input"
        assert node.data["content"] == "Input Node"
        assert node.data["input_type"] == "Text"
        assert node.data["input_text"] == ""
    
    @patch('nodes.get_all_models')
    def test_create_node_llm(self, mock_models):
        """Test creating LLM node"""
        from nodes import create_node
        
        mock_models.return_value = ["llama3", "gpt-4"]
        
        node = create_node("2", "LLM")
        
        assert node.id == "2"
        assert node.type == "LLM"
        assert node.data["content"] == "LLM Node"
        assert node.data["model_name"] == "llama3"  # First available model
        assert node.data["temperature"] == 0.7
        assert node.data["conversation_history"] == []
    
    def test_create_node_processing(self):
        """Test creating processing node"""
        from nodes import create_node
        
        node = create_node("3", "Processing")
        
        assert node.id == "3"
        assert node.type == "Processing"
        assert node.data["processing_type"] == "Preprocessing"
        assert node.data["preprocessing_steps"] == []
    
    def test_create_node_output(self):
        """Test creating output node"""
        from nodes import create_node
        
        node = create_node("4", "Output")
        
        assert node.id == "4"
        assert node.type == "Output"
        assert node.data["output_type"] == "Text"
        assert node.data["output_label"] == "Output:"
    
    def test_create_node_data_retrieval(self):
        """Test creating data retrieval node"""
        from nodes import create_node
        
        node = create_node("5", "DataRetrieval")
        
        assert node.id == "5"
        assert node.type == "DataRetrieval"
        assert node.data["retrieval_type"] == "API"
        assert "api_endpoint" in node.data
    
    def test_create_node_control(self):
        """Test creating control node"""
        from nodes import create_node
        
        node = create_node("6", "Control")
        
        assert node.id == "6"
        assert node.type == "Control"
        assert node.data["control_type"] == "Conditional"
        assert "condition" in node.data
    
    def test_create_node_integration(self):
        """Test creating integration node"""
        from nodes import create_node
        
        node = create_node("7", "Integration")
        
        assert node.id == "7"
        assert node.type == "Integration"
        assert node.data["integration_type"] == "Webhook"
        assert "webhook_url" in node.data
    
    def test_create_node_utility(self):
        """Test creating utility node"""
        from nodes import create_node
        
        node = create_node("8", "Utility")
        
        assert node.id == "8"
        assert node.type == "Utility"
        assert node.data["utility_type"] == "Format"
        assert "format_type" in node.data
    
    def test_create_node_unavailable_type(self):
        """Test creating node with unavailable type"""
        from nodes import create_node
        
        with patch('nodes.AVAILABLE_NODE_TYPES', {"Input": False}):
            with pytest.raises(ValueError, match="Node type 'Input' is not available"):
                create_node("1", "Input")


class TestScriptExport:
    """Test script export functionality"""
    
    def test_export_workflow_as_script(self):
        """Test exporting workflow as Python script"""
        from nodes import export_workflow_as_script, Node, Edge
        
        nodes = [
            Node("1", "Input", {"input_text": "Hello World"}),
            Node("2", "LLM", {}),
            Node("3", "Output", {})
        ]
        edges = [
            Edge("1-2", "1", "2"),
            Edge("2-3", "2", "3")
        ]
        
        script = export_workflow_as_script(nodes, edges)
        
        assert "import requests" in script
        assert "def execute_workflow():" in script
        assert "Node 1: Input" in script
        assert "Node 2: LLM" in script
        assert "Node 3: Output" in script
        assert "if __name__ == '__main__':" in script
        assert "Hello World" in script


class TestConstants:
    """Test module constants"""
    
    def test_available_node_types(self):
        """Test available node types constant"""
        from nodes import AVAILABLE_NODE_TYPES
        
        expected_types = [
            "Input", "Processing", "LLM", "Output", 
            "DataRetrieval", "Control", "Integration", "Utility"
        ]
        
        for node_type in expected_types:
            assert node_type in AVAILABLE_NODE_TYPES
            assert AVAILABLE_NODE_TYPES[node_type] is True
    
    def test_node_colors(self):
        """Test node colors constant"""
        from nodes import NODE_COLORS, AVAILABLE_NODE_TYPES
        
        # All available node types should have colors
        for node_type in AVAILABLE_NODE_TYPES:
            assert node_type in NODE_COLORS
            assert NODE_COLORS[node_type].startswith("#")  # Hex color format
    
    def test_node_emojis(self):
        """Test node emojis constant"""
        from nodes import NODE_EMOJIS, AVAILABLE_NODE_TYPES
        
        # All available node types should have emojis
        for node_type in AVAILABLE_NODE_TYPES:
            assert node_type in NODE_EMOJIS
            assert len(NODE_EMOJIS[node_type]) > 0


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_execute_workflow_exception_handling(self):
        """Test workflow execution with node exception"""
        from nodes import execute_workflow, Node, Edge
        
        # Create a node that will cause an exception
        nodes = [
            Node("1", "Input", {"input_type": "Unknown"}),  # Unknown input type
            Node("2", "Output", {"output_type": "Text", "output_label": "Result"})
        ]
        edges = [Edge("1-2", "1", "2")]
        
        with patch('nodes.load_api_keys', return_value={}):
            results = execute_workflow(nodes, edges)
        
        # Should handle exception gracefully
        assert "1" in results
        assert "Unknown input type" in results["1"]
    
    def test_handle_llm_node_no_input(self):
        """Test LLM node without input"""
        from nodes import handle_llm_node, Node
        
        node = Node("1", "LLM", {})
        
        result = handle_llm_node(node, [], lambda x: "", {})
        
        assert "Error: LLM node requires input" in result
    
    def test_handle_output_node_no_input(self):
        """Test output node without input"""
        from nodes import handle_output_node, Node
        
        node = Node("1", "Output", {})
        
        result = handle_output_node(node, [], lambda x: "")
        
        assert "Error: Output node requires input" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
