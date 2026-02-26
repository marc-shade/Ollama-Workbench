"""
Tests for nodes.py - Node creation, edge management, workflow validation,
and execution flow.

Tests patch all external dependencies (Streamlit, Ollama API, HTTP requests)
so they can run without any running services.
"""

import json
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Shared mock context
# ---------------------------------------------------------------------------

def _make_module_mocks():
    st = MagicMock()
    st.session_state = {}
    plt_mock = MagicMock()
    plt_mock.figure.return_value = MagicMock()
    return {
        "streamlit": st,
        "requests": MagicMock(),
        "matplotlib": MagicMock(),
        "matplotlib.pyplot": plt_mock,
        "pandas": MagicMock(),
        "ollama_workbench.providers.ollama_utils": MagicMock(
            get_available_models=MagicMock(return_value=["llama3"]),
            call_ollama_endpoint=MagicMock(return_value=("response", None, None, None)),
            load_api_keys=MagicMock(return_value={}),
        ),
        "ollama_workbench.providers.openai_utils": MagicMock(
            OPENAI_MODELS=[], call_openai_api=MagicMock(return_value="openai response")
        ),
        "ollama_workbench.providers.groq_utils": MagicMock(
            GROQ_MODELS=[], call_groq_api=MagicMock(return_value="groq response")
        ),
        "ollama_workbench.ui.prompts": MagicMock(
            get_agent_prompt=MagicMock(return_value={}),
            get_metacognitive_prompt=MagicMock(return_value={"Chain of Thought": "cot prompt"}),
            get_voice_prompt=MagicMock(return_value={}),
            get_identity_prompt=MagicMock(return_value={}),
        ),
    }


@pytest.fixture(scope="module")
def nodes_mod():
    with patch.dict("sys.modules", _make_module_mocks()):
        import importlib
        import ollama_workbench.workflows.nodes as n
        importlib.reload(n)
        return n


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class TestNodeClass:
    def test_create_node_with_data(self, nodes_mod):
        data = {"content": "My Node", "input_type": "Text", "input_text": "hello"}
        node = nodes_mod.Node("node-1", "Input", data)
        assert node.id == "node-1"
        assert node.type == "Input"
        assert node.data == data

    def test_to_dict_round_trip(self, nodes_mod):
        data = {"content": "Test", "model": "llama3"}
        node = nodes_mod.Node("n2", "LLM", data)
        d = node.to_dict()
        assert d == {"id": "n2", "type": "LLM", "data": data}

    def test_from_dict_restores_node(self, nodes_mod):
        """Node.from_dict should call create_node to get defaults then merge."""
        raw = {
            "id": "42",
            "type": "Input",
            "data": {"input_text": "custom text"}
        }
        node = nodes_mod.Node.from_dict(raw)
        assert node.id == "42"
        assert node.type == "Input"
        # Custom data must survive the merge
        assert node.data.get("input_text") == "custom text"

    def test_from_dict_invalid_type_raises(self, nodes_mod):
        raw = {"id": "x", "type": "NonExistent", "data": {}}
        with pytest.raises(ValueError):
            nodes_mod.Node.from_dict(raw)


# ---------------------------------------------------------------------------
# Edge class
# ---------------------------------------------------------------------------

class TestEdgeClass:
    def test_create_edge(self, nodes_mod):
        edge = nodes_mod.Edge("e1", "node-1", "node-2")
        assert edge.id == "e1"
        assert edge.source == "node-1"
        assert edge.target == "node-2"

    def test_to_dict(self, nodes_mod):
        edge = nodes_mod.Edge("e2", "a", "b")
        assert edge.to_dict() == {"id": "e2", "source": "a", "target": "b"}

    def test_from_dict(self, nodes_mod):
        edge = nodes_mod.Edge.from_dict({"id": "e3", "source": "x", "target": "y"})
        assert edge.source == "x"
        assert edge.target == "y"

    def test_from_dict_missing_key_raises(self, nodes_mod):
        with pytest.raises(KeyError):
            nodes_mod.Edge.from_dict({"id": "bad"})


# ---------------------------------------------------------------------------
# create_node factory
# ---------------------------------------------------------------------------

class TestCreateNode:
    def test_creates_input_node(self, nodes_mod):
        node = nodes_mod.create_node("1", "Input")
        assert node.type == "Input"
        assert "input_type" in node.data
        assert "input_text" in node.data

    def test_creates_llm_node(self, nodes_mod):
        node = nodes_mod.create_node("2", "LLM")
        assert node.type == "LLM"
        assert "model_name" in node.data
        assert "temperature" in node.data
        assert "max_tokens" in node.data
        assert "conversation_history" in node.data

    def test_creates_output_node(self, nodes_mod):
        node = nodes_mod.create_node("3", "Output")
        assert node.type == "Output"
        assert "output_type" in node.data
        assert "output_label" in node.data

    def test_creates_processing_node(self, nodes_mod):
        node = nodes_mod.create_node("4", "Processing")
        assert "processing_type" in node.data

    def test_creates_data_retrieval_node(self, nodes_mod):
        node = nodes_mod.create_node("5", "DataRetrieval")
        assert "retrieval_type" in node.data

    def test_creates_control_node(self, nodes_mod):
        node = nodes_mod.create_node("6", "Control")
        assert "control_type" in node.data

    def test_creates_integration_node(self, nodes_mod):
        node = nodes_mod.create_node("7", "Integration")
        assert "integration_type" in node.data

    def test_creates_utility_node(self, nodes_mod):
        node = nodes_mod.create_node("8", "Utility")
        assert "utility_type" in node.data

    def test_invalid_type_raises(self, nodes_mod):
        with pytest.raises(ValueError, match="not available"):
            nodes_mod.create_node("9", "Bogus")

    def test_node_id_is_preserved(self, nodes_mod):
        node = nodes_mod.create_node("special-id-99", "Output")
        assert node.id == "special-id-99"


# ---------------------------------------------------------------------------
# validate_workflow
# ---------------------------------------------------------------------------

class TestValidateWorkflow:
    def _make_simple_workflow(self, nodes_mod):
        """Input -> Output with one edge — minimal valid workflow."""
        n_in = nodes_mod.create_node("1", "Input")
        n_out = nodes_mod.create_node("2", "Output")
        edge = nodes_mod.Edge("e1", "1", "2")
        return [n_in, n_out], [edge]

    def test_valid_workflow(self, nodes_mod):
        nodes, edges = self._make_simple_workflow(nodes_mod)
        result = nodes_mod.validate_workflow(nodes, edges)
        assert result["valid"] is True

    def test_missing_input_node(self, nodes_mod):
        n_out = nodes_mod.create_node("2", "Output")
        result = nodes_mod.validate_workflow([n_out], [])
        assert result["valid"] is False
        assert "Input" in result["error"]

    def test_missing_output_node(self, nodes_mod):
        n_in = nodes_mod.create_node("1", "Input")
        result = nodes_mod.validate_workflow([n_in], [])
        assert result["valid"] is False
        assert "Output" in result["error"]

    def test_disconnected_node(self, nodes_mod):
        n_in = nodes_mod.create_node("1", "Input")
        n_out = nodes_mod.create_node("2", "Output")
        n_extra = nodes_mod.create_node("3", "Processing")
        edge = nodes_mod.Edge("e1", "1", "2")
        result = nodes_mod.validate_workflow([n_in, n_out, n_extra], [edge])
        assert result["valid"] is False

    def test_edge_to_nonexistent_node_fails(self, nodes_mod):
        n_in = nodes_mod.create_node("1", "Input")
        n_out = nodes_mod.create_node("2", "Output")
        bad_edge = nodes_mod.Edge("e1", "1", "999")  # 999 doesn't exist
        result = nodes_mod.validate_workflow([n_in, n_out], [bad_edge])
        assert result["valid"] is False

    def test_cyclic_workflow_fails(self, nodes_mod):
        n_in = nodes_mod.create_node("1", "Input")
        n_out = nodes_mod.create_node("2", "Output")
        n_proc = nodes_mod.create_node("3", "Processing")
        edges = [
            nodes_mod.Edge("e1", "1", "3"),
            nodes_mod.Edge("e2", "3", "2"),
            nodes_mod.Edge("e3", "2", "3"),  # cycle
        ]
        result = nodes_mod.validate_workflow([n_in, n_out, n_proc], edges)
        assert result["valid"] is False
        assert "cycle" in result["error"].lower()


# ---------------------------------------------------------------------------
# has_cycle / path_exists_input_to_output
# ---------------------------------------------------------------------------

class TestGraphAlgorithms:
    def test_no_cycle_in_linear_chain(self, nodes_mod):
        n1 = nodes_mod.create_node("1", "Input")
        n2 = nodes_mod.create_node("2", "Processing")
        n3 = nodes_mod.create_node("3", "Output")
        edges = [nodes_mod.Edge("e1", "1", "2"), nodes_mod.Edge("e2", "2", "3")]
        assert nodes_mod.has_cycle([n1, n2, n3], edges) is False

    def test_cycle_detected(self, nodes_mod):
        n1 = nodes_mod.create_node("1", "Input")
        n2 = nodes_mod.create_node("2", "Processing")
        edges = [nodes_mod.Edge("e1", "1", "2"), nodes_mod.Edge("e2", "2", "1")]
        assert nodes_mod.has_cycle([n1, n2], edges) is True

    def test_path_exists_simple(self, nodes_mod):
        n1 = nodes_mod.create_node("1", "Input")
        n2 = nodes_mod.create_node("2", "Output")
        edges = [nodes_mod.Edge("e1", "1", "2")]
        assert nodes_mod.path_exists_input_to_output([n1, n2], edges) is True

    def test_path_not_exists_disconnected(self, nodes_mod):
        n1 = nodes_mod.create_node("1", "Input")
        n2 = nodes_mod.create_node("2", "Output")
        assert nodes_mod.path_exists_input_to_output([n1, n2], []) is False


# ---------------------------------------------------------------------------
# handle_input_node
# ---------------------------------------------------------------------------

class TestHandleInputNode:
    def test_text_input(self, nodes_mod):
        node = nodes_mod.create_node("1", "Input")
        node.data["input_type"] = "Text"
        node.data["input_text"] = "hello world"
        result = nodes_mod.handle_input_node(node)
        assert result == "hello world"

    def test_api_input_success(self, nodes_mod):
        node = nodes_mod.create_node("1", "Input")
        node.data["input_type"] = "API"
        node.data["api_endpoint"] = "https://api.example.com/data"

        mock_response = MagicMock()
        mock_response.text = "api data"
        mock_response.raise_for_status = MagicMock()

        # The module already holds a reference to the mock requests injected at
        # import time.  Patch that mock's .get attribute directly.
        nodes_mod.requests.get.return_value = mock_response
        nodes_mod.requests.get.side_effect = None
        result = nodes_mod.handle_input_node(node)
        assert result == "api data"

    def test_api_input_failure_returns_error(self, nodes_mod):
        node = nodes_mod.create_node("1", "Input")
        node.data["input_type"] = "API"
        node.data["api_endpoint"] = "https://bad.endpoint"

        # Inject a RequestException via the already-mocked requests object
        nodes_mod.requests.RequestException = Exception
        nodes_mod.requests.get.side_effect = Exception("connection refused")
        result = nodes_mod.handle_input_node(node)
        # Restore for subsequent tests
        nodes_mod.requests.get.side_effect = None
        assert "Error" in result

    def test_unknown_input_type(self, nodes_mod):
        node = nodes_mod.create_node("1", "Input")
        node.data["input_type"] = "Telepathy"
        result = nodes_mod.handle_input_node(node)
        assert "Unknown" in result


# ---------------------------------------------------------------------------
# handle_output_node
# ---------------------------------------------------------------------------

class TestHandleOutputNode:
    def test_text_output(self, nodes_mod):
        node = nodes_mod.create_node("2", "Output")
        node.data["output_type"] = "Text"
        node.data["output_label"] = "Result"
        edge = nodes_mod.Edge("e1", "1", "2")

        def fake_process(nid):
            return "some text"

        result = nodes_mod.handle_output_node(node, [edge], fake_process)
        assert "Result" in result
        assert "some text" in result

    def test_output_with_no_edges_returns_error(self, nodes_mod):
        node = nodes_mod.create_node("2", "Output")
        result = nodes_mod.handle_output_node(node, [], lambda nid: "")
        assert "Error" in result

    def test_unknown_output_type(self, nodes_mod):
        node = nodes_mod.create_node("2", "Output")
        node.data["output_type"] = "Hologram"
        edge = nodes_mod.Edge("e1", "1", "2")
        result = nodes_mod.handle_output_node(node, [edge], lambda nid: "data")
        assert "Unknown" in result


# ---------------------------------------------------------------------------
# handle_processing_node
# ---------------------------------------------------------------------------

class TestHandleProcessingNode:
    def _make_edge(self, nodes_mod, source="1", target="2"):
        return nodes_mod.Edge("e1", source, target)

    def test_preprocessing_lowercasing(self, nodes_mod):
        node = nodes_mod.create_node("2", "Processing")
        node.data["processing_type"] = "Preprocessing"
        node.data["preprocessing_steps"] = ["Lowercasing"]
        edge = self._make_edge(nodes_mod)
        result = nodes_mod.handle_processing_node(node, [edge], lambda nid: "HELLO WORLD")
        assert result == "hello world"

    def test_preprocessing_remove_punctuation(self, nodes_mod):
        node = nodes_mod.create_node("2", "Processing")
        node.data["processing_type"] = "Preprocessing"
        node.data["preprocessing_steps"] = ["Remove Punctuation"]
        edge = self._make_edge(nodes_mod)
        result = nodes_mod.handle_processing_node(node, [edge], lambda nid: "Hello, World!")
        assert "," not in result
        assert "!" not in result

    def test_vectorization_returns_truncated(self, nodes_mod):
        node = nodes_mod.create_node("2", "Processing")
        node.data["processing_type"] = "Vectorization"
        edge = self._make_edge(nodes_mod)
        result = nodes_mod.handle_processing_node(node, [edge], lambda nid: "data to vectorize")
        assert "Vectorized" in result

    def test_no_edges_returns_error(self, nodes_mod):
        node = nodes_mod.create_node("2", "Processing")
        result = nodes_mod.handle_processing_node(node, [], lambda nid: "")
        assert "Error" in result

    def test_unknown_processing_type(self, nodes_mod):
        node = nodes_mod.create_node("2", "Processing")
        node.data["processing_type"] = "TeleportData"
        edge = self._make_edge(nodes_mod)
        result = nodes_mod.handle_processing_node(node, [edge], lambda nid: "data")
        assert "Unknown" in result


# ---------------------------------------------------------------------------
# handle_utility_node
# ---------------------------------------------------------------------------

class TestHandleUtilityNode:
    def _edge(self, nodes_mod):
        return nodes_mod.Edge("e1", "1", "2")

    def test_format_json_valid(self, nodes_mod):
        node = nodes_mod.create_node("2", "Utility")
        node.data["utility_type"] = "Format"
        node.data["format_type"] = "JSON"
        result = nodes_mod.handle_utility_node(node, [self._edge(nodes_mod)], lambda nid: '{"a":1}')
        assert '"a"' in result

    def test_format_html(self, nodes_mod):
        node = nodes_mod.create_node("2", "Utility")
        node.data["utility_type"] = "Format"
        node.data["format_type"] = "HTML"
        result = nodes_mod.handle_utility_node(node, [self._edge(nodes_mod)], lambda nid: "hello")
        assert "<html>" in result

    def test_format_markdown(self, nodes_mod):
        node = nodes_mod.create_node("2", "Utility")
        node.data["utility_type"] = "Format"
        node.data["format_type"] = "Markdown"
        result = nodes_mod.handle_utility_node(node, [self._edge(nodes_mod)], lambda nid: "hello")
        assert "```" in result

    def test_transform_uppercase(self, nodes_mod):
        node = nodes_mod.create_node("2", "Utility")
        node.data["utility_type"] = "Transform"
        node.data["transform_type"] = "Uppercase"
        result = nodes_mod.handle_utility_node(node, [self._edge(nodes_mod)], lambda nid: "hello")
        assert result == "HELLO"

    def test_transform_lowercase(self, nodes_mod):
        node = nodes_mod.create_node("2", "Utility")
        node.data["utility_type"] = "Transform"
        node.data["transform_type"] = "Lowercase"
        result = nodes_mod.handle_utility_node(node, [self._edge(nodes_mod)], lambda nid: "HELLO")
        assert result == "hello"

    def test_transform_count(self, nodes_mod):
        node = nodes_mod.create_node("2", "Utility")
        node.data["utility_type"] = "Transform"
        node.data["transform_type"] = "Count"
        result = nodes_mod.handle_utility_node(node, [self._edge(nodes_mod)], lambda nid: "abc")
        assert "3" in result

    def test_no_edges_returns_error(self, nodes_mod):
        node = nodes_mod.create_node("2", "Utility")
        result = nodes_mod.handle_utility_node(node, [], lambda nid: "")
        assert "Error" in result

    def test_unknown_utility_type(self, nodes_mod):
        node = nodes_mod.create_node("2", "Utility")
        node.data["utility_type"] = "Telekinesis"
        result = nodes_mod.handle_utility_node(node, [self._edge(nodes_mod)], lambda nid: "data")
        assert "Unknown" in result


# ---------------------------------------------------------------------------
# handle_control_node
# ---------------------------------------------------------------------------

class TestHandleControlNode:
    def _edge(self, nodes_mod):
        return nodes_mod.Edge("e1", "1", "2")

    def test_conditional_true_branch(self, nodes_mod):
        node = nodes_mod.create_node("2", "Control")
        node.data["control_type"] = "Conditional"
        # The module does: eval(condition.replace('{input}', repr(input_data)))
        # repr("hello") -> "'hello'" so condition becomes: len('hello') > 2 -> True
        node.data["condition"] = "len({input}) > 2"
        node.data["true_branch"] = "long"
        node.data["false_branch"] = "short"
        result = nodes_mod.handle_control_node(node, [self._edge(nodes_mod)], lambda nid: "hello")
        assert "TRUE" in result

    def test_conditional_false_branch(self, nodes_mod):
        node = nodes_mod.create_node("2", "Control")
        node.data["control_type"] = "Conditional"
        # repr("hi") -> "'hi'", so condition becomes: len('hi') > 100 -> False
        node.data["condition"] = "len({input}) > 100"
        node.data["true_branch"] = "long"
        node.data["false_branch"] = "short"
        result = nodes_mod.handle_control_node(node, [self._edge(nodes_mod)], lambda nid: "hi")
        assert "FALSE" in result

    def test_loop_produces_iterations(self, nodes_mod):
        node = nodes_mod.create_node("2", "Control")
        node.data["control_type"] = "Loop"
        node.data["iterations"] = 3
        node.data["loop_body"] = "step {i}"
        result = nodes_mod.handle_control_node(node, [self._edge(nodes_mod)], lambda nid: "data")
        assert "Iteration 1" in result
        assert "Iteration 2" in result
        assert "Iteration 3" in result

    def test_no_edges_returns_error(self, nodes_mod):
        node = nodes_mod.create_node("2", "Control")
        result = nodes_mod.handle_control_node(node, [], lambda nid: "")
        assert "Error" in result


# ---------------------------------------------------------------------------
# handle_data_retrieval_node
# ---------------------------------------------------------------------------

class TestHandleDataRetrievalNode:
    def test_database_sql_retrieval(self, nodes_mod):
        node = nodes_mod.create_node("1", "DataRetrieval")
        node.data["retrieval_type"] = "Database"
        node.data["database_type"] = "SQL"
        node.data["query"] = "SELECT * FROM users"
        result = nodes_mod.handle_data_retrieval_node(node, [], lambda nid: "")
        assert "SQL" in result

    def test_file_no_path_returns_error(self, nodes_mod):
        node = nodes_mod.create_node("1", "DataRetrieval")
        node.data["retrieval_type"] = "File"
        node.data["file_path"] = ""
        result = nodes_mod.handle_data_retrieval_node(node, [], lambda nid: "")
        assert "Error" in result

    def test_web_no_url_returns_error(self, nodes_mod):
        node = nodes_mod.create_node("1", "DataRetrieval")
        node.data["retrieval_type"] = "Web"
        node.data["web_url"] = ""
        result = nodes_mod.handle_data_retrieval_node(node, [], lambda nid: "")
        assert "Error" in result

    def test_unknown_retrieval_type(self, nodes_mod):
        node = nodes_mod.create_node("1", "DataRetrieval")
        node.data["retrieval_type"] = "Psychic"
        result = nodes_mod.handle_data_retrieval_node(node, [], lambda nid: "")
        assert "Unknown" in result


# ---------------------------------------------------------------------------
# construct_prompt
# ---------------------------------------------------------------------------

class TestConstructPrompt:
    def test_basic_prompt_includes_input(self, nodes_mod):
        node = nodes_mod.create_node("1", "LLM")
        node.data["agent_type"] = "None"
        node.data["metacognitive_type"] = "None"
        node.data["voice_type"] = "None"
        node.data["identity_type"] = "None"
        node.data["conversation_history"] = []
        result = nodes_mod.construct_prompt(node, "Do the thing", "my input text")
        assert "Do the thing" in result
        assert "my input text" in result

    def test_conversation_history_appended(self, nodes_mod):
        node = nodes_mod.create_node("1", "LLM")
        node.data["agent_type"] = "None"
        node.data["metacognitive_type"] = "None"
        node.data["voice_type"] = "None"
        node.data["identity_type"] = "None"
        node.data["conversation_history"] = [
            {"role": "user", "content": "prior question"},
            {"role": "assistant", "content": "prior answer"}
        ]
        result = nodes_mod.construct_prompt(node, "base", "input")
        assert "prior question" in result


# ---------------------------------------------------------------------------
# parse_openai_response / parse_ollama_response
# ---------------------------------------------------------------------------

class TestResponseParsers:
    def test_parse_openai_valid_json(self, nodes_mod):
        json_str = json.dumps({"nodes": [], "edges": []})
        result = nodes_mod.parse_openai_response(json_str)
        assert result == {"nodes": [], "edges": []}

    def test_parse_openai_embedded_json(self, nodes_mod):
        text = 'Sure! Here you go: {"nodes": [], "edges": []} That is the workflow.'
        result = nodes_mod.parse_openai_response(text)
        assert "nodes" in result
        assert "edges" in result

    def test_parse_openai_no_json_raises(self, nodes_mod):
        with pytest.raises(ValueError):
            nodes_mod.parse_openai_response("No JSON here at all")

    def test_parse_openai_missing_keys_raises(self, nodes_mod):
        with pytest.raises(ValueError):
            nodes_mod.parse_openai_response('{"only_nodes": []}')

    def test_parse_ollama_valid(self, nodes_mod):
        payload = json.dumps({"nodes": [{"id": "1", "type": "Input", "data": {}}], "edges": []})
        # Ollama response is split on double newline; last message contains JSON
        response = f"preamble\n\n{payload}"
        result = nodes_mod.parse_ollama_response(response)
        assert "nodes" in result


# ---------------------------------------------------------------------------
# call_api_and_decode_response
# ---------------------------------------------------------------------------

class TestCallApiAndDecodeResponse:
    def test_valid_response_returned(self, nodes_mod):
        payload = json.dumps({"nodes": [], "edges": []})
        fn = MagicMock(return_value=payload)
        result = nodes_mod.call_api_and_decode_response(fn)
        assert result == {"nodes": [], "edges": []}

    def test_empty_response_raises(self, nodes_mod):
        fn = MagicMock(return_value="")
        with pytest.raises(ValueError):
            nodes_mod.call_api_and_decode_response(fn)

    def test_invalid_json_raises(self, nodes_mod):
        fn = MagicMock(return_value="this is not json")
        with pytest.raises(ValueError):
            nodes_mod.call_api_and_decode_response(fn)

    def test_missing_nodes_key_raises(self, nodes_mod):
        fn = MagicMock(return_value=json.dumps({"edges": []}))
        with pytest.raises(ValueError):
            nodes_mod.call_api_and_decode_response(fn)


# ---------------------------------------------------------------------------
# AVAILABLE_NODE_TYPES / NODE_COLORS / NODE_EMOJIS constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_all_node_types_enabled(self, nodes_mod):
        for name, available in nodes_mod.AVAILABLE_NODE_TYPES.items():
            assert available is True, f"Node type '{name}' should be enabled"

    def test_colors_defined_for_all_types(self, nodes_mod):
        for node_type in nodes_mod.AVAILABLE_NODE_TYPES:
            assert node_type in nodes_mod.NODE_COLORS

    def test_emojis_defined_for_all_types(self, nodes_mod):
        for node_type in nodes_mod.AVAILABLE_NODE_TYPES:
            assert node_type in nodes_mod.NODE_EMOJIS
