"""
Tests for tool_playground.py - Tool definition parsing, execution dispatch, result formatting.

Tests focus on pure logic functions and helper functions that do not require
a running Streamlit instance or live Ollama API.
"""

import json
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _import_module():
    """Import tool_playground with Streamlit and ollama mocked out."""
    st_mock = MagicMock()
    st_mock.session_state = {}
    ollama_mock = MagicMock()

    # matplotlib.pyplot tries to build a font cache via system_profiler on macOS,
    # which hangs in the sandbox.  Mock the whole matplotlib stack.
    mpl_mock = MagicMock()
    mpl_mock.figure.return_value = MagicMock()
    plt_mock = MagicMock()
    plt_mock.figure.return_value = MagicMock()

    with patch.dict("sys.modules", {
        "streamlit": st_mock,
        "ollama": ollama_mock,
        "matplotlib": mpl_mock,
        "matplotlib.pyplot": plt_mock,
        "ollama_workbench.providers.ollama_utils": MagicMock(),
        "ollama_workbench.ui.mcp_tools": MagicMock(
            get_available_mcp_tools=MagicMock(return_value=({}, {})),
            execute_mcp_tool=MagicMock(return_value={"error": "not available"}),
        ),
        "ollama_workbench.models.model_capability_registry": MagicMock(
            filter_models_by_capability=MagicMock(return_value=[]),
            is_tools_capable=MagicMock(return_value=False),
        ),
    }):
        import importlib
        import ollama_workbench.ui.tool_playground as tp
        importlib.reload(tp)
        return tp


@pytest.fixture(scope="module")
def tp():
    return _import_module()


# ---------------------------------------------------------------------------
# sanitize_for_json
# ---------------------------------------------------------------------------

class TestSanitizeForJson:
    def test_plain_dict_passes_through(self, tp):
        data = {"key": "value", "num": 42}
        assert tp.sanitize_for_json(data) == data

    def test_strips_client_key(self, tp):
        data = {"key": "value", "_client": object()}
        result = tp.sanitize_for_json(data)
        assert "_client" not in result
        assert result["key"] == "value"

    def test_nested_dict(self, tp):
        data = {"outer": {"inner": 1, "_client": "bad"}}
        result = tp.sanitize_for_json(data)
        assert result["outer"] == {"inner": 1}

    def test_list_of_dicts(self, tp):
        data = [{"a": 1}, {"b": 2}]
        assert tp.sanitize_for_json(data) == data

    def test_object_with_to_dict(self, tp):
        class Obj:
            def to_dict(self):
                return {"x": 99}
        assert tp.sanitize_for_json(Obj()) == {"x": 99}

    def test_non_serializable_becomes_string(self, tp):
        result = tp.sanitize_for_json(object())
        assert isinstance(result, str)

    def test_primitive_values(self, tp):
        assert tp.sanitize_for_json(42) == 42
        assert tp.sanitize_for_json("hello") == "hello"
        assert tp.sanitize_for_json(True) is True
        assert tp.sanitize_for_json(None) is None

    def test_toolcall_like_object(self, tp):
        """Objects with .function and .id (but no to_dict or __dict__) are serialized to a dict."""
        # Use a simple class instead of MagicMock to avoid MagicMock's magic __dict__ / to_dict
        class FakeFunction:
            name = "calculator"
            arguments = {"a": 1}

        class FakeToolCall:
            id = "tc-1"
            function = FakeFunction()
            # Explicitly no to_dict, no __dict__ proxy that causes recursion

        result = tp.sanitize_for_json(FakeToolCall())
        # The result must be JSON-serializable
        json.dumps(result)


# ---------------------------------------------------------------------------
# TOOL_TEMPLATES
# ---------------------------------------------------------------------------

class TestToolTemplates:
    def test_expected_templates_present(self, tp):
        expected = {"json", "calculator", "weather", "search", "plot", "database"}
        assert expected.issubset(set(tp.TOOL_TEMPLATES.keys()))

    def test_each_template_has_type_and_function(self, tp):
        for name, template in tp.TOOL_TEMPLATES.items():
            assert "type" in template, f"Template '{name}' missing 'type'"
            assert "function" in template, f"Template '{name}' missing 'function'"

    def test_calculator_has_required_fields(self, tp):
        calc = tp.TOOL_TEMPLATES["calculator"]
        assert calc["type"] == "function"
        fn = calc["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn

    def test_json_template_required_fields(self, tp):
        jt = tp.TOOL_TEMPLATES["json"]
        params = jt["function"]["parameters"]
        assert "text" in params["properties"]
        assert "text" in params.get("required", [])


# ---------------------------------------------------------------------------
# TOOL_IMPLEMENTATIONS dispatch
# ---------------------------------------------------------------------------

class TestToolImplementations:
    def test_all_implementations_callable(self, tp):
        for name, fn in tp.TOOL_IMPLEMENTATIONS.items():
            assert callable(fn), f"Implementation for '{name}' is not callable"

    def test_alias_names_resolve_to_same_function(self, tp):
        assert tp.TOOL_IMPLEMENTATIONS["calculator"] is tp.TOOL_IMPLEMENTATIONS["calculate"]
        assert tp.TOOL_IMPLEMENTATIONS["calculator"] is tp.TOOL_IMPLEMENTATIONS["math"]
        assert tp.TOOL_IMPLEMENTATIONS["get_weather"] is tp.TOOL_IMPLEMENTATIONS["weather"]
        assert tp.TOOL_IMPLEMENTATIONS["json_tool"] is tp.TOOL_IMPLEMENTATIONS["json"]


# ---------------------------------------------------------------------------
# calculator_impl
# ---------------------------------------------------------------------------

class TestCalculatorImpl:
    def test_add(self, tp):
        result = tp.calculator_impl(operation="add", a=3, b=4)
        assert "7" in result

    def test_subtract(self, tp):
        result = tp.calculator_impl(operation="subtract", a=10, b=3)
        assert "7" in result

    def test_multiply(self, tp):
        result = tp.calculator_impl(operation="multiply", a=6, b=7)
        assert "42" in result

    def test_divide(self, tp):
        result = tp.calculator_impl(operation="divide", a=10, b=2)
        assert "5" in result

    def test_divide_by_zero(self, tp):
        result = tp.calculator_impl(operation="divide", a=10, b=0)
        assert "zero" in result.lower() or "error" in result.lower()

    def test_expression_evaluation(self, tp):
        result = tp.calculator_impl(expression="2+2*3")
        assert "8" in result

    def test_natural_language_multiplication(self, tp):
        result = tp.calculator_impl(operation="342 multiplied by 15")
        assert "5130" in result

    def test_missing_params_returns_error(self, tp):
        result = tp.calculator_impl()
        assert "error" in result.lower() or "missing" in result.lower()

    def test_unknown_operation_keyword(self, tp):
        # Should still attempt to parse
        result = tp.calculator_impl(operation="what is 6 + 7")
        # Either gives a result or a sensible error
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# weather_impl
# ---------------------------------------------------------------------------

class TestWeatherImpl:
    def test_returns_weather_for_location(self, tp):
        result = tp.weather_impl(location="Berlin")
        assert "Berlin" in result
        assert "Temperature" in result

    def test_celsius_default(self, tp):
        result = tp.weather_impl(location="Paris")
        assert "C" in result

    def test_fahrenheit_unit(self, tp):
        result = tp.weather_impl(location="London", unit="fahrenheit")
        assert "F" in result

    def test_empty_location_returns_error(self, tp):
        result = tp.weather_impl(location="")
        assert "error" in result.lower() or "location" in result.lower()

    def test_none_location_returns_error(self, tp):
        result = tp.weather_impl(location=None)
        assert "error" in result.lower() or "location" in result.lower()

    def test_natural_language_location(self, tp):
        result = tp.weather_impl(location="weather in Pittsburgh")
        assert "Pittsburgh" in result


# ---------------------------------------------------------------------------
# search_impl
# ---------------------------------------------------------------------------

class TestSearchImpl:
    def test_returns_formatted_results(self, tp):
        result = tp.search_impl(query="machine learning")
        assert "machine learning" in result.lower()
        assert "1." in result  # Numbered results

    def test_num_results_respected(self, tp):
        result = tp.search_impl(query="test", num_results=2)
        # Should not have a result 3
        assert "3." not in result

    def test_default_num_results(self, tp):
        result = tp.search_impl(query="python")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# json_tool_impl
# ---------------------------------------------------------------------------

class TestJsonToolImpl:
    def test_validate_valid_json(self, tp):
        result = tp.json_tool_impl(text='{"key": "value"}', operation="validate")
        assert "valid" in result.lower()

    def test_validate_invalid_json(self, tp):
        result = tp.json_tool_impl(text="{bad json}", operation="validate")
        assert "invalid" in result.lower() or "error" in result.lower()

    def test_parse_valid_json(self, tp):
        result = tp.json_tool_impl(text='{"a": 1}', operation="parse")
        assert "a" in result
        assert "1" in result

    def test_format_operation(self, tp):
        result = tp.json_tool_impl(text='{"z":1,"a":2}', operation="format")
        assert "Formatted" in result or "json" in result.lower()

    def test_repair_trailing_comma(self, tp):
        result = tp.json_tool_impl(text='{"a": 1,}', operation="repair")
        assert "repaired" in result.lower() or "valid" in result.lower() or "formatted" in result.lower()

    def test_empty_text_returns_error(self, tp):
        result = tp.json_tool_impl(text="", operation="parse")
        assert "error" in result.lower()

    def test_unknown_operation(self, tp):
        result = tp.json_tool_impl(text='{"a":1}', operation="frobnicate")
        assert "error" in result.lower() or "unknown" in result.lower()


# ---------------------------------------------------------------------------
# query_database_impl
# ---------------------------------------------------------------------------

class TestQueryDatabaseImpl:
    def test_valid_table_users(self, tp):
        result = tp.query_database_impl(table="users")
        assert "users" in result.lower()
        assert "row" in result.lower()

    def test_valid_table_products(self, tp):
        result = tp.query_database_impl(table="products")
        assert "products" in result.lower()

    def test_limit_reduces_results(self, tp):
        result = tp.query_database_impl(table="users", limit=1)
        assert "1 row" in result

    def test_filter_by_field(self, tp):
        result = tp.query_database_impl(table="users", filter={"city": "New York"})
        assert "New York" in result

    def test_filter_no_match_returns_empty_message(self, tp):
        result = tp.query_database_impl(table="users", filter={"city": "Atlantis"})
        assert "no results" in result.lower() or "0" in result
