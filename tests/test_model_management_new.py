"""
Tests for model_management.py — database operations, model tracking,
status updates, and statistics functions.

All tests use an in-memory (or temp-file) SQLite database and mock Streamlit
and Ollama API calls to avoid external dependencies.
"""

import sqlite3
import os
import sys
import tempfile
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixture: isolated temporary database per test
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path):
    """Return a path to a fresh temporary SQLite database file."""
    return str(tmp_path / "test_models.db")


@pytest.fixture()
def mm(db_path, monkeypatch):
    """
    Import model_management with all external dependencies mocked, and
    redirect DATABASE_PATH to the temp db so tests stay isolated.
    """
    st_mock = MagicMock()
    ollama_utils_mock = MagicMock(
        get_available_models=MagicMock(return_value=["llama3", "mistral"]),
        get_ollama_resource_usage=MagicMock(return_value={
            "cpu_usage": "25%",
            "memory_usage": "50%",
            "gpu_usage": "N/A"
        }),
        show_model_info=MagicMock(return_value={"size": 4_000_000_000, "modified_at": "2024-01-01"})
    )

    plt_mock = MagicMock()
    plt_mock.figure.return_value = MagicMock()

    with patch.dict("sys.modules", {
        "streamlit": st_mock,
        "matplotlib": MagicMock(),
        "matplotlib.pyplot": plt_mock,
        "plotly": MagicMock(),
        "plotly.express": MagicMock(),
        "plotly.graph_objects": MagicMock(),
        "ollama_workbench.providers.ollama_utils": ollama_utils_mock,
        "ollama_workbench.models.model_capability_registry": MagicMock(
            get_model_capabilities=MagicMock(return_value={
                "vision": False, "tools": False, "embedding": False
            })
        ),
    }):
        import importlib
        import ollama_workbench.models.model_management as mm_mod
        # Redirect DB path before init
        monkeypatch.setattr(mm_mod, "DATABASE_PATH", db_path)
        importlib.reload(mm_mod)
        monkeypatch.setattr(mm_mod, "DATABASE_PATH", db_path)
        mm_mod.init_db()
        yield mm_mod


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

class TestInitDb:
    def test_creates_all_four_tables(self, mm, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        expected = {"model_usage", "model_metadata", "model_performance", "resource_utilization"}
        assert expected.issubset(tables)

    def test_idempotent_can_run_twice(self, mm):
        """Calling init_db a second time must not raise."""
        mm.init_db()


# ---------------------------------------------------------------------------
# log_model_usage
# ---------------------------------------------------------------------------

class TestLogModelUsage:
    def test_inserts_usage_record(self, mm, db_path):
        mm.log_model_usage("llama3", tokens_generated=100, response_time=1.5)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT model_name, tokens_generated, response_time FROM model_usage")
        rows = cursor.fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0] == ("llama3", 100, 1.5)

    def test_inserts_with_custom_operation_type(self, mm, db_path):
        mm.log_model_usage("mistral", 50, 0.8, operation_type="embed")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT operation_type FROM model_usage")
        row = cursor.fetchone()
        conn.close()
        assert row[0] == "embed"

    def test_multiple_entries_accumulate(self, mm, db_path):
        mm.log_model_usage("llama3", 10, 0.1)
        mm.log_model_usage("llama3", 20, 0.2)
        mm.log_model_usage("mistral", 30, 0.3)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM model_usage")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 3


# ---------------------------------------------------------------------------
# log_model_performance
# ---------------------------------------------------------------------------

class TestLogModelPerformance:
    def test_inserts_performance_record(self, mm, db_path):
        mm.log_model_performance("llama3", "test prompt", 45.0, 1.2, 0.7, 512)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT model_name, tokens_per_second, latency, temperature, max_tokens "
            "FROM model_performance"
        )
        row = cursor.fetchone()
        conn.close()
        assert row == ("llama3", 45.0, 1.2, 0.7, 512)

    def test_prompt_text_stored(self, mm, db_path):
        mm.log_model_performance("mistral", "summarize this", 30.0, 2.0, 0.5, 1024)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT prompt_text FROM model_performance")
        row = cursor.fetchone()
        conn.close()
        assert row[0] == "summarize this"


# ---------------------------------------------------------------------------
# get_model_usage_stats
# ---------------------------------------------------------------------------

class TestGetModelUsageStats:
    def _seed_usage(self, db_path, model="llama3", days_ago=0, tokens=100, requests_n=1):
        """Insert usage records directly into the db for a given timestamp."""
        ts = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for _ in range(requests_n):
            cursor.execute(
                "INSERT INTO model_usage (model_name, timestamp, tokens_generated, response_time, operation_type) "
                "VALUES (?, ?, ?, ?, ?)",
                (model, ts, tokens, 1.0, "generate")
            )
        conn.commit()
        conn.close()

    def test_returns_dict_with_expected_keys(self, mm, db_path):
        self._seed_usage(db_path)
        result = mm.get_model_usage_stats("llama3", days=30)
        assert "dates" in result
        assert "tokens" in result
        assert "requests" in result
        assert "avg_response_times" in result

    def test_all_models_when_no_filter(self, mm, db_path):
        self._seed_usage(db_path, model="llama3")
        self._seed_usage(db_path, model="mistral")
        result = mm.get_model_usage_stats(model_name=None, days=30)
        assert len(result["dates"]) >= 1

    def test_filters_by_model_name(self, mm, db_path):
        self._seed_usage(db_path, model="llama3", tokens=500)
        self._seed_usage(db_path, model="mistral", tokens=200)
        result = mm.get_model_usage_stats("llama3", days=30)
        # Only llama3 data returned
        assert all(t == 500 for t in result["tokens"])

    def test_empty_result_for_no_data(self, mm, db_path):
        result = mm.get_model_usage_stats("unknown-model", days=30)
        assert result["dates"] == []
        assert result["tokens"] == []

    def test_respects_days_limit(self, mm, db_path):
        self._seed_usage(db_path, model="llama3", days_ago=60, tokens=999)
        result = mm.get_model_usage_stats("llama3", days=30)
        # 60-day-old record should not appear in 30-day window
        assert result["dates"] == []


# ---------------------------------------------------------------------------
# get_model_performance
# ---------------------------------------------------------------------------

class TestGetModelPerformance:
    def _seed_performance(self, db_path, model="llama3", tps=40.0, latency=1.0, days_ago=0):
        ts = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO model_performance (model_name, timestamp, prompt_text, "
            "tokens_per_second, latency, temperature, max_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (model, ts, "test", tps, latency, 0.7, 512)
        )
        conn.commit()
        conn.close()

    def test_individual_model_returns_timestamps(self, mm, db_path):
        self._seed_performance(db_path, model="llama3")
        result = mm.get_model_performance("llama3", days=30)
        assert "timestamps" in result
        assert "tokens_per_second" in result
        assert "latencies" in result

    def test_all_models_comparative_format(self, mm, db_path):
        self._seed_performance(db_path, model="llama3", tps=50.0)
        self._seed_performance(db_path, model="mistral", tps=35.0)
        result = mm.get_model_performance(model_name=None, days=30)
        assert "models" in result
        assert "avg_tokens_per_second" in result
        assert "avg_latencies" in result

    def test_empty_for_unknown_model(self, mm, db_path):
        result = mm.get_model_performance("no-such-model", days=30)
        assert result["timestamps"] == []


# ---------------------------------------------------------------------------
# get_resource_utilization
# ---------------------------------------------------------------------------

class TestGetResourceUtilization:
    def _seed_resource(self, db_path, model=None, cpu=30.0, mem=60.0, gpu=0.0, hours_ago=1):
        ts = (datetime.now() - timedelta(hours=hours_ago)).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO resource_utilization (timestamp, model_name, cpu_usage, memory_usage, gpu_usage) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts, model, cpu, mem, gpu)
        )
        conn.commit()
        conn.close()

    def test_returns_expected_keys(self, mm, db_path):
        self._seed_resource(db_path)
        result = mm.get_resource_utilization(hours=24)
        assert "timestamps" in result
        assert "cpu_usage" in result
        assert "memory_usage" in result
        assert "gpu_usage" in result

    def test_empty_for_no_data(self, mm, db_path):
        result = mm.get_resource_utilization(hours=1)
        assert result["timestamps"] == []

    def test_model_specific_filter(self, mm, db_path):
        self._seed_resource(db_path, model="llama3", cpu=75.0)
        self._seed_resource(db_path, model="mistral", cpu=10.0)
        result = mm.get_resource_utilization(model_name="llama3", hours=24)
        assert all(v == 75.0 for v in result["cpu_usage"])


# ---------------------------------------------------------------------------
# get_models_by_usage
# ---------------------------------------------------------------------------

class TestGetModelsByUsage:
    def _seed_usage(self, db_path, model, count, tokens=100):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for _ in range(count):
            cursor.execute(
                "INSERT INTO model_usage (model_name, timestamp, tokens_generated, response_time, operation_type) "
                "VALUES (?, ?, ?, ?, ?)",
                (model, ts, tokens, 1.0, "generate")
            )
        conn.commit()
        conn.close()

    def test_returns_expected_keys(self, mm, db_path):
        self._seed_usage(db_path, "llama3", 3)
        result = mm.get_models_by_usage(days=30)
        assert "model_names" in result
        assert "use_counts" in result
        assert "total_tokens" in result

    def test_orders_by_use_count_descending(self, mm, db_path):
        self._seed_usage(db_path, "mistral", 1)
        self._seed_usage(db_path, "llama3", 5)
        result = mm.get_models_by_usage(days=30)
        assert result["model_names"][0] == "llama3"

    def test_empty_when_no_data(self, mm, db_path):
        result = mm.get_models_by_usage(days=30)
        assert result["model_names"] == []
        assert result["use_counts"] == []


# ---------------------------------------------------------------------------
# get_operation_types
# ---------------------------------------------------------------------------

class TestGetOperationTypes:
    def _seed_ops(self, db_path, ops):
        """ops: list of (model, op_type)"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for model, op in ops:
            cursor.execute(
                "INSERT INTO model_usage (model_name, timestamp, tokens_generated, response_time, operation_type) "
                "VALUES (?, ?, ?, ?, ?)",
                (model, ts, 10, 0.5, op)
            )
        conn.commit()
        conn.close()

    def test_returns_expected_keys(self, mm, db_path):
        self._seed_ops(db_path, [("llama3", "generate")])
        result = mm.get_operation_types(days=30)
        assert "operation_types" in result
        assert "operation_counts" in result

    def test_counts_operations_correctly(self, mm, db_path):
        self._seed_ops(db_path, [
            ("llama3", "generate"),
            ("llama3", "generate"),
            ("mistral", "embed"),
        ])
        result = mm.get_operation_types(days=30)
        op_idx = result["operation_types"].index("generate")
        assert result["operation_counts"][op_idx] == 2

    def test_model_filter_applied(self, mm, db_path):
        self._seed_ops(db_path, [
            ("llama3", "generate"),
            ("mistral", "embed"),
        ])
        result = mm.get_operation_types(model_name="llama3", days=30)
        assert "generate" in result["operation_types"]
        assert "embed" not in result["operation_types"]


# ---------------------------------------------------------------------------
# generate_simulation_data
# ---------------------------------------------------------------------------

class TestGenerateSimulationData:
    def test_returns_three_components(self, mm):
        usage_data, perf_data, resource_data = mm.generate_simulation_data(["llama3", "mistral"])

        assert isinstance(usage_data, dict)
        assert isinstance(perf_data, dict)
        assert isinstance(resource_data, dict)

    def test_simulation_skips_embed_models(self, mm):
        usage_data, _, _ = mm.generate_simulation_data(["llama3", "nomic-embed-text"])
        assert "nomic-embed-text" not in usage_data

    def test_resource_data_has_expected_keys(self, mm):
        _, _, resource_data = mm.generate_simulation_data(["llama3"])
        assert "timestamps" in resource_data
        assert "cpu_usage" in resource_data
        assert "memory_usage" in resource_data
        assert "gpu_usage" in resource_data

    def test_simulation_data_has_31_days(self, mm):
        usage_data, _, _ = mm.generate_simulation_data(["llama3"])
        assert len(usage_data["llama3"]["dates"]) == 31

    def test_empty_model_list_returns_empty_dicts(self, mm):
        usage_data, perf_data, _ = mm.generate_simulation_data([])
        assert usage_data == {}
        assert perf_data == {}


# ---------------------------------------------------------------------------
# update_model_metadata
# ---------------------------------------------------------------------------

class TestUpdateModelMetadata:
    def test_inserts_metadata_for_available_models(self, mm, db_path):
        mm.update_model_metadata()

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT model_name FROM model_metadata")
        models = {row[0] for row in cursor.fetchall()}
        conn.close()

        # Mocked get_available_models returns ["llama3", "mistral"]
        assert "llama3" in models
        assert "mistral" in models

    def test_skips_embed_models(self, mm, db_path):
        with patch.object(
            mm,
            "get_available_models",
            return_value=["llama3", "nomic-embed-text"]
        ):
            mm.update_model_metadata()

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT model_name FROM model_metadata")
        models = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "nomic-embed-text" not in models

    def test_updates_existing_metadata(self, mm, db_path):
        # First call inserts
        mm.update_model_metadata()

        # Modify the mock to return updated size, then call again
        with patch.object(
            mm,
            "show_model_info",
            return_value={"size": 8_000_000_000, "modified_at": "2024-06-01"}
        ):
            mm.update_model_metadata()

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT size_bytes FROM model_metadata WHERE model_name = 'llama3'")
        row = cursor.fetchone()
        conn.close()
        assert row[0] == 8_000_000_000
