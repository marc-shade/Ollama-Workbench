"""
Shared pytest fixtures for Ollama Workbench tests.

Provides reusable fixtures for:
- Mock Streamlit session state
- Mock Ollama client and API responses
- Temporary SQLite database with schema from db_init.py
- Mock API keys with load_api_keys patched
"""

import json
import sqlite3

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 0. Auto-clear caches before every test (prevents stale data interference)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear all module-level caches before and after each test."""
    def _clear():
        try:
            import ollama_workbench.providers.ollama_utils as ou
            ou._api_keys_cache = None
            ou._api_keys_cache_time = 0
        except (ImportError, AttributeError):
            pass
        try:
            from ollama_workbench.ui.prompts import clear_prompts_cache
            clear_prompts_cache()
        except (ImportError, AttributeError):
            pass

    _clear()
    yield
    _clear()


# ---------------------------------------------------------------------------
# 1. Mock Streamlit session state
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_session_state():
    """Provide a clean dict that replaces st.session_state for the test.

    Usage:
        def test_something(mock_session_state):
            mock_session_state["selected_model"] = "llama2"
            ...
    The fixture automatically patches streamlit.session_state and restores
    it when the test finishes.
    """
    state = {}
    with patch("streamlit.session_state", state):
        yield state


# ---------------------------------------------------------------------------
# 2. Mock Ollama client
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_ollama_client():
    """Provide a MagicMock Ollama client with sensible default responses.

    The fixture patches ``get_ollama_client`` in the providers module so any
    code that calls it will receive the mock.  It also patches
    ``get_available_models`` to return a small list of model names.

    Returns the mock client object so tests can override return values:

        def test_generate(mock_ollama_client):
            mock_ollama_client.generate.return_value = [{"response": "Hi"}]
            ...
    """
    client = MagicMock()

    # Default: model list
    client.list.return_value = {
        "models": [
            {"name": "llama2"},
            {"name": "mistral"},
        ]
    }

    # Default: generate (streaming list)
    client.generate.return_value = [
        {"response": "Test response", "done": True, "eval_count": 1, "eval_duration": 1_000_000_000}
    ]

    # Default: chat
    client.chat.return_value = {
        "message": {"content": "Test chat response"},
        "eval_count": 1,
        "eval_duration": 1_000_000_000,
    }

    # Default: embeddings
    client.embeddings.return_value = {
        "embedding": [0.1, 0.2, 0.3],
        "total_duration": 1_000_000_000,
        "load_duration": 500_000_000,
        "prompt_eval_count": 5,
    }

    # Default: show
    client.show.return_value = {
        "name": "llama2",
        "modified_at": "2024-01-01",
        "size": 3_825_819_519,
    }

    with patch("ollama_workbench.providers.ollama_utils.get_ollama_client", return_value=client), \
         patch("ollama_workbench.providers.ollama_utils.get_available_models", return_value=["llama2", "mistral"]):
        yield client


# ---------------------------------------------------------------------------
# 3. Temporary database
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path):
    """Create a temporary SQLite database initialised with the db_init schema.

    Returns the path (str) to the database file.  The database contains the
    ``models`` table with default rows, matching the schema created by
    ``ollama_workbench.core.db_init.init_db``.

    Usage:
        def test_db_query(tmp_db):
            conn = sqlite3.connect(tmp_db)
            ...
    """
    db_file = str(tmp_path / "test_ollama_models.db")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            model_name TEXT PRIMARY KEY,
            description TEXT,
            capabilities TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    default_models = {
        "llama2": {
            "description": "Meta's Llama 2 model",
            "capabilities": "General text generation, chat, code completion, reasoning",
        },
        "mistral": {
            "description": "Mistral AI's language model",
            "capabilities": "Chat, text generation, analysis, code completion",
        },
        "codellama": {
            "description": "Code-focused variant of Llama 2",
            "capabilities": "Code completion, code explanation, debugging",
        },
    }

    for model_name, info in default_models.items():
        cursor.execute(
            "INSERT OR REPLACE INTO models (model_name, description, capabilities) VALUES (?, ?, ?)",
            (model_name, info["description"], info["capabilities"]),
        )

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_created_at ON models (created_at)")
    conn.commit()
    conn.close()

    return db_file


# ---------------------------------------------------------------------------
# 4. Mock API keys
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_api_keys():
    """Provide a dict of fake API keys and patch ``load_api_keys`` to return it.

    Usage:
        def test_provider(mock_api_keys):
            assert mock_api_keys["openai"] == "test-key-openai"
            # load_api_keys() is already patched to return this dict
    """
    keys = {
        "openai": "test-key-openai",
        "groq": "test-key-groq",
        "mistral": "test-key-mistral",
        "anthropic": "test-key-anthropic",
    }

    with patch("ollama_workbench.providers.ollama_utils.load_api_keys", return_value=keys):
        yield keys
