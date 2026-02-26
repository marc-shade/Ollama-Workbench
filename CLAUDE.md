# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ollama Workbench is a Streamlit-based platform for managing, testing, and interacting with AI models from Ollama and external providers (OpenAI, Groq, Mistral). It runs as a Streamlit app on port 8501 with a companion Flask API server on a dynamic port for Chrome extension communication.

## Commands

```bash
# Run the app (recommended - handles Ollama server startup)
./start_workbench.sh

# Run manually
streamlit run main.py

# Run all tests
python run_all_tests.py

# Run pytest suite
python -m pytest tests/ -v
python -m pytest tests/test_chat_interfaces.py -v                    # specific file
python -m pytest tests/test_chat_interfaces.py::TestChatInterface -k "test_session"  # specific test

# Lint
ruff check .
flake8 .

# Setup from scratch
python setup_workbench.py
```

## Architecture

### Dual Server Design
`main.py` runs both a Streamlit app and a Flask API server (in a background thread). The Flask server (`/prompts`, `/openai-key`, `/port`) supports the Chrome extension. The Streamlit app handles all UI.

### Navigation and Routing
`main.py` defines `SIDEBAR_SECTIONS` (dict of category -> list of (label, key) tuples) and uses `streamlit_option_menu` for sidebar navigation. `create_sidebar()` sets `st.session_state.selected_test`, and `main_content()` dispatches to the appropriate module's interface function via a long if/elif chain.

### Adding a New Feature
1. Create `my_feature.py` with a `def my_feature_interface():` function
2. Add entry to `SIDEBAR_SECTIONS` in `main.py`
3. Add elif branch in `main_content()` in `main.py`

### Chat Interface Layering
There are multiple chat interface implementations with a specific inheritance chain:
- `chat_interface.py` - Base implementation with all core logic (token counting, code extraction, RAG, agent prompts, adaptive CoT)
- `enhanced_chat_interface.py` - Wraps and imports from `chat_interface.py`, adds Open WebUI-inspired styling
- `modern_chat_interface.py`, `simple_modern_interface.py` - Alternative modernized versions
- `fixed_chat_interface.py`, `fixed_chat_settings.py` - Bug fix overlays

The active chat in production is `enhanced_chat_interface()` imported in `main.py`.

### Provider Abstraction
Four provider modules with similar interfaces:
- `ollama_utils.py` - Local Ollama (primary). Key functions: `call_ollama_endpoint()`, `get_available_models()`, `load_api_keys()`
- `openai_utils.py` - `call_openai_api()`
- `groq_utils.py` - `call_groq_api()`
- `mistral_utils.py` - `call_mistral_api()`

All accept `model`, `messages`/`prompt`, and `stream` parameters. `ollama_utils.py` also handles model management (pull, remove, list) and system resource monitoring.

### Session Management
`session_utils.py` centralizes session state initialization, settings load/save (`chat-settings.json`), and chat session persistence (`sessions/` directory). Multiple interfaces import from it to stay synchronized. The key session state variables are `selected_model`, `current_model`, `chat_history`, and `messages`.

### Data Storage
- `ollama_models.db` - SQLite database initialized by `db_init.py` (model metadata)
- `sessions/` - JSON chat session files
- `prompts/` - Agent, identity, metacognitive, and voice prompt templates (JSON)
- ChromaDB - Vector storage for RAG (`chroma_client.py`)

### Workflow Modules
- `research.py` - Multi-source research automation
- `brainstorm.py` - Collaborative AI brainstorming with multiple agents
- `build.py` / `build_manager.py` - Autonomous software development agents
- `projects.py` - AI-assisted project/task management
- `nodes.py` - Visual workflow builder (CEF - drag-and-drop pipeline designer)

### TTS Server
`tts_server/app.py` is a separate Flask+CORS app (not part of the main Streamlit process) that provides text-to-speech via gTTS. Run independently with `cd tts_server && python app.py`. The main app communicates with it via `tts_utils.py`.

### Observability
Optional Opik integration for LLM tracing. The `observability` module is imported by `main.py` and `ollama_utils.py` with try/except fallbacks. When available, `ollama_utils.py` decorates LLM calls with `@trace_llm_call` for request/response tracing and performance metrics.

### Styling
`styles.py` provides centralized theming via `apply_styles()` which returns color dict and theme name. Supports light/dark mode detection via `streamlit_javascript`.

### Optional Dependencies Pattern
The codebase uses try/except imports extensively for optional features (voice, observability, tool playground). When a dependency is missing, a fallback function is defined inline that shows `st.warning()`. Check for this pattern before assuming a feature is broken.

### `fix_*.py` and `fixed_*.py` Files
These are one-off bug fix scripts and patched module copies accumulated over time. They are NOT loaded in production except `fixed_chat_settings.py` (imported by `enhanced_chat_interface.py` for settings load/save fixes). The rest can be ignored.

## Key Dependency Versions

Pinned versions that matter (from `requirements.txt`):
- `ollama>=0.5.1` - Ollama Python client
- `streamlit>=1.47.1` - Web framework
- `chromadb==0.5.5` - Vector DB for RAG
- `httpx==0.27.0` - Pinned to avoid conflicts
- Python 3.11 (venv is 3.11)

## Gotchas

- `start_workbench.sh` hardcodes the path `/Volumes/FILES/code/Ollama-Workbench` - update if the repo moves.
- The app expects Ollama running at `http://localhost:11434`. Check: `curl -s http://localhost:11434/api/tags`
- `selected_model` and `current_model` in session state must stay synchronized (see `main.py:448-454` and `session_utils.py`). Setting one without the other causes model selection bugs.
- `multimodel_chat.py` uses its own settings file (`multimodel-chat-settings.json`), separate from the main chat's `chat-settings.json`.
- The venv is Python 3.11. Do not use the system Python.
- `ollama_utils.py` uses `from ollama_utils import *` in `main.py`, which pulls many functions into the global namespace. Be aware of name collisions when adding new modules.
