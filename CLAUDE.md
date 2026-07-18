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
python scripts/run_all_tests.py

# Run pytest suite
python -m pytest tests/ -v
python -m pytest tests/test_chat_interfaces.py -v                    # specific file
python -m pytest tests/test_chat_interfaces.py::TestChatInterface -k "test_session"  # specific test

# Verify all package imports work
python scripts/test_imports.py

# Lint
ruff check .
flake8 .

# Setup from scratch
python scripts/setup_workbench.py
```

## Package Structure

The codebase is organized into the `ollama_workbench` package with 8 sub-packages. `main.py` is the only Python file at the root — it's the Streamlit entry point.

```
main.py                          # Streamlit entry point + Flask API
ollama_workbench/
    providers/                   # AI provider API clients
        ollama_utils.py          #   Ollama (primary) - model mgmt, embeddings, monitoring
        openai_utils.py          #   OpenAI
        groq_utils.py            #   Groq
        mistral_utils.py         #   Mistral
        external_providers.py    #   Provider configuration UI
    chat/                        # Chat interfaces
        chat_interface.py        #   Base chat (token counting, RAG, agent prompts, CoT)
        enhanced_chat_interface.py  # Wraps chat_interface with Open WebUI styling
        multimodel_chat.py       #   Multi-model simultaneous chat
        multimodal_chat.py       #   Vision/image chat
        voice_interface.py       #   Voice chat (requires pyaudio)
        persona_chat.py          #   Persona group chat
        collaborative_workspace.py  # Collaborative workspace
        canvas.py                #   Document canvas for workspace
        voice_utils.py           #   Voice/TTS utilities
        tts_utils.py             #   Text-to-speech helpers
    workflows/                   # AI workflow orchestration
        build.py                 #   Autonomous software development
        research.py              #   Multi-source research automation
        brainstorm.py            #   AI brainstorming with multiple agents
        projects.py              #   AI-assisted project/task management
        nodes.py                 #   CEF visual workflow builder
        agents.py                #   Agent definitions
        info_brainstorm.py       #   Brainstorm info helpers
    knowledge/                   # Knowledge management & RAG
        simplified_rag.py        #   Enhanced RAG interface (active)
        enhanced_corpus.py       #   GraphRAG corpus management
        repo_docs.py             #   Repository analysis
        web_to_corpus.py         #   Web crawling to corpus
        search_libraries.py      #   Multi-engine web search
    models/                      # Model management & testing
        model_comparison.py      #   Response quality comparison
        model_tests.py           #   Performance benchmarks
        feature_test.py          #   Feature capability tests
        vision_comparison.py     #   Vision model comparison
        local_models.py          #   List/manage local models
        pull_model.py            #   Download new models
        show_model.py            #   Model details viewer
        remove_model.py          #   Model removal
        update_models.py         #   Batch model updates
        model_management.py      #   Management dashboard
        model_capabilities.py    #   Capability testing UI
        model_capability_registry.py  # Capability detection
        model_onboarding.py      #   Model onboarding tests
        test_visualization.py    #   Test result visualization
    server/                      # Server management
        server_configuration.py  #   Ollama server config
        server_monitoring.py     #   Resource monitoring
        performance_metrics.py   #   Performance tracking
        openai_compatibility.py  #   OpenAI-compatible API
    core/                        # Core infrastructure
        config.py                #   Application configuration
        session_utils.py         #   Session state management
        db_init.py               #   Database initialization
        error_handling.py        #   Error handling utilities
    ui/                          # UI components
        styles.py                #   Theming (apply_styles -> colors, theme)
        prompts.py               #   Prompt template management
        file_management.py       #   File browser/editor
        structured_output.py     #   JSON schema output UI
        tool_playground.py       #   Tool calling playground
        contextual_response.py   #   Contextual response testing
        welcome.py               #   Help/welcome page
        global_vrm_loader.py     #   VRM model loader
tests/                           # All test files
scripts/                         # Standalone utility scripts
persona_lab/                     # Persona generation lab
observability/                   # Optional Opik integration
tts_server/                      # Standalone TTS Flask server
prompts/                         # JSON prompt templates
```

## Architecture

### Navigation and Routing
`main.py` defines `SIDEBAR_SECTIONS` (dict of category -> list of (label, key) tuples) and uses `streamlit_option_menu` for sidebar navigation. `create_sidebar()` sets `st.session_state.selected_test`, and `main_content()` dispatches to the appropriate module's interface function.

### Adding a New Feature
1. Create `ollama_workbench/<subpackage>/my_feature.py` with `def my_feature_interface():`
2. Add import in `main.py`: `from ollama_workbench.<subpackage>.my_feature import my_feature_interface`
3. Add entry to `SIDEBAR_SECTIONS` in `main.py`
4. Add elif branch in `main_content()` in `main.py`

### Import Conventions
- Cross-package: `from ollama_workbench.providers.ollama_utils import call_ollama_endpoint`
- Same package: `from .ollama_utils import load_api_keys` (relative imports)
- main.py: Always absolute `from ollama_workbench.xxx.yyy import zzz`

### Provider Abstraction
Four provider modules in `ollama_workbench/providers/` with similar interfaces:
- `ollama_utils.py` - Local Ollama (primary). Key: `call_ollama_endpoint()`, `get_available_models()`, `load_api_keys()`
- `openai_utils.py` - `call_openai_api()`
- `groq_utils.py` - `call_groq_api()`
- `mistral_utils.py` - `call_mistral_api()`

Circular dependency: `ollama_utils` imports from the other three (for model lists and API calls), and they import `load_api_keys`/`save_api_keys` from `ollama_utils`. This works because `load_api_keys` is defined BEFORE the cross-provider imports in `ollama_utils.py`. Do not reorder.

### Session Management
`core/session_utils.py` centralizes session state initialization and settings persistence. The canonical model variable is `selected_model` (no `current_model`). Settings files: `chat-settings.json`, `multimodel-chat-settings.json`.

### Data Storage
- `ollama_models.db` - SQLite (initialized by `core/db_init.py`)
- `sessions/` - JSON chat session files
- `prompts/` - JSON prompt templates (agent, identity, metacognitive, voice)
- ChromaDB - Vector storage for RAG

### TTS Server
`tts_server/app.py` is a separate Flask+CORS app providing text-to-speech via gTTS. Run independently: `cd tts_server && python app.py`.

### Optional Dependencies Pattern
The codebase uses try/except imports for optional features (voice, observability, MCP tools). Missing deps show `st.warning()` via inline fallback functions.

## Key Dependency Versions

- `ollama>=0.5.1` - Ollama Python client
- `streamlit>=1.47.1` - Web framework
- `chromadb==0.5.5` - Vector DB for RAG
- `httpx==0.27.0` - Pinned to avoid conflicts
- Python 3.11 (venv; built from the `ollamaworkbench` miniconda env)

## Gotchas

- `start_workbench.sh` derives the repo path from its own location (the drive has mounted as both `/Volumes/FILES` and `/Volumes/FILES 1`).
- The venv's `bin/` entry-point scripts (pip, streamlit, pytest, ...) have shebangs baked to the old `/Volumes/FILES/...` mount path and a shebang cannot contain a space, so they break when the drive mounts as `FILES 1`. Always invoke via `venv/bin/python -m <tool>`.
- Ollama must be running at `http://localhost:11434`. Check: `curl -s http://localhost:11434/api/tags`
- `multimodel_chat.py` uses its own settings file (`multimodel-chat-settings.json`), separate from `chat-settings.json`.
- The venv is Python 3.11. Do not use the system Python.
- `main.py` uses `from ollama_workbench.providers.ollama_utils import *` which pulls many functions into scope. Be aware of name collisions.
