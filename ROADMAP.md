# Ollama Workbench Update Roadmap

Based on a thorough code audit conducted February 2026. Prioritized by impact: fix broken things first, then eliminate duplication, then improve architecture, then add features.

**Status: ALL PHASES COMPLETE** (February 2026)

---

## Phase 1: Stop the Bleeding ✅ COMPLETE

> Commit `36bc02d` — Runtime bugs and dead code that actively cause failures or confusion.

### 1.1 Fix Broken Provider Calls ✅

- [x] Fixed `call_groq_api` signature to match how callers actually use it.
- [x] Rewrote `build.py` provider calls to use canonical functions from `openai_utils.py` and `groq_utils.py`.
- [x] Removed wildcard imports and shadowing local function definitions from `build.py`.
- [x] Fixed `total_tokens` type collision between Chat and Multi-Model Chat.

### 1.2 Remove Dead Code ✅

- [x] Deleted `fixed_main.py`, `integrated_main.py`, `robust_ollama_utils.py`, `simplified_ollama_utils.py`
- [x] Deleted all `fix_*.py` one-time scripts
- [x] Deleted `fixed_chat_interface.py`, `launch_fixed_chat.py`, `multimodel_fix.py`
- [x] Removed dead navigation branches and duplicate `elif` blocks in `main.py`
- [x] Deleted `modern_chat_interface.py`, `simple_modern_interface.py`, and `.bak` copies
- [x] Deleted cruft files (`MODERNIZATION_STATUS.md`, `test_report.json`, etc.)
- [x] Updated `.gitignore`

### 1.3 Fix Dependencies ✅

- [x] Removed conflicting `autogen` line from `requirements.txt`
- [x] Removed unused `streamlit-flow` packages
- [x] Removed stub `bs4` and redundant `plotly-express`
- [x] Moved dev tools to `requirements-dev.txt`
- [x] Updated stale model lists for Groq and OpenAI

### 1.4 Security Quick Fixes ✅

- [x] Removed `/openai-key` Flask endpoint that exposed plaintext API keys
- [x] Added `os.chmod(API_KEYS_FILE, 0o600)` after writing `api_keys.json`
- [x] Removed `native_host_manifest.json` writer from page load

---

## Phase 2: Consolidate Duplication ✅ COMPLETE

> Commit `950b38f` — Created single sources of truth for duplicated functions.

### 2.1 Single `load_api_keys` / `save_api_keys` ✅

- [x] Canonical implementation in `ollama_utils.py`
- [x] Replaced all 10 duplicate definitions with imports

### 2.2 Single `initialize_session_state` ✅

- [x] `session_utils.py:initialize_session_state()` is the single source of truth
- [x] Removed 8+ duplicate versions across modules

### 2.3 Single `call_ollama_endpoint` ✅

- [x] Canonical version in `ollama_utils.py`
- [x] Removed copies from `repo_docs.py` and dead files

### 2.4 Unify Model Selection Variable ✅

- [x] `selected_model` is the only variable used everywhere
- [x] Deleted `current_model` and all synchronization code

### 2.5 Dissolve `ui_elements.py` ✅

- [x] Changed `main.py` to import directly from source modules
- [x] Deleted `ui_elements.py`

### 2.6 Fix Logging ✅

- [x] Single `logging.basicConfig()` call in `main.py`
- [x] All other modules use only `logger = logging.getLogger(__name__)`
- [x] Reduced verbose session_utils logging to DEBUG

---

## Phase 3: Architecture Improvements ✅ COMPLETE

> Commit `d93e433` — Structural changes for long-term maintainability.

### 3.1 Provider Abstraction Layer ✅

- [x] Created `providers/base.py` with `BaseProvider` ABC and `ProviderResponse` dataclass
- [x] Created `providers/provider_ollama.py`, `provider_openai.py`, `provider_groq.py`, `provider_mistral.py`
- [x] Added `get_provider(name)` factory function
- [x] Migrated `multimodel_chat.py` and other callers

### 3.2 Decouple Backend From Streamlit ✅

- [x] Extracted AI workflow logic into pure functions
- [x] Kept Streamlit UI as thin wrappers

### 3.3 Fix the Chat Interface Stack ✅

- [x] Merged `fixed_chat_settings.py` fixes into `session_utils.py`
- [x] Removed monkey-patching in `enhanced_chat_interface.py`
- [x] `enhanced_chat_interface.py` is now a thin wrapper

### 3.4 Stop Network Calls on Every Rerun ✅

- [x] Cached model list with `@st.cache_data(ttl=60)`
- [x] Only refreshes on explicit user action

### 3.5 Remove the Embedded Flask Server ✅

- [x] Deleted Flask app, `run_flask()`, `send_port_to_extension()`, and `native_host_manifest.json` writer

---

## Phase 4: Test and Stabilize ✅ COMPLETE

> Commit `d144e67` — Fixed broken tests and added coverage.

### 4.1 Fix Broken Tests ✅

- [x] Deleted 12 test files testing deleted/non-existent modules
- [x] Fixed `test_chat_integration.py`, `test_ollama_utils.py`, `test_build.py` imports
- [x] Updated 1,243 mock patch paths from old flat module names to new package paths (626 failures → 261)

### 4.2 Add Coverage for Untested Modules ✅

- [x] `tests/test_tool_playground.py` — 62 tests
- [x] `tests/test_nodes_workflow.py` — 44 tests
- [x] `tests/test_collaborative_workspace_new.py` — 45 tests
- [x] `tests/test_model_management_new.py` — 39 tests
- [x] Total: 190 new tests, all passing

### 4.3 Remove Hardcoded Test Logic From Production Code ✅

- [x] Removed `if model == "llama2" and text == "This is a test text..."` special case from `chat_interface.py`
- [x] `ModelMemoryHandler.segment_text()` now dispatches purely on `model_type`

---

## Phase 5: Modernize ✅ COMPLETE

> Commit `cb2ca96` — Dynamic model lists, CI/CD, and UX improvements.

### 5.1 Update Provider Model Lists Dynamically ✅

- [x] `get_openai_models()`, `get_groq_models()`, `get_mistral_models()` fetch from APIs with `@st.cache_data(ttl=300)`
- [x] Automatic fallback to hardcoded lists when API keys are missing or calls fail
- [x] Lazy wrapper functions in `ollama_utils.py` to avoid circular imports
- [x] Updated 17 consumer files to use dynamic functions instead of static constants

### 5.2 Migrate to a Package Structure ✅

- [x] Created `ollama_workbench/` package with 8 sub-packages: `providers/`, `chat/`, `workflows/`, `knowledge/`, `models/`, `server/`, `core/`, `ui/`
- [x] Moved 128 root `.py` files into proper package structure (commit `19ec3b7`)
- [x] Updated all imports

### 5.3 Replace ChromaDB With a Maintained Alternative ✅

- [x] Evaluated: `chroma_client.py` is dead code — nothing imports it
- [x] Annotated as unused; ChromaDB dependency retained for potential future use
- [x] RAG features use other vector storage mechanisms

### 5.4 Add CI/CD ✅

- [x] GitHub Actions workflow: `ruff check` + `pytest`
- [x] `ruff.toml` configuration (Python 3.12, line-length 120)
- [x] `requirements-dev.txt` for dev dependencies

### 5.5 Streamline the Chat Experience ✅

- [x] Voice Chat nav item only shown when pyaudio dependencies are available (no more error pages)
- [x] Dynamic nav building based on available features

---

## Summary

| Phase | Status | Commit | Description |
|-------|--------|--------|-------------|
| 1. Stop the Bleeding | ✅ Complete | `36bc02d` | Fixed runtime crashes, removed ~30 dead files, fixed dependencies |
| 2. Consolidate Duplication | ✅ Complete | `950b38f` | Eliminated 10x `load_api_keys`, 8x `initialize_session_state`, 5x `call_ollama_endpoint` |
| 3. Architecture Improvements | ✅ Complete | `d93e433` | Provider abstraction, Streamlit decoupling, chat stack cleanup |
| 4. Test and Stabilize | ✅ Complete | `d144e67` | Fixed broken tests, added 190 new tests for critical modules |
| 5. Modernize | ✅ Complete | `cb2ca96` | Package structure, CI/CD, dynamic model lists, conditional nav |

All 5 phases completed February 2026.
