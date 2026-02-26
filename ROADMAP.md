# Ollama Workbench Update Roadmap

Based on a thorough code audit conducted February 2026. Prioritized by impact: fix broken things first, then eliminate duplication, then improve architecture, then add features.

---

## Phase 1: Stop the Bleeding (1-2 weeks)

Runtime bugs and dead code that actively cause failures or confusion.

### 1.1 Fix Broken Provider Calls

**`groq_utils.py` signature mismatch** - Every Groq call outside `chat_interface.py` is broken. `call_groq_api()` expects a `Groq` client object as its first argument, but `build.py:229,341,393`, `projects.py:176`, `repo_docs.py:327`, and `model_tests.py:31` all pass a model name string. These calls crash at runtime.

- [ ] Change `call_groq_api` signature to match how callers actually use it (model string, not client object), or fix all call sites.

**`build.py` uses removed OpenAI API** - `build.py:18` imports `from openai import ChatCompletion` and `build.py:99` calls `openai.ChatCompletion.create()`. This was removed in `openai>=1.0.0`. The project requires `openai>=1.99.1`. All OpenAI usage in `build.py` crashes.

- [ ] Rewrite `build.py` provider calls to use the canonical functions from `openai_utils.py` and `groq_utils.py` instead of local broken copies.
- [ ] Remove the wildcard imports (`from openai_utils import *`, `from groq_utils import *`, `from ollama_utils import *`) and the shadowing local function definitions at `build.py:48-119`.

**`total_tokens` type collision** - `chat_interface.py` writes `st.session_state.total_tokens` as an `int`. `multimodel_chat.py:82-84` overwrites it as a `dict`. Navigating between Chat and Multi-Model Chat causes `TypeError`.

- [ ] Use separate session state keys: `total_tokens` (int, for single chat) and `multimodel_total_tokens` (dict, for multi-model chat).

### 1.2 Remove Dead Code

**Dead entry points and utility duplicates:**
- [ ] Delete `fixed_main.py`, `integrated_main.py` (dead entry points)
- [ ] Delete `robust_ollama_utils.py`, `simplified_ollama_utils.py` (dead utility copies)
- [ ] Delete all 8 `fix_*.py` files (one-time scripts, none imported by active code)
- [ ] Delete `fixed_chat_interface.py` (only imported by dead `fixed_main.py`)
- [ ] Keep `fixed_chat_settings.py` (actively imported by `enhanced_chat_interface.py`)
- [ ] Delete `launch_fixed_chat.py`, `multimodel_fix.py`

**Dead navigation and duplicates in `main.py`:**
- [ ] Remove the `elif main_menu == "Multimodal Chat"` branch at `main.py:242` (no sidebar entry exists for it)
- [ ] Remove duplicate `elif` blocks for "Tool Playground" (`main.py:364`) and "Structured Output" (`main.py:366`)
- [ ] Remove or gate the CHECKPOINT debug strings shown to users in Voice Chat error page (`main.py:296-305`)

**Dead unreachable modules:**
- [ ] `modern_chat_interface.py` and `simple_modern_interface.py` are not imported by `main.py`. Either delete them or integrate them. The `.bak` copies should be deleted regardless.

**Cruft files:**
- [ ] Delete all `.bak` files
- [ ] Delete `MODERNIZATION_STATUS.md`, `test_report.json`, `test_run.log`, `modernization_validation_report.json`
- [ ] Add `*.bak*`, `test_report.json`, `test_run.log`, `modernization_validation_report.json`, `native_host_manifest.json` to `.gitignore`

### 1.3 Fix Dependencies

**`requirements.txt` conflicts:**
- [ ] Remove `autogen` line (conflicts with pinned `pyautogen==0.2.35`)
- [ ] Remove `streamlit-flow==0.1.0` and `streamlit-flow-component==1.2.9` (neither is imported anywhere)
- [ ] Remove `bs4==0.0.2` (stub package, `beautifulsoup4` already listed)
- [ ] Remove `plotly-express` (included in `plotly`)
- [ ] Move `pytest`, `pytest-html`, `flake8`, `ruff`, `radon`, `Pygments` to a separate `requirements-dev.txt`

**Stale model lists:**
- [ ] Update `groq_utils.py` `GROQ_MODELS` list (remove deprecated `llama-3.1-70b-versatile`, `gemma-7b-it`, etc.)
- [ ] Update `openai_utils.py` `OPENAI_MODELS` list (add `o3-mini`, `gpt-4.1`, remove deprecated `o1-preview`, `o1-mini`)

### 1.4 Security Quick Fixes

- [ ] Remove the `/openai-key` Flask endpoint (`main.py:417-421`) that returns the plaintext API key over unauthenticated HTTP. If the Chrome extension needs it, pass it via a different mechanism.
- [ ] Add `os.chmod(API_KEYS_FILE, 0o600)` after writing `api_keys.json` in `openai_utils.py`, `groq_utils.py`, `mistral_utils.py`, `ollama_utils.py`.
- [ ] Stop writing `native_host_manifest.json` on every page load (`main.py:124-131`). Move to a one-time setup script or gate with `if not os.path.exists()`.

---

## Phase 2: Consolidate Duplication (2-3 weeks)

The codebase has the same functions defined in 5-10 places. This phase creates single sources of truth.

### 2.1 Single `load_api_keys` / `save_api_keys`

Currently defined identically in 10 files.

- [ ] Keep the canonical implementation in `ollama_utils.py` (or extract to a new `config_utils.py`).
- [ ] Replace all other definitions with imports from the canonical location.
- [ ] Files to fix: `openai_utils.py`, `groq_utils.py`, `mistral_utils.py`, `build.py`, `research.py`, `projects.py`.

### 2.2 Single `initialize_session_state`

Currently defined in 8+ places with different defaults.

- [ ] Make `session_utils.py:initialize_session_state()` the single source of truth.
- [ ] Remove the version in `main.py:182` (replace with import).
- [ ] Remove versions in `projects.py`, `simplified_rag.py`, `enhanced_rag.py`, `multimodel_chat.py`.
- [ ] Ensure all session keys and their defaults are documented in one place.

### 2.3 Single `call_ollama_endpoint`

Currently defined in 5 files with different signatures.

- [ ] Canonical version stays in `ollama_utils.py`.
- [ ] Remove copies from `repo_docs.py`, and the dead files (`simplified_ollama_utils.py`, `integrated_main.py`, `robust_ollama_utils.py`) that should already be deleted from Phase 1.

### 2.4 Unify Model Selection Variable

Two variables (`selected_model` and `current_model`) with 50+ lines of sync logic scattered across `session_utils.py`, `main.py`, and `modern_chat_interface.py`.

- [ ] Pick one variable name (`selected_model`) and use it everywhere.
- [ ] Delete `current_model` and all synchronization code.
- [ ] Remove `synchronize_model_variables()` from `session_utils.py`.

### 2.5 Dissolve `ui_elements.py`

`ui_elements.py` is 18 wildcard imports re-exporting everything. It creates an opaque namespace.

- [ ] Change `main.py` to import directly from the actual source modules instead of through `ui_elements.py`.
- [ ] Delete `ui_elements.py`.

### 2.6 Fix Logging

`logging.basicConfig()` is called 75+ times across modules. Only the first call has any effect.

- [ ] Remove all `logging.basicConfig()` calls except one in `main.py`.
- [ ] Each module should only do `logger = logging.getLogger(__name__)`.
- [ ] Reduce `session_utils.py` INFO-level logging of every setting access to DEBUG.

---

## Phase 3: Architecture Improvements (3-4 weeks)

Structural changes that make the codebase maintainable going forward.

### 3.1 Provider Abstraction Layer

The four providers have incompatible calling conventions:
- Ollama: returns `(response, input_tokens, output_tokens, latency)` tuple
- OpenAI: returns string or `None`
- Groq: requires pre-instantiated client, returns string or `None`
- Mistral: returns string or `None`

- [ ] Create a `providers/base.py` with a common interface: `call(model, messages, temperature, max_tokens, stream) -> ProviderResponse`
- [ ] `ProviderResponse` should include: `text`, `input_tokens`, `output_tokens`, `latency`, `error`
- [ ] Wrap each provider (`providers/ollama.py`, `providers/openai.py`, etc.) to conform to this interface.
- [ ] Add a `get_provider(name)` factory function.
- [ ] Migrate callers incrementally (start with `multimodel_chat.py` which already branches on provider).

### 3.2 Decouple Backend From Streamlit

`build.py`, `research.py`, `brainstorm.py`, and `projects.py` import `streamlit.session_state` directly. This makes them untestable without mocking Streamlit.

- [ ] Extract the AI workflow logic from each module into pure functions that accept parameters and return results.
- [ ] Keep the Streamlit UI code as thin wrappers that read session state, call the pure functions, and display results.

### 3.3 Fix the Chat Interface Stack

The current layering is fragile: `enhanced_chat_interface.py` wraps `chat_interface.py` by monkey-patching `st.selectbox` at runtime.

- [ ] Merge `fixed_chat_settings.py` fixes directly into `session_utils.py` (where settings management belongs).
- [ ] Remove the monkey-patching in `enhanced_chat_interface.py:223-262`. Fix the underlying selectbox issue properly.
- [ ] Consider merging `enhanced_chat_interface.py` styling back into `chat_interface.py` since there's only one active chat interface.

### 3.4 Stop Network Calls on Every Rerun

`session_utils.py:51-66` calls `get_available_models()` (HTTP to Ollama) inside `initialize_session_state()`. Streamlit reruns the full script on every interaction.

- [ ] Cache the model list with `@st.cache_data(ttl=60)` or similar.
- [ ] Only refresh on explicit user action (e.g., a "Refresh Models" button).

### 3.5 Remove the Embedded Flask Server

The Flask server in `main.py:404-508` runs in a background thread, exposes API keys, and uses `subprocess.Popen(['npx', ...])` to communicate with a Chrome extension. This is fragile and a security risk.

- [ ] If the Chrome extension is still in use, move the Flask server to a separate process with proper authentication.
- [ ] If the Chrome extension is abandoned, delete the entire Flask app, `run_flask()`, `send_port_to_extension()`, and the `native_host_manifest.json` writer.

---

## Phase 4: Test and Stabilize (2-3 weeks)

### 4.1 Fix Broken Tests

- [ ] `tests/test_e2e_workflows.py` imports 9 classes that don't exist (`BuildWorkflow`, `ResearchWorkflow`, `BrainstormWorkflow`, `ProjectManager`, `VoiceInterface`, `FileManager`, `SessionManager`, `PerformanceTracker`, `enhanced_chat_interface` from wrong module). Either rewrite with correct imports or delete the file.
- [ ] `tests/test_build.py` mocks `build.openai.ChatCompletion.create` which no longer exists. Update after Phase 1 `build.py` fix.
- [ ] `tests/test_chat_interfaces.py` imports from `modern_chat_interface` and `simple_modern_interface` which are orphaned. Remove those test classes or update imports.

### 4.2 Add Coverage for Untested Modules

Zero test coverage exists for:
- [ ] `collaborative_workspace.py`
- [ ] `enhanced_corpus.py`
- [ ] `model_management.py`
- [ ] `persona_lab/`
- [ ] `tool_playground.py`
- [ ] `nodes.py` (existing tests import non-existent `create_node`)

Priority: `tool_playground.py` and `nodes.py` since they handle user input and model execution.

### 4.3 Remove Hardcoded Test Logic From Production Code

`chat_interface.py:93-98` contains:
```python
if model == "llama2" and text == "This is a test text for segmentation.":
    # special test handling
```

- [ ] Move test-specific logic to the test files using mocks/fixtures.

---

## Phase 5: Modernize (4-6 weeks, after Phases 1-4)

Only after the foundation is solid.

### 5.1 Update Provider Model Lists Dynamically

Instead of hardcoded `GROQ_MODELS`, `OPENAI_MODELS`, `MISTRAL_MODELS` lists that go stale:

- [ ] Fetch available models from each provider's API at startup (with caching).
- [ ] Fall back to a hardcoded list only if the API is unreachable.

### 5.2 Migrate to a Package Structure

127 Python files in the root directory is unmanageable.

- [ ] Create a package structure:
  ```
  ollama_workbench/
    providers/       # ollama, openai, groq, mistral
    chat/            # chat_interface, multimodel_chat, voice_interface
    workflows/       # build, research, brainstorm, projects, nodes
    knowledge/       # corpus, rag, chroma_client
    testing/         # model_tests, feature_test, vision_comparison
    ui/              # styles, ui components
    config/          # settings, api_keys, session management
  ```
- [ ] Update all imports. This is a large change; do it after duplication is eliminated (Phase 2) so there's less to move.

### 5.3 Replace ChromaDB With a Maintained Alternative

`chromadb==0.5.5` is pinned. ChromaDB has had significant breaking changes between versions and the project can't upgrade without testing.

- [ ] Evaluate whether the RAG features are actually used.
- [ ] If yes, consider migrating to a simpler vector store or upgrading ChromaDB with proper migration.
- [ ] If no, remove the dependency and simplify.

### 5.4 Add CI/CD

No GitHub Actions, no pre-commit hooks, no automated quality gates.

- [ ] Add a GitHub Actions workflow: `pytest`, `ruff check .`, `flake8`.
- [ ] Add a pre-commit hook for linting.
- [ ] Block merges on test failures.

### 5.5 Streamline the Chat Experience

Three chat modes (Chat, Multi-Model Chat, Voice Chat) plus Collaborative Workspace share significant code but diverge in session state handling.

- [ ] Evaluate whether Multi-Model Chat can be a mode within the main Chat (select 1 or N models).
- [ ] If Voice Chat dependencies are rarely installed, make it a plugin rather than a top-level nav item that shows an error page.

---

## Summary

| Phase | Effort | Impact |
|-------|--------|--------|
| 1. Stop the Bleeding | 1-2 weeks | Fixes runtime crashes, removes ~30 dead files, fixes dependency conflicts |
| 2. Consolidate Duplication | 2-3 weeks | Eliminates 10x `load_api_keys`, 8x `initialize_session_state`, 5x `call_ollama_endpoint` |
| 3. Architecture Improvements | 3-4 weeks | Provider abstraction, Streamlit decoupling, chat stack cleanup |
| 4. Test and Stabilize | 2-3 weeks | Fixes broken test suite, adds coverage for critical modules |
| 5. Modernize | 4-6 weeks | Package structure, CI/CD, dynamic model lists, chat UX |

Total: ~12-18 weeks for a single developer working on this full-time. Phases 1-2 deliver the most value per hour invested.
