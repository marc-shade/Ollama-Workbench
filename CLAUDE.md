# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ollama Workbench is a comprehensive enterprise-grade platform for managing, testing, and utilizing AI models from the Ollama library and external providers (OpenAI, Groq, Mistral). Built with Streamlit, it provides a unified interface for AI model interaction, workflow orchestration, and knowledge management.

## Architecture

### Core Components
- **`main.py`**: Streamlit app entry point with navigation and page routing
- **Provider Utilities**: `ollama_utils.py`, `openai_utils.py`, `groq_utils.py`, `mistral_utils.py` - API client implementations
- **Chat Interfaces**: Multiple specialized chat implementations (`chat_interface.py`, `multimodal_chat.py`, `multimodel_chat.py`, `voice_interface.py`)
- **Workflow Systems**: `build.py`, `research.py`, `brainstorm.py`, `projects.py`, `nodes.py` - AI agent orchestration
- **TTS Server**: Flask app in `tts_server/` for text-to-speech functionality

### Key Design Patterns
- **Session Management**: Uses Streamlit session state extensively for preserving UI state
- **Provider Abstraction**: Unified interface across different AI providers with fallback handling
- **Modular Features**: Each feature is self-contained in its own module with an `_interface()` function
- **Error Resilience**: Comprehensive try-except blocks with fallback functions for missing dependencies

## Development Commands

### Initial Setup
```bash
# Clone and setup
git clone https://github.com/marc-shade/Ollama-Workbench.git
cd Ollama-Workbench
python setup_workbench.py  # Automated setup script

# Or manual setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Application
```bash
# Start with script (recommended - handles Ollama server)
./start_workbench.sh

# Or manually
streamlit run main.py

# Start TTS server separately if needed
cd tts_server && python app.py
```

### Testing
```bash
# Run all tests
python run_all_tests.py

# Run specific test suites
python feature_test.py          # Model capability tests
python model_tests.py           # Performance benchmarks
python vision_comparison.py     # Vision model comparison

# Run pytest tests
python -m pytest tests/         # All tests
python -m pytest tests/test_chat_interfaces.py -v  # Specific test with verbose
python -m pytest tests/ -k "test_session"  # Run tests matching pattern

# Run individual test functions
python -m pytest tests/test_chat_interfaces.py::TestChatInterface::test_session_management
```

### Dependency Management
```bash
# Fix dependency issues
./fix_dependencies.sh

# Key dependency versions to maintain:
# numpy==1.23.5 (compatibility)
# torch==2.0.1 (stability)
# streamlit>=1.27.2 (asyncio fixes)
# ollama>=0.4.8,!=0.7.0 (avoid breaking changes)
```

### Ollama Server Management
```bash
# Check if Ollama is running
curl -s http://localhost:11434/api/tags

# Start Ollama server
ollama serve

# Pull models
ollama pull llama3.2
ollama pull mistral

# List installed models
ollama list
```

## Common Development Tasks

### Adding a New Feature
1. Create module file: `my_feature.py`
2. Define interface function: `def my_feature_interface():`
3. Add to `SIDEBAR_SECTIONS` in `main.py:115-200`
4. Add case in `main_content()` function in `main.py:250-350`

### Working with AI Providers
```python
# Each provider has its own utility module with consistent interface:
from ollama_utils import call_ollama_api
from openai_utils import call_openai_api
from groq_utils import call_groq_api
from mistral_utils import call_mistral_api

# All follow similar pattern:
response = call_provider_api(
    model="model-name",
    messages=[{"role": "user", "content": "..."}],
    stream=True  # For streaming responses
)
```

### Session State Management
```python
# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Access and modify
st.session_state.messages.append(new_message)

# Clear session
st.session_state.clear()
```

### Error Handling Pattern
```python
try:
    from advanced_module import advanced_feature
    advanced_available = True
except ImportError as e:
    advanced_available = False
    def advanced_feature():
        st.warning("Feature not available. Install dependencies.")
```

## Configuration Files

### API Configuration
- **`api_keys.json`**: External provider API keys
- **`app_settings.json`**: Global application settings
- **`chat-settings.json`**: Chat interface preferences
- **`multimodel-chat-settings.json`**: Multi-model chat config
- **`voice-settings.json`**: Voice/TTS settings

### Prompt Templates
- **`prompts/agent_prompts.json`**: Agent type definitions
- **`prompts/identity_prompts.json`**: Model personas
- **`prompts/metacognitive_prompts.json`**: Reasoning enhancements
- **`prompts/voice_prompts.json`**: Voice personality templates

## Debugging & Monitoring

### Logs
```bash
# Application logs
tail -f app.log
tail -f test_run.log
tail -f dependency_fix.log

# TTS server logs
tail -f tts_server/logs/tts_server.log

# Test logs
tail -f tests/test_chat.log
tail -f tests/test_session.log
```

### Performance Monitoring
```python
# Built-in monitoring (server_monitoring.py)
python server_monitoring.py

# Check Ollama resource usage
curl http://localhost:11434/api/ps
```

### Common Issues & Solutions

1. **NumPy compatibility errors**
   ```bash
   pip uninstall numpy && pip install numpy==1.23.5
   ```

2. **Streamlit session issues**
   ```bash
   pip install streamlit==1.27.2
   ```

3. **Ollama connection failed**
   ```bash
   ollama serve  # Start server
   curl http://localhost:11434/api/tags  # Verify
   ```

4. **Missing json-schema-for-humans**
   ```bash
   pip install json-schema-for-humans==0.44.2
   ```

## Testing Best Practices

### Unit Test Structure
```python
# Standard test pattern in tests/
class TestFeature(unittest.TestCase):
    def setUp(self):
        # Mock Streamlit session
        self.mock_session = MagicMock()
        
    def test_functionality(self):
        # Test with mocked dependencies
        with patch('streamlit.session_state', self.mock_session):
            result = function_to_test()
            self.assertEqual(result, expected)
```

### Integration Testing
- Use `run_all_tests.py` for comprehensive testing
- Tests create temporary files in `tests/` subdirectories
- Mock external API calls to avoid rate limits

## Critical Implementation Notes

1. **Provider Fallbacks**: Always implement fallback behavior when providers are unavailable
2. **Session Persistence**: Chat sessions save to `sessions/` directory automatically
3. **Model Loading**: Check model availability before attempting operations
4. **Streaming Responses**: Use streaming for better UX with long responses
5. **Error Messages**: Provide actionable error messages with fix suggestions

## Performance Considerations

- **Lazy Loading**: Import heavy modules only when needed
- **Caching**: Use `@st.cache_data` for expensive operations
- **Streaming**: Stream LLM responses to improve perceived performance
- **Session Management**: Clear old sessions to prevent memory buildup
- **Resource Monitoring**: Monitor GPU/CPU usage with built-in tools