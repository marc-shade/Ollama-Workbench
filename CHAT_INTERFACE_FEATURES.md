# Chat Interface Features Documentation

This document outlines all the features in the chat interface to ensure feature completeness and prevent regression.

## Core UI Structure

### 1. **Tabbed Interface**
- **💬 Chat Tab**: Main chat functionality
- **📜 Workspace Tab**: Document and code management

### 2. **Sidebar Components**

#### Model Selection
- Dropdown with all available models (Ollama, OpenAI, Groq, Mistral)
- Model descriptions from database
- "Apply Model Selection" button to confirm changes

#### 🤖 Chat Agent Settings (Expander)
- **Agent Type**: None, Researcher, Assistant, Teacher, Expert, Creative, Developer, Analyst
- **Metacognitive Type**: None, Reflection, Goal-Oriented, Hypothesis-Driven, Evidence-Based, Iterative, Systematic
- **Voice Type**: None, Professional, Friendly, Educational, Technical, Creative, Analytical

#### ⚙️ Advanced Settings (Expander)
- **Temperature**: 0.0 - 2.0 (default 0.7)
- **Max Tokens**: 1 - 8000 (default 4000)
- **Presence Penalty**: -2.0 - 2.0 (default 0.0)
- **Frequency Penalty**: -2.0 - 2.0 (default 0.0)
- **Knowledge Corpus**: Dropdown for RAG integration
- **Episodic Memory**: Toggle for memory features
- **Advanced Thinking**: Toggle for step-by-step reasoning
- **Instance-adaptive CoT**: Toggle for dynamic chain-of-thought

#### 📁 Saved Chats (Expander)
- **Save Current Chat** button
- List of saved chats (latest 10)
- Load chat functionality
- Delete saved chats
- **Clear Current Chat** button

#### Control Buttons
- **💾 Save All Settings**: Persists all configuration
- **⚠️ Reset All Settings**: Resets to defaults

## Chat Features

### 1. **AI-Assisted Prompt Writing**
- 🪄 Button to open modal
- Modal with:
  - Input field for user needs
  - AI-generated prompt suggestions
  - Edit capability before use
  - "Use this prompt" to apply

### 2. **Multimodal Support (Vision Models)**
- Automatic detection of vision models (vision, llava, bakllava, minicpm-v, moondream, cogvlm)
- 🖼️ Upload Image button when vision model selected
- Supported formats: PNG, JPG, JPEG, GIF, WEBP
- Image preview after upload
- Image passed to model with prompt

### 3. **Message Display**
- Chat history with role-based display
- Code block extraction and syntax highlighting
- Article block extraction
- TTS (Text-to-Speech) button for last assistant message
- Markdown rendering

### 4. **Advanced Thinking Process**
When enabled:
- Progress indicator with steps:
  - Understanding the request
  - Analyzing context
  - Formulating response
- Results integrated into prompt context

### 5. **Instance-Adaptive CoT**
When enabled:
- Dynamically selects best chain-of-thought prompt
- Adds reasoning guidance to user prompts

### 6. **RAG Integration**
- Corpus selection from available knowledge bases
- Context retrieval for relevant information
- Integration into prompt construction

### 7. **Streaming Responses**
- Real-time token streaming for Ollama models
- Cursor animation during generation
- Support for all model providers

### 8. **Auto-Save to Workspace**
- Automatic extraction of code blocks and articles
- Saves to workspace for later reference

### 9. **Error Handling**
- Graceful error messages
- Suggestion to try different models
- Error logging for debugging

## Session State Management

### Key Variables:
- `chat_history`: List of messages
- `selected_model`: Currently selected model
- `current_model`: Compatibility alias
- `agent_type`: Selected agent personality
- `metacognitive_type`: Reasoning approach
- `voice_type`: Response style
- `selected_corpus`: RAG corpus
- `temperature_slider_chat`: Temperature setting
- `max_tokens_slider_chat`: Max tokens
- `presence_penalty_slider_chat`: Presence penalty
- `frequency_penalty_slider_chat`: Frequency penalty
- `episodic_memory_enabled`: Memory toggle
- `advanced_thinking_enabled`: Thinking toggle
- `instance_adaptive_cot_enabled`: CoT toggle
- `uploaded_image`: Current uploaded image for vision models
- `show_prompt_modal`: Modal visibility state

## Integration Points

### 1. **Model Providers**
- Ollama (local models)
- OpenAI (GPT models)
- Groq (fast inference)
- Mistral (Mistral models)

### 2. **Utility Functions**
- `get_all_models()`: Retrieve available models
- `construct_agent_prompt()`: Build system prompts
- `get_graphrag_context()`: RAG retrieval
- `instance_adaptive_cot()`: CoT selection
- `advanced_thinking_step()`: Thinking process
- `text_to_speech()`: TTS generation
- `play_speech()`: Audio playback
- `save_ai_content_to_workspace()`: Workspace integration
- `count_tokens()`: Token counting
- `extract_content_blocks()`: Content parsing

### 3. **Performance Monitoring**
- Token usage tracking
- Latency measurement (placeholder)
- Metrics recording integration

## File Storage

### Settings
- Stored in `chat-settings.json`
- Includes all configuration options

### Chat Sessions
- Stored in `sessions/` directory
- JSON format with timestamp
- Includes model and full message history

## Best Practices

1. **Always check model type** before applying model-specific features
2. **Maintain session state** consistency across reruns
3. **Handle errors gracefully** with user-friendly messages
4. **Preserve user data** through proper state management
5. **Support all model providers** equally
6. **Test vision features** with actual vision models
7. **Ensure streaming works** for real-time experience

## Future Enhancements

1. Multi-turn conversation memory
2. Enhanced RAG with citations
3. Voice input support
4. Collaborative features
5. Export capabilities
6. Advanced prompt templates
7. Model performance analytics