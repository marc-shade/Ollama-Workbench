"""
Simple Modern Chat Interface for Ollama Workbench

This module provides a simplified modern chat interface inspired by Open WebUI,
but with minimal dependencies to ensure compatibility with the existing codebase.
"""

import streamlit as st
import os
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def simple_modern_interface():
    """Simplified modern interface inspired by Open WebUI with full agent control."""
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_type" not in st.session_state:
        st.session_state.agent_type = "None"
    if "metacognitive_type" not in st.session_state:
        st.session_state.metacognitive_type = "None"
    if "voice_type" not in st.session_state:
        st.session_state.voice_type = "None"
    if "selected_corpus" not in st.session_state:
        st.session_state.selected_corpus = "None"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 4000
    if "presence_penalty" not in st.session_state:
        st.session_state.presence_penalty = 0.0
    if "frequency_penalty" not in st.session_state:
        st.session_state.frequency_penalty = 0.0
    
    # Apply some basic styling
    st.markdown("""
        <style>
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .chat-message.user {
            background-color: #2563EB20;
            border-left: 4px solid #2563EB;
        }
        .chat-message.assistant {
            background-color: #6B728020;
            border-left: 4px solid #6B7280;
        }
        .chat-message .content {
            display: flex;
            margin-top: 0.5rem;
        }
        .model-info {
            text-align: center;
            margin-bottom: 1rem;
            padding: 0.5rem;
            background-color: #f3f4f6;
            border-radius: 0.5rem;
            font-size: 0.9rem;
        }
        .stTextInput > div > div > input {
            border-radius: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header with model info
    from ollama_utils import get_available_models
    
    # Get the selected model (from whatever is available)
    available_models = get_available_models()
    if "selected_model" not in st.session_state or not st.session_state.selected_model:
        st.session_state.selected_model = available_models[0] if available_models else "llama2"
    
    # Display model info header
    model_name = st.session_state.selected_model
    st.markdown(f"<div class='model-info'>Using model: <strong>{model_name}</strong></div>", unsafe_allow_html=True)
    
    # Main layout with sidebar and content
    with st.sidebar:
        st.title("Ollama Workbench")
        
        # Agent settings in expandable sections
        with st.expander("🤖 Agent Settings", expanded=False):
            # Model selection
            st.selectbox(
                "📦 Model:",
                available_models,
                index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
                key="selected_model"
            )
            
            # Agent/Prompt types
            try:
                from prompts import get_agent_prompt, get_metacognitive_prompt, get_voice_prompt
                
                # Agent type selection
                agent_prompts = get_agent_prompt()
                agent_types = ["None"] + list(agent_prompts.keys())
                st.session_state.agent_type = st.selectbox(
                    "🧑‍🔧 Agent Type:",
                    agent_types,
                    index=agent_types.index(st.session_state.agent_type) if st.session_state.agent_type in agent_types else 0
                )
                
                # Metacognitive type selection
                metacognitive_prompts = get_metacognitive_prompt()
                metacognitive_types = ["None"] + list(metacognitive_prompts.keys())
                st.session_state.metacognitive_type = st.selectbox(
                    "🧠 Metacognitive Type:",
                    metacognitive_types,
                    index=metacognitive_types.index(st.session_state.metacognitive_type) if st.session_state.metacognitive_type in metacognitive_types else 0
                )
                
                # Voice type selection
                voice_prompts = get_voice_prompt()
                voice_types = ["None"] + list(voice_prompts.keys())
                st.session_state.voice_type = st.selectbox(
                    "🗣️ Voice Type:",
                    voice_types,
                    index=voice_types.index(st.session_state.voice_type) if st.session_state.voice_type in voice_types else 0
                )
            except ImportError:
                st.warning("Agent prompt types not available. Using default settings.")
        
        # RAG corpus selection
        with st.expander("📚 Knowledge Base", expanded=False):
            try:
                # Get available corpora
                ragtest_dir = "ragtest"
                if os.path.exists(ragtest_dir) and os.path.isdir(ragtest_dir):
                    corpus_options = ["None"] + [d for d in os.listdir(ragtest_dir) if os.path.isdir(os.path.join(ragtest_dir, d))]
                    st.session_state.selected_corpus = st.selectbox(
                        "Corpus:",
                        corpus_options,
                        index=corpus_options.index(st.session_state.selected_corpus) if st.session_state.selected_corpus in corpus_options else 0
                    )
                else:
                    st.info("No knowledge bases available. Create one in the Enhanced RAG section.")
            except Exception as e:
                st.warning(f"Knowledge base selection not available: {str(e)}")
        
        # Advanced model settings
        with st.expander("⚙️ Advanced Settings", expanded=False):
            st.slider(
                "🌡️ Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.1,
                key="temperature"
            )
            
            st.slider(
                "📊 Max Tokens",
                min_value=1000,
                max_value=16000,
                value=st.session_state.max_tokens,
                step=1000,
                key="max_tokens"
            )
            
            st.slider(
                "🚫 Presence Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=st.session_state.presence_penalty,
                step=0.1,
                key="presence_penalty"
            )
            
            st.slider(
                "🔁 Frequency Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=st.session_state.frequency_penalty,
                step=0.1,
                key="frequency_penalty"
            )
        
        # Chat management controls
        if st.button("🗑️ New Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
            
        # Save chat button
        if st.button("💾 Save Chat", use_container_width=True):
            try:
                # Create sessions directory if it doesn't exist
                os.makedirs("sessions", exist_ok=True)
                
                # Generate a timestamp and name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                chat_name = st.text_input("Enter a name for the chat:", value=f"Chat_{timestamp}")
                
                if chat_name:
                    # Save chat history to file
                    chat_data = {
                        "chat_history": st.session_state.chat_history,
                        "model": st.session_state.selected_model,
                        "agent_type": st.session_state.agent_type,
                        "metacognitive_type": st.session_state.metacognitive_type,
                        "voice_type": st.session_state.voice_type,
                        "selected_corpus": st.session_state.selected_corpus,
                        "temperature": st.session_state.temperature,
                        "max_tokens": st.session_state.max_tokens,
                        "presence_penalty": st.session_state.presence_penalty,
                        "frequency_penalty": st.session_state.frequency_penalty
                    }
                    
                    file_path = os.path.join("sessions", f"{chat_name}.json")
                    with open(file_path, "w") as f:
                        import json
                        json.dump(chat_data, f, indent=2)
                    
                    st.success(f"Chat saved as {chat_name}")
            except Exception as e:
                st.error(f"Error saving chat: {str(e)}")
    
    # Chat area
    st.title("Chat")
    
    # Display chat history
    for message in st.session_state.chat_history:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        
        with st.chat_message(role):
            st.write(content)
    
    # Chat input
    prompt = st.chat_input("Type your message here...")
    
    if prompt:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Show the user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        
        # Call the model
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Get the selected model and settings
                model = st.session_state.selected_model
                temperature = st.session_state.temperature
                max_tokens = st.session_state.max_tokens
                presence_penalty = st.session_state.presence_penalty
                frequency_penalty = st.session_state.frequency_penalty
                
                # Build the complete prompt with agent settings
                final_prompt = prompt
                
                try:
                    # Construct context for RAG if enabled
                    rag_context = ""
                    if st.session_state.selected_corpus != "None":
                        try:
                            from enhanced_corpus import GraphRAGCorpus, OllamaEmbedder
                            # Initialize embedder
                            embedder = OllamaEmbedder(model=model)
                            # Load corpus
                            corpus = GraphRAGCorpus.load(st.session_state.selected_corpus, embedder)
                            # Query corpus
                            results = corpus.query(prompt, n_results=3)
                            # Construct context
                            if results:
                                rag_context = "\nRelevant information from knowledge base:\n"
                                for result in results:
                                    rag_context += f"- {result.get('content', '')}\n"
                        except Exception as e:
                            st.warning(f"Error retrieving context: {str(e)}")
                    
                    # Construct the agent prompt
                    agent_prompt = ""
                    if st.session_state.agent_type != "None":
                        from prompts import get_agent_prompt
                        agent_prompts = get_agent_prompt()
                        if st.session_state.agent_type in agent_prompts:
                            agent_prompt += agent_prompts[st.session_state.agent_type] + "\n\n"
                    
                    # Add metacognitive prompt
                    if st.session_state.metacognitive_type != "None":
                        from prompts import get_metacognitive_prompt
                        metacog_prompts = get_metacognitive_prompt()
                        if st.session_state.metacognitive_type in metacog_prompts:
                            agent_prompt += metacog_prompts[st.session_state.metacognitive_type] + "\n\n"
                    
                    # Add voice prompt
                    if st.session_state.voice_type != "None":
                        from prompts import get_voice_prompt
                        voice_prompts = get_voice_prompt()
                        if st.session_state.voice_type in voice_prompts:
                            agent_prompt += voice_prompts[st.session_state.voice_type] + "\n\n"
                    
                    # Combine everything for the final prompt
                    if agent_prompt or rag_context:
                        # Add chat history for context if available
                        history_context = ""
                        if len(st.session_state.chat_history) > 0:
                            history_context = "Previous conversation:\n"
                            # Add last few messages for context
                            for msg in st.session_state.chat_history[-3:]:
                                role = msg["role"].capitalize()
                                content = msg["content"]
                                history_context += f"{role}: {content}\n"
                        
                        final_prompt = f"{agent_prompt}\n{history_context}\n{rag_context}\nUser: {prompt}\n\nAssistant:"
                except Exception as e:
                    st.warning(f"Error building enhanced prompt: {str(e)}")
                    # Fall back to simple prompt
                    final_prompt = prompt
                
                # Get Ollama client
                from ollama_utils import get_ollama_client, call_ollama_endpoint
                client = get_ollama_client()
                
                full_response = ""
                
                with st.spinner("Thinking..."):
                    if client:
                        # Stream response with all settings
                        for response_chunk in client.generate(
                            model=model,
                            prompt=final_prompt,
                            stream=True,
                            options={
                                "temperature": temperature,
                                "num_predict": max_tokens,
                                "presence_penalty": presence_penalty,
                                "frequency_penalty": frequency_penalty
                            }
                        ):
                            content = response_chunk["response"]
                            full_response += content
                            message_placeholder.write(full_response + "▌")
                            time.sleep(0.01)
                        
                        message_placeholder.write(full_response)
                    else:
                        # Use non-streaming fallback with all settings
                        full_response, _, _, _ = call_ollama_endpoint(
                            model=model,
                            prompt=final_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty
                        )
                        message_placeholder.write(full_response)
                
                # Add assistant message to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                message_placeholder.error(error_message)
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    simple_modern_interface()