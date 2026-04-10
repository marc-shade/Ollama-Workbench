import os
import json
import time
import logging
import threading
import datetime
from typing import Dict, Any, List, Optional, Union, Callable
from flask import Flask, request, jsonify, Response, stream_with_context
import ollama
from ollama_workbench.core.error_handling import WorkbenchError, handle_ollama_api_error
from ollama_workbench.core.config import get_config, CONFIG

logger = logging.getLogger(__name__)

class OpenAICompatibilityLayer:
    """
    OpenAI API compatibility layer for Ollama.
    
    This class implements a Flask app that provides OpenAI-compatible API endpoints
    which internally call the Ollama API, allowing clients designed for OpenAI to 
    work with Ollama models.
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        """
        Initialize the OpenAI compatibility layer.
        
        Args:
            host: Host to bind the server to
            port: Port to run the server on
        """
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.ollama_client = ollama.Client(host=CONFIG.get("OLLAMA_HOST", "http://localhost:11434"))
        
        # Register routes
        self.register_routes()
    
    def register_routes(self):
        """Register all API routes."""
        
        @self.app.route('/v1/models', methods=['GET'])
        def list_models():
            """List available models (OpenAI compatible endpoint)."""
            try:
                # Get models from Ollama
                models_response = self.ollama_client.list()
                # v0.4.8+: ListResponse object with .models attribute
                # older versions: dict with "models" key
                models = getattr(models_response, 'models', None)
                if models is None:
                    models = models_response.get('models', []) if isinstance(models_response, dict) else []

                # Format as OpenAI response
                result = {
                    "object": "list",
                    "data": []
                }

                for model in models:
                    # v0.4.8+: Model object with .model attribute
                    # older versions: dict with "name" key
                    if hasattr(model, 'model'):
                        model_name = model.model
                    elif isinstance(model, dict):
                        model_name = model.get('name', '')
                    else:
                        model_name = str(model)
                    result["data"].append({
                        "id": model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "ollama",
                        "permission": [],
                        "root": model_name,
                        "parent": None
                    })
                
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Error listing models: {str(e)}")
                return jsonify({
                    "error": {
                        "message": f"Error listing models: {str(e)}",
                        "type": "server_error",
                        "code": 500
                    }
                }), 500
        
        @self.app.route('/v1/chat/completions', methods=['POST'])
        def chat_completions():
            """Create a chat completion (OpenAI compatible endpoint)."""
            try:
                # Parse request data
                data = request.get_json()
                
                # Convert OpenAI format to Ollama format
                ollama_request = self._convert_openai_chat_to_ollama(data)
                
                # Check for streaming
                stream = data.get('stream', False)
                
                if stream:
                    return self._stream_chat_response(ollama_request, data)
                else:
                    # Call Ollama API
                    ollama_response = self.ollama_client.chat(**ollama_request)
                    
                    # Convert Ollama response to OpenAI format
                    openai_response = self._convert_ollama_chat_to_openai(
                        ollama_response, data.get("model", "unknown")
                    )
                    
                    return jsonify(openai_response)
            
            except Exception as e:
                logger.error(f"Error in chat completions: {str(e)}")
                return jsonify({
                    "error": {
                        "message": f"Error processing request: {str(e)}",
                        "type": "server_error",
                        "code": 500
                    }
                }), 500
        
        @self.app.route('/v1/completions', methods=['POST'])
        def completions():
            """Create a completion (OpenAI compatible endpoint)."""
            try:
                # Parse request data
                data = request.get_json()
                
                # Convert OpenAI format to Ollama format
                ollama_request = self._convert_openai_completion_to_ollama(data)
                
                # Check for streaming
                stream = data.get('stream', False)
                
                if stream:
                    return self._stream_completion_response(ollama_request, data)
                else:
                    # Call Ollama API
                    ollama_response = self.ollama_client.generate(**ollama_request)
                    
                    # Convert Ollama response to OpenAI format
                    openai_response = self._convert_ollama_completion_to_openai(
                        ollama_response, data.get("model", "unknown")
                    )
                    
                    return jsonify(openai_response)
            
            except Exception as e:
                logger.error(f"Error in completions: {str(e)}")
                return jsonify({
                    "error": {
                        "message": f"Error processing request: {str(e)}",
                        "type": "server_error",
                        "code": 500
                    }
                }), 500
        
        @self.app.route('/v1/embeddings', methods=['POST'])
        def embeddings():
            """Create embeddings (OpenAI compatible endpoint)."""
            try:
                # Parse request data
                data = request.get_json()
                
                # Get input text(s)
                input_texts = data.get('input', [])
                if isinstance(input_texts, str):
                    input_texts = [input_texts]
                
                # Get model
                model = data.get('model', 'text-embedding-ada-002')
                
                # Convert to Ollama model if needed
                ollama_model = self._map_embedding_model(model)
                
                # Process each input
                results = []
                for i, text in enumerate(input_texts):
                    # Call Ollama API
                    ollama_response = self.ollama_client.embeddings(
                        model=ollama_model,
                        prompt=text
                    )
                    
                    # Extract embedding
                    embedding = ollama_response.get('embedding', [])
                    
                    # Add to results
                    results.append({
                        "object": "embedding",
                        "embedding": embedding,
                        "index": i
                    })
                
                # Format OpenAI response
                openai_response = {
                    "object": "list",
                    "data": results,
                    "model": model,
                    "usage": {
                        "prompt_tokens": sum(len(text.split()) for text in input_texts),
                        "total_tokens": sum(len(text.split()) for text in input_texts)
                    }
                }
                
                return jsonify(openai_response)
            
            except Exception as e:
                logger.error(f"Error in embeddings: {str(e)}")
                return jsonify({
                    "error": {
                        "message": f"Error processing embeddings: {str(e)}",
                        "type": "server_error",
                        "code": 500
                    }
                }), 500
        
        @self.app.route('/v1/models/<model_id>', methods=['GET'])
        def get_model(model_id):
            """Get model information (OpenAI compatible endpoint)."""
            try:
                # Check if model exists in Ollama
                models_response = self.ollama_client.list()
                # v0.4.8+: ListResponse object with .models attribute
                models = getattr(models_response, 'models', None)
                if models is None:
                    models = models_response.get('models', []) if isinstance(models_response, dict) else []

                # v0.4.8+: Model object with .model attribute (NO .name)
                model = next(
                    (m for m in models
                     if (getattr(m, 'model', None) or (m.get('name') if isinstance(m, dict) else None)) == model_id),
                    None
                )
                
                if not model:
                    return jsonify({
                        "error": {
                            "message": f"Model '{model_id}' not found",
                            "type": "not_found",
                            "code": 404
                        }
                    }), 404
                
                # Format as OpenAI response
                result = {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ollama",
                    "permission": [],
                    "root": model_id,
                    "parent": None
                }
                
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Error getting model: {str(e)}")
                return jsonify({
                    "error": {
                        "message": f"Error getting model: {str(e)}",
                        "type": "server_error",
                        "code": 500
                    }
                }), 500
    
    def _convert_openai_chat_to_ollama(self, openai_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenAI chat API request to Ollama format.
        
        Args:
            openai_request: OpenAI API request
            
        Returns:
            Dict[str, Any]: Ollama API request
        """
        # Extract basic parameters
        model = openai_request.get('model', 'gpt-3.5-turbo')
        messages = openai_request.get('messages', [])
        
        # Map model name if needed
        ollama_model = self._map_model_name(model)
        
        # Convert messages (should already be in the right format)
        ollama_messages = messages
        
        # Extract generation parameters
        temperature = openai_request.get('temperature', 0.7)
        max_tokens = openai_request.get('max_tokens', 4096)
        top_p = openai_request.get('top_p', 1.0)
        frequency_penalty = openai_request.get('frequency_penalty', 0.0)
        presence_penalty = openai_request.get('presence_penalty', 0.0)
        
        # Construct Ollama request
        ollama_request = {
            "model": ollama_model,
            "messages": ollama_messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty
            }
        }
        
        return ollama_request
    
    def _convert_openai_completion_to_ollama(self, openai_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenAI completion API request to Ollama format.
        
        Args:
            openai_request: OpenAI API request
            
        Returns:
            Dict[str, Any]: Ollama API request
        """
        # Extract basic parameters
        model = openai_request.get('model', 'text-davinci-003')
        prompt = openai_request.get('prompt', '')
        
        # Map model name if needed
        ollama_model = self._map_model_name(model)
        
        # Extract generation parameters
        temperature = openai_request.get('temperature', 0.7)
        max_tokens = openai_request.get('max_tokens', 4096)
        top_p = openai_request.get('top_p', 1.0)
        frequency_penalty = openai_request.get('frequency_penalty', 0.0)
        presence_penalty = openai_request.get('presence_penalty', 0.0)
        
        # Construct Ollama request
        ollama_request = {
            "model": ollama_model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty
            }
        }
        
        return ollama_request
    
    def _convert_ollama_chat_to_openai(self, ollama_response: Dict[str, Any], model: str) -> Dict[str, Any]:
        """
        Convert Ollama chat API response to OpenAI format.
        
        Args:
            ollama_response: Ollama API response
            model: Model name
            
        Returns:
            Dict[str, Any]: OpenAI API response
        """
        # Extract response content
        message = ollama_response.get('message', {})
        content = message.get('content', '')
        
        # Calculate tokens (approximate)
        prompt_tokens = sum(len(msg.get('content', '').split()) for msg in ollama_response.get('prompt', []))
        completion_tokens = len(content.split())
        
        # Construct OpenAI response
        openai_response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
        
        return openai_response
    
    def _convert_ollama_completion_to_openai(self, ollama_response: Dict[str, Any], model: str) -> Dict[str, Any]:
        """
        Convert Ollama completion API response to OpenAI format.
        
        Args:
            ollama_response: Ollama API response
            model: Model name
            
        Returns:
            Dict[str, Any]: OpenAI API response
        """
        # Extract response content
        response = ollama_response.get('response', '')
        
        # Calculate tokens (approximate)
        prompt_tokens = len(ollama_response.get('prompt', '').split())
        completion_tokens = len(response.split())
        
        # Construct OpenAI response
        openai_response = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "text": response,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
        
        return openai_response
    
    def _stream_chat_response(self, ollama_request: Dict[str, Any], openai_request: Dict[str, Any]):
        """
        Stream chat completion response (OpenAI compatible).
        
        Args:
            ollama_request: Ollama API request
            openai_request: OpenAI API request
            
        Returns:
            Response: Streaming response
        """
        def generate():
            try:
                # Need to add stream=True
                ollama_request['stream'] = True
                
                # Get response stream from Ollama
                response_stream = self.ollama_client.chat(**ollama_request)
                
                # Generate ID once for all chunks
                response_id = f"chatcmpl-{int(time.time())}"
                model = openai_request.get('model', 'unknown')
                created = int(time.time())
                
                # Stream each chunk
                for chunk in response_stream:
                    content = chunk.get('message', {}).get('content', '')
                    
                    # Create OpenAI-compatible chunk
                    openai_chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": content
                                },
                                "finish_reason": None
                            }
                        ]
                    }
                    
                    # Send the chunk
                    yield f"data: {json.dumps(openai_chunk)}\n\n"
                
                # Send final chunk
                final_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            except Exception as e:
                logger.error(f"Error in streaming chat response: {str(e)}")
                error_chunk = {
                    "error": {
                        "message": f"Error processing request: {str(e)}",
                        "type": "server_error",
                        "code": 500
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
        
        # Return streaming response
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream'
        )
    
    def _stream_completion_response(self, ollama_request: Dict[str, Any], openai_request: Dict[str, Any]):
        """
        Stream completion response (OpenAI compatible).
        
        Args:
            ollama_request: Ollama API request
            openai_request: OpenAI API request
            
        Returns:
            Response: Streaming response
        """
        def generate():
            try:
                # Need to add stream=True
                ollama_request['stream'] = True
                
                # Get response stream from Ollama
                response_stream = self.ollama_client.generate(**ollama_request)
                
                # Generate ID once for all chunks
                response_id = f"cmpl-{int(time.time())}"
                model = openai_request.get('model', 'unknown')
                created = int(time.time())
                
                # Stream each chunk
                for chunk in response_stream:
                    content = chunk.get('response', '')
                    
                    # Create OpenAI-compatible chunk
                    openai_chunk = {
                        "id": response_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "text": content,
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None
                            }
                        ]
                    }
                    
                    # Send the chunk
                    yield f"data: {json.dumps(openai_chunk)}\n\n"
                
                # Send final chunk
                final_chunk = {
                    "id": response_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "text": "",
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "stop"
                        }
                    ]
                }
                
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            except Exception as e:
                logger.error(f"Error in streaming completion response: {str(e)}")
                error_chunk = {
                    "error": {
                        "message": f"Error processing request: {str(e)}",
                        "type": "server_error",
                        "code": 500
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
        
        # Return streaming response
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream'
        )
    
    def _map_model_name(self, openai_model: str) -> str:
        """
        Map OpenAI model name to Ollama model name.
        
        Args:
            openai_model: OpenAI model name
            
        Returns:
            str: Ollama model name
        """
        # Model mapping (can be expanded)
        model_mapping = {
            "gpt-3.5-turbo": "llama3",
            "gpt-3.5-turbo-16k": "llama3",
            "gpt-4": "llama3",
            "gpt-4-32k": "llama3",
            "gpt-4-1106-preview": "llama3",
            "gpt-4-0125-preview": "llama3",
            "text-davinci-003": "llama3",
            "text-davinci-002": "llama3"
        }
        
        # Return mapped model or original if not in mapping
        return model_mapping.get(openai_model, openai_model)
    
    def _map_embedding_model(self, openai_model: str) -> str:
        """
        Map OpenAI embedding model to Ollama model.
        
        Args:
            openai_model: OpenAI model name
            
        Returns:
            str: Ollama model name
        """
        # Embedding model mapping
        embedding_mapping = {
            "text-embedding-ada-002": "nomic-embed-text",
            "text-embedding-3-small": "nomic-embed-text",
            "text-embedding-3-large": "nomic-embed-text"
        }
        
        # Return mapped model or default to nomic-embed-text
        return embedding_mapping.get(openai_model, "nomic-embed-text")
    
    def run(self, debug: bool = False):
        """
        Run the OpenAI compatibility server.
        
        Args:
            debug: Whether to run in debug mode
        """
        self.app.run(host=self.host, port=self.port, debug=debug)

def start_openai_compatibility_server():
    """Start the OpenAI compatibility server in a separate thread."""
    config = get_config()
    host = config.get("OPENAI_COMPAT_HOST", "127.0.0.1")
    port = int(config.get("OPENAI_COMPAT_PORT", 8000))
    
    compatibility_layer = OpenAICompatibilityLayer(host=host, port=port)
    
    # Start in a separate thread
    server_thread = threading.Thread(target=compatibility_layer.run)
    server_thread.daemon = True
    server_thread.start()
    
    logger.info(f"OpenAI compatibility server started on {host}:{port}")
    return compatibility_layer

def openai_compatibility_ui():
    """Streamlit UI for OpenAI compatibility server."""
    import streamlit as st
    
    st.title("🔄 OpenAI API Compatibility")
    st.write("This feature allows you to use OpenAI-compatible clients with Ollama models.")
    
    # Configuration
    with st.expander("Server Configuration", expanded=True):
        config = get_config()
        
        # Host/port settings
        col1, col2 = st.columns(2)
        
        with col1:
            host = st.text_input(
                "Host",
                value=config.get("OPENAI_COMPAT_HOST", "127.0.0.1"),
                help="The host to bind the server to (use 127.0.0.1 for local-only access)"
            )
        
        with col2:
            port = st.number_input(
                "Port",
                value=int(config.get("OPENAI_COMPAT_PORT", 8000)),
                min_value=1,
                max_value=65535,
                help="The port to run the server on"
            )
        
        # Enable/disable
        enable_server = st.checkbox(
            "Enable OpenAI Compatibility Server",
            value=config.get("ENABLE_OPENAI_COMPAT", True),
            help="Enable the OpenAI compatibility server"
        )
        
        # Save button
        if st.button("Save Configuration"):
            from ollama_workbench.core.config import update_config
            
            update_config({
                "OPENAI_COMPAT_HOST": host,
                "OPENAI_COMPAT_PORT": port,
                "ENABLE_OPENAI_COMPAT": enable_server
            })
            
            st.success("Configuration saved!")
    
    # Model mapping
    with st.expander("Model Mapping"):
        st.write("This shows how OpenAI model names are mapped to Ollama models:")
        
        # Default mappings
        mappings = {
            "gpt-3.5-turbo": "llama3",
            "gpt-3.5-turbo-16k": "llama3",
            "gpt-4": "llama3",
            "gpt-4-32k": "llama3",
            "gpt-4-1106-preview": "llama3",
            "gpt-4-0125-preview": "llama3",
            "text-davinci-003": "llama3",
            "text-davinci-002": "llama3",
            "text-embedding-ada-002": "nomic-embed-text"
        }
        
        # Create a simple table
        data = {"OpenAI Model": [], "Ollama Model": []}
        
        for openai_model, ollama_model in mappings.items():
            data["OpenAI Model"].append(openai_model)
            data["Ollama Model"].append(ollama_model)
        
        # Show the table
        import pandas as pd
        st.table(pd.DataFrame(data))
        
        st.info("These mappings are used when clients send requests with OpenAI model names.")
    
    # Usage instructions
    with st.expander("Usage Instructions"):
        st.markdown("""
        ### How to Use
        
        1. **Start the Server**: Enable the server in the configuration above and click "Save Configuration".
        
        2. **Connect with OpenAI Clients**: Configure your OpenAI clients to use the following endpoints:
            - API Base URL: `http://localhost:8000/v1` (adjust host/port as needed)
            - API Key: (any value, not checked)
        
        3. **Supported Endpoints**:
            - `/v1/models` - List available models
            - `/v1/chat/completions` - Create chat completions
            - `/v1/completions` - Create text completions
            - `/v1/embeddings` - Create embeddings
        
        ### Example (Python)
        
        ```python
        from openai import OpenAI
        
        # Connect to Ollama through compatibility layer
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="dummy-key"  # Not checked
        )
        
        # Chat completion
        response = client.chat.completions.create(
            model="llama3",  # Or use "gpt-3.5-turbo" which maps to llama3
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ]
        )
        
        print(response.choices[0].message.content)
        ```
        
        ### Limitations
        
        - Not all OpenAI API features are supported
        - Model capabilities may differ from actual OpenAI models
        - Token counting is approximate
        """)
    
    # Server status
    st.subheader("Server Status")
    
    # Check if server is running
    import socket
    
    def check_server_running(host, port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect((host, port))
                return True
        except Exception:
            return False
    
    server_running = check_server_running(
        "localhost" if host == "0.0.0.0" else host,
        port
    )
    
    if server_running:
        st.success(f"✅ OpenAI compatibility server is running on {host}:{port}")
        
        # Add a stop button
        if st.button("Stop Server"):
            st.warning("This functionality is not implemented yet. Please restart the Workbench to stop the server.")
    else:
        st.error("❌ OpenAI compatibility server is not running")
        
        # Add a start button
        if st.button("Start Server"):
            if enable_server:
                # Start the server
                start_openai_compatibility_server()
                st.success("Server started! Refresh the page to see updated status.")
            else:
                st.warning("Please enable the server in the configuration first.")

# Start the server on import if enabled
if CONFIG.get("ENABLE_OPENAI_COMPAT", True):
    try:
        start_openai_compatibility_server()
    except Exception as e:
        logger.error(f"Error starting OpenAI compatibility server: {str(e)}")