# multimodal_chat.py

import streamlit as st
import pandas as pd
import time
import json
import re
import io
import base64
import subprocess
from datetime import datetime
import ollama
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from ollama_workbench.providers.ollama_utils import get_available_models, call_ollama_endpoint, get_ollama_client
from ollama_workbench.models.model_capability_registry import filter_models_by_capability, is_vision_capable
import logging

logger = logging.getLogger(__name__)

def format_chat_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Format a chat message for display, handling multimodal content"""
    formatted = {"role": message["role"]}
    
    if isinstance(message.get("content"), str):
        # Simple text content
        formatted["content"] = message["content"]
    elif isinstance(message.get("content"), list):
        # Multimodal content (text and images)
        text_parts = []
        image_parts = []
        
        for part in message["content"]:
            if isinstance(part, str) or part.get("type") == "text":
                if isinstance(part, str):
                    text_parts.append(part)
                else:
                    text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                # Handle image content
                image_parts.append(part)
        
        formatted["content"] = "\n\n".join(text_parts)
        formatted["images"] = image_parts
    else:
        # Handle unexpected format
        formatted["content"] = str(message.get("content", ""))
    
    return formatted

def image_to_base64(image_file) -> str:
    """Convert an image file to base64 encoding"""
    image_file.seek(0)
    img_bytes = image_file.read()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    return encoded

def prepare_message_with_image(text: str, image_file) -> List[Dict[str, Any]]:
    """Prepare a message that includes both text and an image"""
    # Reset file pointer to beginning
    image_file.seek(0)
    
    # Get image mime type
    mime_type = image_file.type
    if not mime_type:
        # Try to guess mime type
        try:
            img = Image.open(image_file)
            mime_type = f"image/{img.format.lower()}"
        except:
            mime_type = "image/jpeg"  # Default to JPEG
    
    # Encode the image
    base64_image = image_to_base64(image_file)
    
    # Create message with text and image
    return [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': text
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ]
        }
    ]

def multimodal_chat_interface():
    """Multimodal chat interface with enhanced image handling"""
    st.title("🖼️ Multimodal Chat")
    st.write("Chat with models that support images and text input.")
    
    # Initialize session state
    if "multimodal_chat_history" not in st.session_state:
        st.session_state.multimodal_chat_history = []
    if "multimodal_selected_model" not in st.session_state:
        st.session_state.multimodal_selected_model = None
    if "multimodal_provider" not in st.session_state:
        st.session_state.multimodal_provider = "Ollama"
    
    # Sidebar for model selection and settings
    with st.sidebar:
        with st.expander("🤖 Model Settings", expanded=True):
            # First, select the provider
            provider_options = ["Ollama", "OpenAI", "Groq", "Mistral"]
            selected_provider = st.selectbox(
                "Select Provider:",
                provider_options,
                index=0,
                key="multimodal_provider"
            )
            
            # Log provider selection for debugging
            logger.info(f"Selected provider: {selected_provider}")
            
            # Get models based on provider
            if selected_provider == "Ollama":
                # Get available models with multiple fallback approaches
                try:
                    # First try the API approach
                    available_models = get_available_models()
                    logging.info(f"Got {len(available_models)} models from API")
                    
                    # If API returns empty but Ollama is running, try CLI fallback
                    if not available_models:
                        try:
                            # Run ollama list command to get models
                            import subprocess
                            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
                            lines = result.stdout.strip().split('\n')[1:]  # Skip header line
                            available_models = []
                            for line in lines:
                                if line.strip():
                                    # The model name is the first column, which may contain colons and other special chars
                                    parts = line.split()
                                    if parts:  # Make sure line has content
                                        model_name = parts[0]
                                        available_models.append(model_name)
                            logging.info(f"Got {len(available_models)} models from CLI fallback")
                        except Exception as cli_ex:
                            logging.error(f"Error in CLI fallback: {cli_ex}")
                            # Continue with empty list
                except Exception as api_error:
                    logging.error(f"API error fetching models: {api_error}")
                    # Try CLI approach as fallback
                    try:
                        # Run ollama list command to get models
                        import subprocess
                        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
                        lines = result.stdout.strip().split('\n')[1:]  # Skip header line
                        available_models = []
                        for line in lines:
                            if line.strip():
                                # The model name is the first column, which may contain colons and other special chars
                                parts = line.split()
                                if parts:  # Make sure line has content
                                    model_name = parts[0]
                                    available_models.append(model_name)
                        logging.info(f"Got {len(available_models)} models from CLI direct")
                    except Exception as cli_ex:
                        logging.error(f"Error in direct CLI: {cli_ex}")
                        available_models = []
                
                if not available_models:
                    st.warning("No Ollama models found. Please pull some models first.")
                    model_options = []
                else:
                    # Filter models using our capability registry
                    vision_models = filter_models_by_capability(available_models, "vision")
                    
                    # Only show models that officially support vision
                    if vision_models:
                        st.info("Showing only models with vision capabilities")
                        
                        # Add "(vision)" suffix to help users identify official vision models
                        model_options = []
                        for model in vision_models:
                            model_options.append(f"{model} (vision)")
                    else:
                        # If no vision models found, show all models with a warning
                        st.warning("No models with official vision capabilities detected. Any model can be tried, but may not work with images.")
                        model_options = available_models
            
            elif selected_provider == "OpenAI":
                # Import OpenAI models
                from ollama_workbench.providers.openai_utils import OPENAI_MODELS
                
                # Filter to only vision-capable models
                vision_models = ["gpt-4-vision-preview", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
                model_options = [model for model in OPENAI_MODELS if model in vision_models]
                
                if not model_options:
                    st.warning("No OpenAI vision models available.")
                    
                # API key input
                openai_api_key = st.text_input("OpenAI API Key:", type="password", 
                                              key="multimodal_openai_api_key")
                if not openai_api_key:
                    st.warning("Please enter your OpenAI API key.")
            
            elif selected_provider == "Groq":
                # Import Groq models
                from ollama_workbench.providers.groq_utils import GROQ_MODELS
                
                # Currently, Groq doesn't support vision models, but we'll prepare for when they do
                st.warning("Groq currently doesn't support multimodal/vision models. Please use another provider for image processing.")
                model_options = GROQ_MODELS
                
                # API key input
                groq_api_key = st.text_input("Groq API Key:", type="password", 
                                            key="multimodal_groq_api_key")
                if not groq_api_key:
                    st.warning("Please enter your Groq API key.")
                
            elif selected_provider == "Mistral":
                # Import Mistral models
                from ollama_workbench.providers.mistral_utils import MISTRAL_MODELS
                
                # Filter to vision-capable models when available
                # For now, just use all models but warn
                st.warning("Only Mistral Large 2 (and newer) supports vision. Other models will not process images.")
                
                # Model options with vision label
                model_options = []
                for model in MISTRAL_MODELS:
                    if "large-2" in model.lower():
                        model_options.append(f"{model} (vision)")
                    else:
                        model_options.append(model)
                
                # API key input
                mistral_api_key = st.text_input("Mistral API Key:", type="password", 
                                              key="multimodal_mistral_api_key")
                if not mistral_api_key:
                    st.warning("Please enter your Mistral API key.")
            
            # Model selection
            if not model_options:
                if selected_provider == "Ollama":
                    st.warning("No models found. Please pull some models first.")
                else:
                    st.warning(f"No {selected_provider} models available.")
                selected_model_display = None
                logger.warning("No models found for the selected provider.")
            else:
                selected_model_display = st.selectbox(
                    "Select Model:",
                    model_options,
                    index=0,
                    key="multimodal_model_selector"
                )
            
            # Store model in session state (provider is already stored via the widget key)
            
            # Remove the "(vision)" suffix if present
            if selected_model_display:
                st.session_state.multimodal_selected_model = selected_model_display.split(" (vision)")[0]
            else:
                st.session_state.multimodal_selected_model = None
            
            # Model parameters
            st.session_state.temperature_slider = st.slider(
                "Temperature:",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key="multimodal_temperature"
            )
            
            st.session_state.max_tokens_slider = st.slider(
                "Max Tokens:",
                min_value=1000,
                max_value=32000,
                value=4000,
                step=1000,
                key="multimodal_max_tokens"
            )
        
        # Button to clear chat history
        if st.button("🗑️ Clear Chat", key="multimodal_clear_chat"):
            st.session_state.multimodal_chat_history = []
            st.rerun()
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.multimodal_chat_history:
            with st.chat_message(message["role"]):
                # Display text content
                if "content" in message and message["content"]:
                    st.markdown(message["content"])
                
                # Display images if present
                if "images" in message and message["images"]:
                    for img in message["images"]:
                        if "image_url" in img and "url" in img["image_url"]:
                            url = img["image_url"]["url"]
                            if url.startswith("data:"):
                                # Handle base64 encoded images
                                mime_type, b64_data = url.split(',', 1)
                                mime_type = mime_type.split(':')[1].split(';')[0]
                                image_data = base64.b64decode(b64_data)
                                st.image(image_data, caption="Uploaded Image", use_column_width=True)
                            else:
                                # Handle normal URLs
                                st.image(url, caption="Image", use_column_width=True)
    
    # Input area for text and image
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_area("Type your message:", key="multimodal_text_input", height=100)
        
        with col2:
            uploaded_file = st.file_uploader("Upload an image (optional):", 
                                           type=["jpg", "jpeg", "png", "webp"],
                                           key="multimodal_image_input")
            
            if uploaded_file:
                st.image(uploaded_file, caption="Preview", width=200)
        
        # Send button
        if st.button("Send", key="multimodal_send_button") or user_input and st.session_state.get("enter_to_send", False):
            if not user_input and not uploaded_file:
                st.warning("Please enter a message or upload an image.")
                return
                
            if not st.session_state.multimodal_selected_model:
                st.warning("Please select a model first.")
                return
            
            # Process and display user message
            user_message = {"role": "user", "content": user_input}
            if uploaded_file:
                # Preview the uploaded image
                user_message["images"] = [
                    {
                        "type": "image_url",
                        "image_url": {"url": "uploaded_image"}
                    }
                ]
                with st.chat_message("user"):
                    st.markdown(user_input)
                    st.image(uploaded_file, caption="Uploaded Image", width=300)
            else:
                with st.chat_message("user"):
                    st.markdown(user_input)
            
            # Add user message to history
            st.session_state.multimodal_chat_history.append(user_message)
            
            # Prepare API request
            # Already imported get_ollama_client at the top of the file
            client = get_ollama_client()
            
            # We'll continue even if client is None - we have fallbacks
            if not client:
                logger.warning("No ollama client available, will use fallback methods")
                # Will continue using fallbacks
                
            messages = []
            
            # Check if model officially supports multimodal
            selected_model = st.session_state.multimodal_selected_model
            if not selected_model:
                st.error("No model selected. Please select a model first.")
                return
                
            # Use the registry to check capabilities
            is_vision_model = is_vision_capable(selected_model)
            
            if uploaded_file and not is_vision_model:
                st.warning(f"Model '{selected_model}' does not officially support images. If you experience errors, try using a model with vision capabilities.")
            
            # Format previous messages for context
            for msg in st.session_state.multimodal_chat_history[:-1]:  # Exclude the latest message which we'll handle specially
                if msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                else:
                    messages.append({"role": "assistant", "content": msg["content"]})
            
            # Add the current message with image if present
            if uploaded_file:
                try:
                    # Reset file to beginning
                    uploaded_file.seek(0)
                    
                    # Create message with both text and image
                    prompt_text = user_input if user_input else "Describe this image:"
                    messages.append(
                        {
                            "role": "user",
                            "content": prompt_text,
                            "images": [uploaded_file]
                        }
                    )
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    logger.error(f"Error processing image: {e}")
                    # Fallback to text-only
                    messages.append({"role": "user", "content": user_input})
            else:
                # Text-only message
                messages.append({"role": "user", "content": user_input})
            
            # Display typing indicator
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                
                try:
                    # Handle different providers
                    provider = st.session_state.get("multimodal_provider", "Ollama")
                    model = st.session_state.multimodal_selected_model
                    
                    if not model:
                        st.error("No model selected. Please select a model first.")
                        return
                    
                    # Process based on provider
                    if provider == "Ollama":
                        # Use version-independent API call to handle different client versions
                        from ollama_workbench.providers.ollama_utils import call_ollama_endpoint
                        
                        # Check if client supports chat API with messages
                        try:
                            if len(messages) > 1:
                                # Use chat endpoint with history
                                response = client.chat(
                                    model=model,
                                    messages=messages,
                                    options={
                                        "temperature": st.session_state.temperature_slider,
                                        "num_predict": st.session_state.max_tokens_slider
                                    }
                                )
                            else:
                                # Single message
                                if uploaded_file:
                                    # With image
                                    response = client.chat(
                                        model=model,
                                        messages=[{
                                            "role": "user",
                                            "content": user_input if user_input else "Describe this image:",
                                            "images": [uploaded_file]
                                        }],
                                        options={
                                            "temperature": st.session_state.temperature_slider,
                                            "num_predict": st.session_state.max_tokens_slider
                                        }
                                    )
                                else:
                                    # Text only
                                    response = client.chat(
                                        model=model,
                                        messages=[{"role": "user", "content": user_input}],
                                        options={
                                            "temperature": st.session_state.temperature_slider,
                                            "num_predict": st.session_state.max_tokens_slider
                                        }
                                    )
                        except (AttributeError, TypeError) as e:
                            # For clients that don't support the chat API or multimodal inputs
                            if uploaded_file:
                                logger.warning(f"Client doesn't support image inputs: {e}, trying fallbacks")
                                
                                try:
                                    # First try call_ollama_endpoint with the image
                                    prompt = user_input if user_input else "Describe this image:"
                                    # Reset file pointer to beginning
                                    uploaded_file.seek(0)
                                    response_text, _, _, _ = call_ollama_endpoint(
                                        model=model,
                                        prompt=prompt,
                                        image=uploaded_file,
                                        temperature=st.session_state.temperature_slider,
                                        max_tokens=st.session_state.max_tokens_slider
                                    )
                                    response = {"message": {"content": response_text}}
                                except Exception as img_ex:
                                    logger.warning(f"Image endpoint fallback failed: {img_ex}, trying CLI fallback")
                                    
                                    # Try CLI fallback for multimodal models (for example with llava models)
                                    try:
                                        # Save image to a temporary file
                                        import tempfile
                                        import os
                                        
                                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                                            uploaded_file.seek(0)
                                            tmp_file.write(uploaded_file.read())
                                            tmp_path = tmp_file.name
                                        
                                        # Build CLI command for image processing
                                        prompt_text = user_input if user_input else "Describe this image:"
                                        
                                        # Run ollama CLI with image
                                        result = subprocess.run(
                                            ["ollama", "run", model, prompt_text, "--image", tmp_path],
                                            capture_output=True,
                                            text=True,
                                            check=True
                                        )
                                        response_text = result.stdout.strip()
                                        response = {"message": {"content": response_text}}
                                        
                                        # Clean up temporary file
                                        os.unlink(tmp_path)
                                        
                                    except Exception as cli_ex:
                                        logger.error(f"CLI fallback failed: {cli_ex}")
                                        st.error(f"Error communicating with Ollama CLI: {cli_ex}")
                                        response = None # Indicate failure
                            else:
                                # Text-only fallback for clients that don't support chat API
                                logger.warning(f"Client doesn't support chat API: {e}, trying text-only endpoint")
                                try:
                                    response_text, _, _, _ = call_ollama_endpoint(
                                        model=model,
                                        prompt=user_input,
                                        temperature=st.session_state.temperature_slider,
                                        max_tokens=st.session_state.max_tokens_slider
                                    )
                                    response = {"message": {"content": response_text}}
                                except Exception as text_ex:
                                    logger.error(f"Text-only endpoint fallback failed: {text_ex}")
                                    st.error(f"Error communicating with Ollama API: {text_ex}")
                                    response = None # Indicate failure
                        
                        # Process response
                        if response and "message" in response and "content" in response["message"]:
                            full_response = response["message"]["content"]
                        else:
                            full_response = "Error: Could not get a response from the model."
                            logger.error("Invalid response format from Ollama API or fallback.")
                            
                    elif provider == "OpenAI":
                        from ollama_workbench.providers.openai_utils import get_openai_client
                        openai_client = get_openai_client(st.session_state.multimodal_openai_api_key)
                        
                        if not openai_client:
                            st.error("OpenAI client not initialized. Please check your API key.")
                            full_response = "Error: OpenAI client not initialized."
                        else:
                            try:
                                # Prepare messages for OpenAI
                                openai_messages = []
                                for msg in st.session_state.multimodal_chat_history[:-1]:
                                    if msg["role"] == "user":
                                        # OpenAI expects user messages to be a list of content blocks if multimodal
                                        if "images" in msg and msg["images"]:
                                            content_blocks = []
                                            if "content" in msg and msg["content"]:
                                                content_blocks.append({"type": "text", "text": msg["content"]})
                                            for img in msg["images"]:
                                                if "image_url" in img and "url" in img["image_url"]:
                                                    content_blocks.append({"type": "image_url", "image_url": {"url": img["image_url"]["url"]}})
                                            openai_messages.append({"role": "user", "content": content_blocks})
                                        else:
                                            # Text only user message
                                            openai_messages.append({"role": "user", "content": msg["content"]})
                                    else:
                                        # Assistant message (text only for now)
                                        openai_messages.append({"role": "assistant", "content": msg["content"]})
                                
                                # Add the current user message (with image if present)
                                current_user_content = []
                                if user_input:
                                    current_user_content.append({"type": "text", "text": user_input})
                                if uploaded_file:
                                    # Need to convert uploaded_file to base64 for OpenAI
                                    base64_image = image_to_base64(uploaded_file)
                                    mime_type = uploaded_file.type if uploaded_file.type else "image/jpeg"
                                    current_user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}})
                                
                                if current_user_content:
                                    openai_messages.append({"role": "user", "content": current_user_content})
                                
                                # Make the API call
                                response = openai_client.chat.completions.create(
                                    model=model,
                                    messages=openai_messages,
                                    temperature=st.session_state.temperature_slider,
                                    max_tokens=st.session_state.max_tokens_slider
                                )
                                full_response = response.choices[0].message.content
                                
                            except Exception as e:
                                logger.error(f"OpenAI API error: {e}")
                                st.error(f"Error communicating with OpenAI API: {e}")
                                full_response = "Error: Could not get a response from the model."
                                
                    elif provider == "Groq":
                        from ollama_workbench.providers.groq_utils import get_groq_client
                        groq_client = get_groq_client(st.session_state.multimodal_groq_api_key)
                        
                        if not groq_client:
                            st.error("Groq client not initialized. Please check your API key.")
                            full_response = "Error: Groq client not initialized."
                        else:
                            if uploaded_file:
                                st.warning("Groq does not support image inputs yet. Processing as text-only.")
                            
                            try:
                                # Prepare messages for Groq (text only)
                                groq_messages = []
                                for msg in st.session_state.multimodal_chat_history:
                                     # Groq expects simple text content
                                    if isinstance(msg.get("content"), list):
                                        # Extract text from multimodal content for Groq
                                        text_content = " ".join([part.get("text", "") for part in msg["content"] if isinstance(part, dict) and part.get("type") == "text"])
                                        if text_content:
                                            groq_messages.append({"role": msg["role"], "content": text_content})
                                    elif isinstance(msg.get("content"), str):
                                         groq_messages.append({"role": msg["role"], "content": msg["content"]})
                                
                                # Make the API call
                                response = groq_client.chat.completions.create(
                                    model=model,
                                    messages=groq_messages,
                                    temperature=st.session_state.temperature_slider,
                                    max_tokens=st.session_state.max_tokens_slider
                                )
                                full_response = response.choices[0].message.content
                                
                            except Exception as e:
                                logger.error(f"Groq API error: {e}")
                                st.error(f"Error communicating with Groq API: {e}")
                                full_response = "Error: Could not get a response from the model."
                                
                    elif provider == "Mistral":
                        from ollama_workbench.providers.mistral_utils import get_mistral_client
                        mistral_client = get_mistral_client(st.session_state.multimodal_mistral_api_key)
                        
                        if not mistral_client:
                            st.error("Mistral client not initialized. Please check your API key.")
                            full_response = "Error: Mistral client not initialized."
                        else:
                            # Check if the selected Mistral model supports vision
                            is_mistral_vision = "large-2" in model.lower() or "next" in model.lower() # Assuming 'next' models will also support vision
                            
                            if uploaded_file and not is_mistral_vision:
                                st.warning(f"Mistral model '{model}' does not support image inputs. Processing as text-only.")
                            
                            try:
                                # Prepare messages for Mistral
                                mistral_messages = []
                                for msg in st.session_state.multimodal_chat_history[:-1]:
                                    if msg["role"] == "user":
                                        # Mistral expects user messages to be a list of content blocks if multimodal
                                        if "images" in msg and msg["images"] and is_mistral_vision:
                                            content_blocks = []
                                            if "content" in msg and msg["content"]:
                                                content_blocks.append({"type": "text", "text": msg["content"]})
                                            for img in msg["images"]:
                                                if "image_url" in img and "url" in img["image_url"]:
                                                    content_blocks.append({"type": "image_url", "image_url": {"url": img["image_url"]["url"]}})
                                            mistral_messages.append({"role": "user", "content": content_blocks})
                                        else:
                                            # Text only user message
                                            mistral_messages.append({"role": "user", "content": msg["content"]})
                                    else:
                                        # Assistant message (text only for now)
                                        mistral_messages.append({"role": "assistant", "content": msg["content"]})
                                
                                # Add the current user message (with image if present and model supports it)
                                current_user_content = []
                                if user_input:
                                    current_user_content.append({"type": "text", "text": user_input})
                                if uploaded_file and is_mistral_vision:
                                    # Need to convert uploaded_file to base64 for Mistral
                                    base64_image = image_to_base64(uploaded_file)
                                    mime_type = uploaded_file.type if uploaded_file.type else "image/jpeg"
                                    current_user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}})
                                
                                if current_user_content:
                                    mistral_messages.append({"role": "user", "content": current_user_content})
                                
                                # Make the API call
                                response = mistral_client.chat.completions.create(
                                    model=model,
                                    messages=mistral_messages,
                                    temperature=st.session_state.temperature_slider,
                                    max_tokens=st.session_state.max_tokens_slider
                                )
                                full_response = response.choices[0].message.content
                                
                            except Exception as e:
                                logger.error(f"Mistral API error: {e}")
                                st.error(f"Error communicating with Mistral API: {e}")
                                full_response = "Error: Could not get a response from the model."
                                
                    else:
                        full_response = "Error: Unknown provider selected."
                        logger.error(f"Unknown provider selected: {provider}")
                        
                    # Update the placeholder with the full response
                    message_placeholder.markdown(full_response)
                    
                    # Add assistant response to history
                    st.session_state.multimodal_chat_history.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    logger.error(f"An unexpected error occurred during chat processing: {e}")
                    st.error(f"An unexpected error occurred: {e}")
                    message_placeholder.markdown("Error: An unexpected error occurred.")
                    
                finally:
                    # Clear input fields after sending
                    st.session_state.multimodal_text_input = ""
                    st.session_state.multimodal_image_input = None
                    st.rerun()


# Run the app
if __name__ == "__main__":
    multimodal_chat_interface()
