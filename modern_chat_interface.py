# modern_chat_interface.py

import os
import json
import base64
import logging
import streamlit as st
import tiktoken

# Import utilities
# Try to use robust_ollama_utils first, then fall back to other modules
try:
    # First try to import from robust_ollama_utils (preferred for full functionality)
    from robust_ollama_utils import (
        get_all_models, load_api_keys, call_ollama_endpoint, save_ai_content_to_workspace,
        create_model_selection_ui
    )
    import logging
    logging.getLogger(__name__).info("CHECKPOINT: Using robust Ollama utilities module with full functionality")
except ImportError:
    try:
        # Try simplified_ollama_utils next
        from simplified_ollama_utils import (
            get_all_models, load_api_keys, call_ollama_endpoint, save_ai_content_to_workspace
        )
        import logging
        logging.getLogger(__name__).info("CHECKPOINT: Using simplified Ollama utilities module with workspace support")
    except ImportError:
        # Fall back to original ollama_utils
        from ollama_utils import (
            get_all_models, load_api_keys, call_ollama_endpoint, save_ai_content_to_workspace
        )
        import logging
        logging.getLogger(__name__).info("CHECKPOINT: Using original Ollama utilities module")
from openai_utils import call_openai_api
from groq_utils import get_groq_client, call_groq_api
from mistral_utils import call_mistral_api
from prompts import (
    get_agent_prompt, get_metacognitive_prompt, get_voice_prompt
)
from tts_utils import text_to_speech, play_speech
# Load chat_workspace_ui dynamically when needed to avoid circular imports

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

# Constants
SESSIONS_FOLDER = "sessions"
SETTINGS_FILE = "chat-settings.json"
RAG_TEST_DIR = "tests/rag_test_data"

# Ensure sessions folder exists
os.makedirs(SESSIONS_FOLDER, exist_ok=True)

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a helpful, harmless, and honest AI assistant."""

# Function to count tokens
def count_tokens(text):
    """Count tokens in text."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        # Log the error and use fallback
        logger.warning(f"Error counting tokens: {str(e)}")
        # Fallback: estimate tokens as words / 0.75
        return len(text.split()) // 0.75

# Function to check if a model is vision-capable
def is_vision_capable(model_name):
    """Check if a model supports vision/multimodal capabilities"""
    # Clean up model name to remove any suffixes
    clean_name = model_name.split(" ")[0].lower()
    
    # List of known vision-capable models
    vision_models = [
        "llava", "bakllava", "llava-llama", "llava-v1.5", "llava-v1.6",
        "moondream", "cogvlm", "gpt-4-vision", "gpt-4o", "gpt-4-turbo",
        "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
        "qwen-vl", "yi-vl", "internvl", "idefics", "fuyu"
    ]
    
    # Check for partial matches
    for vm in vision_models:
        if vm in clean_name:
            logger.info(f"CHECKPOINT: Model {model_name} identified as vision-capable")
            return True
    
    # Check for specific suffixes indicating vision capability
    if "(vision)" in model_name or "-vision" in model_name:
        return True
    
    logger.info(f"CHECKPOINT: Model {model_name} does not appear to be vision-capable")
    return False

# Function to get Ollama client
def get_ollama_client():
    """Get an Ollama client instance"""
    try:
        from ollama import Client
        client = Client(host="http://localhost:11434")
        logger.info("CHECKPOINT: Successfully created Ollama client")
        return client
    except Exception as e:
        logger.error(f"CHECKPOINT: Error creating Ollama client: {str(e)}")
        return None

# Function to extract code blocks from text
def extract_code_blocks(text):
    """Extract code blocks from text for better rendering"""
    if not text:
        return [], [""]
        
    # Split by code blocks
    parts = text.split("```")
    
    # If no code blocks, return the original text
    if len(parts) == 1:
        return [], [text]
    
    # Extract code blocks and text parts
    code_blocks = []
    content_parts = []
    
    for i, part in enumerate(parts):
        if i % 2 == 0:  # Even parts are text
            content_parts.append(part)
        else:  # Odd parts are code
            code_blocks.append(part)
    
    # If there's an odd number of ```, add an empty text part
    if len(parts) % 2 == 0:
        content_parts.append("")
    
    return code_blocks, content_parts

# Function to load settings
def load_settings():
    """Load settings with better error handling and default values."""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
                
                # Initialize session state variables from settings
                st.session_state.selected_model = settings.get("selected_model", "llama2")
                st.session_state.current_model = settings.get("current_model", st.session_state.selected_model)
                st.session_state.agent_type = settings.get("agent_type", "None")
                st.session_state.metacognitive_type = settings.get("metacognitive_type", "None")
                st.session_state.voice_type = settings.get("voice_type", "None")
                st.session_state.selected_corpus = settings.get("selected_corpus", "None")
                st.session_state.temperature_slider_chat = settings.get("temperature_slider_chat", 0.7)
                st.session_state.max_tokens_slider_chat = settings.get("max_tokens_slider_chat", 4000)
                st.session_state.presence_penalty_slider_chat = settings.get("presence_penalty_slider_chat", 0.0)
                st.session_state.frequency_penalty_slider_chat = settings.get("frequency_penalty_slider_chat", 0.0)
                st.session_state.episodic_memory_enabled = settings.get("episodic_memory_enabled", False)
                st.session_state.advanced_thinking_enabled = settings.get("advanced_thinking_enabled", False)
                st.session_state.instance_adaptive_cot_enabled = settings.get("instance_adaptive_cot_enabled", False)
                st.session_state.thinking_steps = settings.get("thinking_steps", [
                    "1. Analyzing the problem",
                    "2. Breaking down into subtasks",
                    "3. Exploring potential solutions",
                    "4. Evaluating approaches",
                    "5. Formulating a comprehensive answer"
                ])
        else:
            # Default settings
            st.session_state.selected_model = "llama2"
            st.session_state.current_model = "llama2"
            st.session_state.agent_type = "None"
            st.session_state.metacognitive_type = "None"
            st.session_state.voice_type = "None"
            st.session_state.selected_corpus = "None"
            st.session_state.temperature_slider_chat = 0.7
            st.session_state.max_tokens_slider_chat = 4000
            st.session_state.presence_penalty_slider_chat = 0.0
            st.session_state.frequency_penalty_slider_chat = 0.0
            st.session_state.episodic_memory_enabled = False
            st.session_state.advanced_thinking_enabled = False
            st.session_state.instance_adaptive_cot_enabled = False
            st.session_state.thinking_steps = [
                "1. Analyzing the problem",
                "2. Breaking down into subtasks",
                "3. Exploring potential solutions",
                "4. Evaluating approaches",
                "5. Formulating a comprehensive answer"
            ]
            
        # Ensure current_model is set
        if "current_model" not in st.session_state:
            st.session_state.current_model = st.session_state.selected_model
            
        logger.info(f"Settings loaded: current_model={st.session_state.get('current_model')}, selected_model={st.session_state.get('selected_model')}")
        return True
    except Exception as e:
        logger.error(f"Error loading settings: {str(e)}")
        return False

# Function to save settings
def save_settings():
    """Save settings with improved error handling and feedback."""
    try:
        # Ensure session state is initialized
        if "selected_model" not in st.session_state:
            logger.warning("Cannot save settings: selected_model not in session state")
            return False
            
        # Prepare settings dictionary
        settings = {
            "selected_model": st.session_state.get("selected_model"),
            "current_model": st.session_state.get("current_model"),  # Save both for compatibility
            "agent_type": st.session_state.get("agent_type", "None"),
            "metacognitive_type": st.session_state.get("metacognitive_type", "None"),
            "voice_type": st.session_state.get("voice_type", "None"),
            "selected_corpus": st.session_state.get("selected_corpus", "None"),
            "temperature_slider_chat": st.session_state.get("temperature_slider_chat", 0.7),
            "max_tokens_slider_chat": st.session_state.get("max_tokens_slider_chat", 4000),
            "presence_penalty_slider_chat": st.session_state.get("presence_penalty_slider_chat", 0.0),
            "frequency_penalty_slider_chat": st.session_state.get("frequency_penalty_slider_chat", 0.0),
            "episodic_memory_enabled": st.session_state.get("episodic_memory_enabled", False),
            "advanced_thinking_enabled": st.session_state.get("advanced_thinking_enabled", False),
            "instance_adaptive_cot_enabled": st.session_state.get("instance_adaptive_cot_enabled", False),
            "thinking_steps": st.session_state.get("thinking_steps", [
                "1. Analyzing the problem",
                "2. Breaking down into subtasks",
                "3. Exploring potential solutions",
                "4. Evaluating approaches",
                "5. Formulating a comprehensive answer"
            ])
        }
        
        # Save to file
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
            
        logger.info("Settings saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving settings: {str(e)}")
        return False

# Function to construct agent prompt
def construct_agent_prompt(agent_type, metacognitive_type, voice_type):
    """Construct a prompt based on agent type, metacognitive type, and voice type."""
    prompt_parts = []
    
    # Get available prompts
    agent_prompts = get_agent_prompt()
    metacognitive_prompts = get_metacognitive_prompt()
    voice_prompts = get_voice_prompt()
    
    # Add agent type prompt if selected
    if agent_type and agent_type != "None":
        if agent_type in agent_prompts:
            if isinstance(agent_prompts[agent_type], dict):
                prompt_parts.append(agent_prompts[agent_type].get("prompt", ""))
            else:
                prompt_parts.append(agent_prompts[agent_type])
        else:
            logger.warning(f"CHECKPOINT: Agent type '{agent_type}' not found in available prompts")
    
    # Add metacognitive type prompt if selected
    if metacognitive_type and metacognitive_type != "None":
        if metacognitive_type in metacognitive_prompts:
            prompt_parts.append(metacognitive_prompts[metacognitive_type])
        else:
            logger.warning(f"CHECKPOINT: Metacognitive type '{metacognitive_type}' not found in available prompts")
    
    # Add voice type prompt if selected
    if voice_type and voice_type != "None":
        if voice_type in voice_prompts:
            prompt_parts.append(voice_prompts[voice_type])
        else:
            logger.warning(f"CHECKPOINT: Voice type '{voice_type}' not found in available prompts")
    
    # If no prompts were added, use default
    if not prompt_parts:
        prompt_parts.append(DEFAULT_SYSTEM_PROMPT)
    
    return "\n\n".join(prompt_parts)

# Function to get GraphRAG context
def get_graphrag_context(query, corpus_name):
    """Get context from GraphRAG corpus."""
    try:
        # Import here to avoid circular imports
        from chat_workspace import get_graphrag_context as workspace_get_graphrag_context
        return workspace_get_graphrag_context(query, corpus_name)
    except Exception as e:
        logger.error(f"Error getting GraphRAG context: {str(e)}")
        return ""

# Function to process user message
def process_message(user_input, uploaded_image=None):
    """Process user input and generate AI response."""
    # Get the selected model
    model_name = st.session_state.get("selected_model", "llama3")
    logger.info(f"CHECKPOINT: Processing message with model {model_name}")
    
    # Check if we should add the message to chat history
    if user_input:
        # Add the user message to chat history
        message_content = user_input
        
        # Add image info to the history if present
        if uploaded_image:
            logger.info(f"CHECKPOINT: Processing with image: {uploaded_image.name}, type: {uploaded_image.type}, size: {uploaded_image.size}")
            message_content += " [Image attached]"  # Add indicator in history
        
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": message_content})
        
        # Get system prompt based on settings
        system_prompt = ""
        
        # Check if we're using an agent type
        if st.session_state.get("use_agent_type", False):
            agent_type = st.session_state.get("agent_type", "helpful")
            system_prompt = get_agent_prompt(agent_type)
            logger.info(f"CHECKPOINT: Using agent type: {agent_type}")
        
        # Check if we're using metacognitive type
        if st.session_state.get("use_metacognitive_type", False):
            metacognitive_type = st.session_state.get("metacognitive_type", "reflective")
            system_prompt = get_metacognitive_prompt(metacognitive_type)
            logger.info(f"CHECKPOINT: Using metacognitive type: {metacognitive_type}")
        
        # Check if we're using voice type
        if st.session_state.get("use_voice_type", False):
            voice_type = st.session_state.get("voice_type", "helpful")
            system_prompt = get_voice_prompt(voice_type)
            logger.info(f"CHECKPOINT: Using voice type: {voice_type}")
        
        # Get chat history as formatted string
        chat_history = ""
        # Only include the last few messages based on the context window setting
        history_limit = st.session_state.get("context_window", 4)
        recent_messages = st.session_state.chat_history[-(history_limit*2):] if len(st.session_state.chat_history) > history_limit*2 else st.session_state.chat_history
        
        for message in recent_messages[:-1]:  # Exclude the current message
            role = message["role"].capitalize()
            content = message["content"]
            chat_history += f"{role}: {content}\n\n"
        
        logger.info(f"CHECKPOINT: Using {len(recent_messages)} messages from history")
        
        # Get RAG context if enabled
        rag_context = ""
        if st.session_state.get("use_rag", False):
            try:
                # This is a placeholder for RAG functionality
                # In a real implementation, you would retrieve relevant documents here
                rag_context = "Context from knowledge base: No relevant documents found."
                logger.info("CHECKPOINT: RAG enabled but no documents retrieved")
            except Exception as e:
                logger.error(f"CHECKPOINT: Error retrieving RAG context: {str(e)}")
                logger.exception(e)
        
        # Add image instruction if an image is uploaded
        image_instruction = ""
        if uploaded_image:
            image_instruction = "\nThe user has uploaded an image. Please analyze the image and respond accordingly.\n"
        
        # Get available prompts
        agent_prompts = get_agent_prompt()
        metacognitive_prompts = get_metacognitive_prompt()
        voice_prompts = get_voice_prompt()
        
        # Construct prompt with agent type, metacognitive type, and voice type
        system_prompt = ""
        
        # Add selected prompts if available
        if st.session_state.get("agent_type") and st.session_state.get("agent_type") != "None":
            agent_type = st.session_state.agent_type
            if agent_type in agent_prompts:
                if isinstance(agent_prompts[agent_type], dict):
                    system_prompt += agent_prompts[agent_type].get("prompt", "") + "\n"
                else:
                    system_prompt += agent_prompts[agent_type] + "\n"
            else:
                logger.warning(f"CHECKPOINT: Agent type '{agent_type}' not found in available prompts")
                
        if st.session_state.get("metacognitive_type") and st.session_state.get("metacognitive_type") != "None":
            metacognitive_type = st.session_state.metacognitive_type
            if metacognitive_type in metacognitive_prompts:
                system_prompt += metacognitive_prompts[metacognitive_type] + "\n"
            else:
                logger.warning(f"CHECKPOINT: Metacognitive type '{metacognitive_type}' not found in available prompts")
                
        if st.session_state.get("voice_type") and st.session_state.get("voice_type") != "None":
            voice_type = st.session_state.voice_type
            if voice_type in voice_prompts:
                system_prompt += voice_prompts[voice_type] + "\n"
            else:
                logger.warning(f"CHECKPOINT: Voice type '{voice_type}' not found in available prompts")
        
        # If no system prompt, use default
        if not system_prompt.strip():
            system_prompt = "You are a helpful AI assistant."
        
        # Construct recent chat history for context
        chat_history = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in st.session_state.chat_history[-5:]
        ])
        
        # Get the model name to include in prompt
        model_name = st.session_state.get("selected_model", st.session_state.get("current_model", "llama2"))
        
        # Combine everything into the full prompt
        # Add special instructions for image handling if an image is present
        image_instruction = ""
        if uploaded_image:
            image_instruction = "\nThe user has uploaded an image. Please analyze the image and respond accordingly.\n"
        
        prompt = f"""
{system_prompt}

You are an AI assistant using the {model_name} model.{image_instruction}

Recent conversation history:
{chat_history}

{rag_context}

Human: {user_input}

Assistant:
"""
        
        logger.info("CHECKPOINT: Created full prompt for model")
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Process with appropriate API
                if model_name.startswith("openai/"):
                    # OpenAI model
                    model_name = model_name[7:]  # Remove "openai/" prefix
                    api_keys = load_api_keys()
                    logger.info(f"CHECKPOINT: Using OpenAI model: {model_name}")
                    
                    for chunk in call_openai_api(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=st.session_state.get("temperature_slider_chat", 0.7),
                        max_tokens=st.session_state.get("max_tokens_slider_chat", 4000),
                        openai_api_key=api_keys.get("openai_api_key"),
                        stream=True
                    ):
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                elif model_name.startswith("groq/"):
                    # Groq model
                    model_name = model_name[5:]  # Remove "groq/" prefix
                    api_keys = load_api_keys()
                    logger.info(f"CHECKPOINT: Using Groq model: {model_name}")
                    
                    client = get_groq_client(api_keys.get("groq_api_key"))
                    full_response = call_groq_api(
                        client=client,
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=st.session_state.get("temperature_slider_chat", 0.7),
                        max_tokens=st.session_state.get("max_tokens_slider_chat", 4000)
                    )
                    message_placeholder.markdown(full_response)
                    
                elif model_name.startswith("mistral/"):
                    # Mistral model
                    model_name = model_name[8:]  # Remove "mistral/" prefix
                    api_keys = load_api_keys()
                    logger.info(f"CHECKPOINT: Using Mistral model: {model_name}")
                    
                    full_response = call_mistral_api(
                        model=model_name,
                        prompt=prompt,
                        temperature=st.session_state.get("temperature_slider_chat", 0.7),
                        max_tokens=st.session_state.get("max_tokens_slider_chat", 4000),
                        mistral_api_key=api_keys.get("mistral_api_key")
                    )
                    message_placeholder.markdown(full_response)
                    
                else:
                    # Ollama model
                    logger.info(f"CHECKPOINT: Using Ollama model: {model_name}")
                    
                    # First try with the Python client library
                    try:
                        # Use the Ollama client first
                        client = get_ollama_client()
                        
                        if client is not None:
                            try:
                                # Check if we have an image and need to use multimodal capabilities
                                if uploaded_image and is_vision_capable(model_name):
                                    logger.info(f"CHECKPOINT: Using vision capabilities for model {model_name}")
                                    
                                    # Reset file pointer
                                    uploaded_image.seek(0)
                                    image_bytes = uploaded_image.read()
                                    
                                    # Get MIME type from file type
                                    mime_type = uploaded_image.type
                                    
                                    # Encode image
                                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                                    image_data = f"data:{mime_type};base64,{base64_image}"
                                    
                                    # Create multimodal message format
                                    messages = [
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "text", "text": prompt},
                                                {"type": "image_url", "image_url": {"url": image_data}}
                                            ]
                                        }
                                    ]
                                    
                                    try:
                                        # Use the generate endpoint with images
                                        for response_chunk in client.generate(
                                            model=model_name,
                                            messages=messages,
                                            stream=True,
                                            options={
                                                "temperature": st.session_state.get("temperature_slider_chat", 0.7),
                                                "num_predict": st.session_state.get("max_tokens_slider_chat", 4000)
                                            }
                                        ):
                                            content = response_chunk.get("message", {}).get("content", "")
                                            if content:
                                                full_response += content
                                                message_placeholder.markdown(full_response + "▌")
                                    except Exception as vision_error:
                                        logger.error(f"CHECKPOINT: Error using multimodal API: {str(vision_error)}")
                                        # Fallback to standard prompt with image description
                                        prompt += "\n[Note: User uploaded an image but the model couldn't process it directly]\n"
                                        # Use standard text generation as fallback
                                        for response_chunk in client.generate(
                                            model=model_name,
                                            prompt=prompt,
                                            stream=True,
                                            options={
                                                "temperature": st.session_state.get("temperature_slider_chat", 0.7),
                                                "num_predict": st.session_state.get("max_tokens_slider_chat", 4000)
                                            }
                                        ):
                                            content = response_chunk["response"]
                                            full_response += content
                                            message_placeholder.markdown(full_response + "▌")
                                else:
                                    # Standard text-only generation
                                    for response_chunk in client.generate(
                                        model=model_name,
                                        prompt=prompt,
                                        stream=True,
                                        options={
                                            "temperature": st.session_state.get("temperature_slider_chat", 0.7),
                                            "num_predict": st.session_state.get("max_tokens_slider_chat", 4000)
                                        }
                                    ):
                                        content = response_chunk["response"]
                                        full_response += content
                                        message_placeholder.markdown(full_response + "▌")
                                
                                # Final display without cursor
                                message_placeholder.markdown(full_response)
                                logger.info("CHECKPOINT: Successfully generated response with Ollama client")
                            except Exception as client_error:
                                logger.exception(f"CHECKPOINT: Error with Ollama client: {str(client_error)}")
                                logger.info("CHECKPOINT: Falling back to endpoint call")
                                
                                # Use non-streaming fallback
                                full_response, _, _, _ = call_ollama_endpoint(
                                    model=model_name,
                                    prompt=prompt,
                                    temperature=st.session_state.get("temperature_slider_chat", 0.7),
                                    max_tokens=st.session_state.get("max_tokens_slider_chat", 4000)
                                )
                                message_placeholder.markdown(full_response)
                                logger.info("CHECKPOINT: Successfully generated response with fallback method")
                        else:
                            # Client is None, use endpoint directly
                            logger.info("CHECKPOINT: Ollama client not available, using endpoint directly")
                            full_response, _, _, _ = call_ollama_endpoint(
                                model=model_name,
                                prompt=prompt,
                                temperature=st.session_state.get("temperature_slider_chat", 0.7),
                                max_tokens=st.session_state.get("max_tokens_slider_chat", 4000)
                            )
                            message_placeholder.markdown(full_response)
                    except Exception as outer_error:
                        logger.exception(f"CHECKPOINT: Error with Ollama client initialization: {str(outer_error)}")
                        logger.info("CHECKPOINT: Falling back to endpoint call")
                        
                        # Use non-streaming fallback
                        full_response, _, _, _ = call_ollama_endpoint(
                            model=model_name,
                            prompt=prompt,
                            temperature=st.session_state.get("temperature_slider_chat", 0.7),
                            max_tokens=st.session_state.get("max_tokens_slider_chat", 4000)
                        )
                        message_placeholder.markdown(full_response)
                        logger.info("CHECKPOINT: Successfully generated response with fallback method")
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                logger.info(f"CHECKPOINT: Response generated, length: {len(full_response)}")
                
                # Check if response contains code blocks and save to workspace
                if "```" in full_response:
                    logger.info("CHECKPOINT: Detected code blocks in response, saving to workspace")
                    try:
                        save_ai_content_to_workspace(full_response)
                        logger.info("CHECKPOINT: Successfully saved content to workspace")
                    except Exception as e:
                        logger.error(f"CHECKPOINT: Error saving content to workspace: {str(e)}")
                        logger.exception(e)
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                message_placeholder.error(error_message)
                logger.exception(f"CHECKPOINT: Error generating response: {str(e)}")
                full_response = error_message
                # Still add to history so user knows something went wrong
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
def generate_prompt_suggestion(user_need):
    """Generate a prompt suggestion based on the user need."""
    try:
        # Simple implementation without API calls
        if not user_need:
            return None
            
        # Just create a standardized format
        suggestion = f"""I need help with: {user_need.strip()}

Please provide a detailed and clear response that addresses the following aspects:
1. A thorough explanation of the key concepts
2. Practical examples or applications if relevant
3. Any important considerations or limitations to be aware of
4. Next steps or recommendations

Please format your response clearly with sections and bullet points where appropriate."""
        
        logger.info(f"CHECKPOINT: Generated prompt suggestion for: {user_need[:50]}...")
        return suggestion
    except Exception as e:
        logger.error(f"CHECKPOINT: Error generating prompt suggestion: {e}")
        return None

# Main function for the modern chat interface
def modern_chat_interface():
    """Main chat interface function."""
    logger.info("CHECKPOINT: Starting chat interface")
    
    # Initialize session state for chat history if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Initialize session state for prompt modal
    if "show_prompt_modal" not in st.session_state:
        st.session_state.show_prompt_modal = False
        
    # Initialize session state for chat input
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""
        
    # Show prompt suggestion modal if enabled
    if st.session_state.show_prompt_modal:
        with st.container():
            st.markdown("""
            <style>
            .stModal > div[data-testid="stHorizontalBlock"]:first-child {
                display: none !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.subheader("AI Prompt Writer")
            
            if st.button("X", key="close_modal", help="Cancel assisted prompt writing."):
                st.session_state.show_prompt_modal = False
                st.rerun()
            
            user_need = st.text_input("What do you need help with?")
            if user_need:
                prompt_suggestion = generate_prompt_suggestion(user_need)
                if prompt_suggestion:
                    st.write("Suggested prompt:")
                    edited_prompt = st.text_area("Edit the prompt before using it:", value=prompt_suggestion)
                    if st.button("Use this prompt"):
                        st.session_state.chat_input = edited_prompt
                        st.session_state.show_prompt_modal = False
                        st.rerun()
                else:
                    st.warning("Unable to generate a prompt suggestion. Please try again or select a different model.")
    
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "avoid_rerun" not in st.session_state:
        st.session_state.avoid_rerun = False
    
    # Load settings
    load_settings()
    logger.info("CHECKPOINT: Settings loaded")
    
    # Initialize tab state if needed
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Chat"
    
    # Add vertical spacing to prevent tabs from being covered by the top bar
    st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)
    
    # Create tabs for different sections with clear styling
    tab1, tab2 = st.tabs(["💬 Chat", "🔍 Workspace"])
    
    # Chat tab
    with tab1:
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            role = message.get("role", "assistant")
            content = message.get("content", "")
            
            # Use Streamlit's built-in chat message component
            with st.chat_message(role):
                # Extract code blocks for better rendering
                code_blocks, content_parts = extract_code_blocks(content)
                
                # Handle TTS for assistant messages if enabled
                if role == "assistant" and st.session_state.get("tts_enabled", False):
                    # Create columns for audio control and message content
                    cols = st.columns([1, 9])
                    
                    # Display TTS control in first column
                    with cols[0]:
                        if st.button("🔊", key=f"tts_{i}"):
                            try:
                                speech_file = text_to_speech(content)
                                play_speech(speech_file)
                            except Exception as e:
                                st.error(f"TTS error: {str(e)}")
                    
                    # Display message content in second column
                    with cols[1]:
                        current_idx = 0
                        for part_idx, part in enumerate(content_parts):
                            if part.strip():
                                st.markdown(part)
                            if current_idx < len(code_blocks):
                                st.code(code_blocks[current_idx])
                                current_idx += 1
                else:
                    # Regular message display
                    current_idx = 0
                    for part_idx, part in enumerate(content_parts):
                        if part.strip():
                            st.markdown(part)
                        if current_idx < len(code_blocks):
                            st.code(code_blocks[current_idx])
                            current_idx += 1
    
        # Workspace tab
    with tab2:
        # Placeholder for the chat workspace
        # Using a different approach to handle the workspace
        # This creates an expander and then only renders the workspace UI when it's expanded
        # By dynamically rendering the content, we avoid nested expanders
        workspace_expanded = st.expander("Workspace", expanded=False)
        
        # This will only execute when the expander is actually expanded by the user
        if workspace_expanded.expanded:
            from chat_workspace import chat_workspace_ui
            # Use a container to isolate the workspace UI elements
            with st.container():
                chat_workspace_ui()

    # Chat input and processing with multimodal support (must be outside any container)
    # Create a container for the input area with custom styling
    input_container = st.container()
    
    # Initialize session state for image upload if not exists
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    
    with input_container:
        # Create columns for the buttons and chat input
        col1, col2, col3, col4 = st.columns([0.05, 0.05, 0.05, 0.85])
        
        # Image upload button in the first column
        with col1:
            # Simple button that looks like a + sign
            if st.button("📷", key="image_upload_button", help="Upload an image"):
                # This will be handled by JavaScript
                pass
                
            # Add CSS to style the button
            st.markdown(
                """
                <style>
                /* Style the image upload button */
                [data-testid="baseButton-secondary"]:has(div:contains("📷")) {
                    background-color: transparent !important;
                    border: 1px solid rgba(49, 51, 63, 0.2) !important;
                    border-radius: 50% !important;
                    min-width: 36px !important;
                    width: 36px !important;
                    height: 36px !important;
                    padding: 0 !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                }
                
                /* Hide the file uploader completely */
                .stFileUploader {
                    position: absolute !important;
                    width: 1px !important;
                    height: 1px !important;
                    padding: 0 !important;
                    margin: -1px !important;
                    overflow: hidden !important;
                    clip: rect(0, 0, 0, 0) !important;
                    white-space: nowrap !important;
                    border-width: 0 !important;
                }
                
                /* Image preview container */
                .image-preview-container {
                    margin: 10px 0;
                    padding: 10px;
                    border-radius: 8px;
                    background-color: rgba(255, 255, 255, 0.05);
                    position: relative;
                }
                
                /* Remove button styling */
                .remove-button {
                    position: absolute;
                    top: 5px;
                    right: 5px;
                    width: 24px;
                    height: 24px;
                    border-radius: 50%;
                    background-color: rgba(0, 0, 0, 0.5);
                    color: white;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    cursor: pointer;
                    font-size: 14px;
                }
                </style>
                """, 
                unsafe_allow_html=True
            )
            
            # Create a hidden file uploader
            uploaded_file = st.file_uploader(
                "Upload an image", 
                type=["jpg", "jpeg", "png", "webp"], 
                key="chat_image_upload",
                label_visibility="collapsed"
            )
            
            # Add JavaScript to connect the button to the file uploader
            st.markdown(
                """
                <script>
                // Function to trigger the hidden file uploader when the button is clicked
                document.addEventListener('DOMContentLoaded', function() {
                    // Find the image upload button
                    const uploadButton = document.querySelector('[data-testid="baseButton-secondary"]:has(div:contains("📷"))');
                    const fileInput = document.querySelector('input[type="file"]');
                    
                    if (uploadButton && fileInput) {
                        // Add click event listener to the button
                        uploadButton.addEventListener('click', function(e) {
                            e.preventDefault();
                            fileInput.click();
                        });
                    }
                });
                </script>
                """,
                unsafe_allow_html=True
            )
        
        # Prompt suggestion button in the second column
        with col2:
            if st.button("✨", key="prompt_button", help="Get prompt suggestions"):
                st.session_state.show_prompt_modal = True
                st.rerun()
                
            # Add CSS to style the button
            st.markdown(
                """
                <style>
                /* Style the prompt button */
                [data-testid="baseButton-secondary"]:has(div:contains("✨")) {
                    background-color: transparent !important;
                    border: 1px solid rgba(49, 51, 63, 0.2) !important;
                    border-radius: 50% !important;
                    min-width: 36px !important;
                    width: 36px !important;
                    height: 36px !important;
                    padding: 0 !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                }
                </style>
                """, 
                unsafe_allow_html=True
            )
        
        # Additional button in the third column (can be used for other features)
        with col3:
            # Empty for now, can be used for additional features
            pass
        
        # Chat input in the fourth column
        with col4:
            # Use the chat input
            user_input = st.chat_input("Type a message...", key="user_chat_input")
            
            # If we have a stored prompt from the prompt generator, set it in the session state
            # This will be picked up by JavaScript to set the input value
            if st.session_state.chat_input:
                # Add JavaScript to set the input value
                js_code = f"""
                <script>
                document.addEventListener('DOMContentLoaded', function() {{                    
                    // Find the chat input field
                    const chatInputs = document.querySelectorAll('textarea');
                    if (chatInputs.length > 0) {{
                        // Set the value to our stored prompt
                        const inputField = chatInputs[chatInputs.length - 1];
                        inputField.value = {json.dumps(st.session_state.chat_input)};
                        // Focus and trigger input event to make Streamlit aware of the change
                        inputField.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        inputField.focus();
                    }}
                    // Clear the stored value after setting it
                }});
                </script>
                """
                st.markdown(js_code, unsafe_allow_html=True)
                # Clear the stored chat input after using it
                st.session_state.chat_input = ""
            
            # Add detailed logging to track the chat input process
            if user_input:
                logger.info(f"CHECKPOINT: User input received in chat interface: {user_input[:50]}...")
    
    # Add detailed logging for file upload component
    logger.info("CHECKPOINT: File uploader component rendered with custom styling")
    
    # Process uploaded file if present
    if uploaded_file:
        logger.info(f"CHECKPOINT: File uploaded: {uploaded_file.name}, type: {uploaded_file.type}, size: {uploaded_file.size} bytes")
        # Store the uploaded file in session state
        st.session_state.uploaded_image = uploaded_file
        logger.info("CHECKPOINT: Image uploaded successfully")
        # Force rerun to show the image preview immediately
        st.rerun()
    else:
        logger.debug("CHECKPOINT: No file uploaded yet")
    
    # Display image preview if an image is uploaded
    if st.session_state.uploaded_image is not None:
        # Create a container for the image preview with a remove button
        st.markdown("""
        <div class="image-preview-container">
            <!-- Image will be displayed here by Streamlit -->
            <div class="remove-button" onclick="
                document.querySelector('#remove_image_button').click();
            ">✖</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the image with caption
        st.image(st.session_state.uploaded_image, width=200, caption=st.session_state.uploaded_image.name)
        
        # Add detailed logging about the image
        logger.info(f"CHECKPOINT: Displaying image preview for {st.session_state.uploaded_image.name}")
        
        # Hidden button that will be triggered by the custom remove button
        if st.button("✖", key="remove_image_button", label_visibility="collapsed"):
            logger.info("CHECKPOINT: Removing uploaded image")
            st.session_state.uploaded_image = None
            st.rerun()
    
    # Process the message when user submits input
    if user_input:
        logger.info(f"CHECKPOINT: Processing message with input: {user_input[:50]}...")
        
        # Check if there's an uploaded image
        if st.session_state.uploaded_image is not None:
            # Process the message with the image
            process_message(user_input, st.session_state.uploaded_image)
            
            # Clear the uploaded image after sending
            st.session_state.uploaded_image = None
            # Don't rerun here to avoid interrupting the chat flow
        else:
            # Process the message without an image
            process_message(user_input)

    # Sidebar for settings and controls
    with st.sidebar:
        st.title("Ollama Workbench")
        
        # Session management
        st.subheader("Session Management")
        
        # Session control buttons in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 New", help="Start a new chat session"):
                # Clear chat history and reset session
                st.session_state.chat_history = []
                st.session_state.total_tokens = 0
                st.success("New chat session started!")
                st.rerun()
        
        with col2:
            if st.button("💾 Save", help="Save current chat session"):
                # Save current session
                if len(st.session_state.chat_history) > 0:
                    filepath = save_settings()
                    if filepath:
                        st.success("Session saved successfully!")
                    else:
                        st.error("Failed to save session")
                else:
                    st.warning("No chat history to save")
        
        with col3:
            if st.button("📂 Load", help="Load a saved chat session"):
                # Show session loading UI
                st.session_state.show_load_ui = True
        
        # Show load UI if requested
        if st.session_state.get("show_load_ui", False):
            session_files = [f for f in os.listdir(SESSIONS_FOLDER) if f.endswith(".json")]
            if session_files:
                selected_session = st.selectbox(
                    "Select a session to load",
                    session_files,
                    format_func=lambda x: x.replace("chat_session_", "").replace(".json", "").replace("_", " ")
                )
                
                if st.button("Load Selected Session"):
                    filepath = os.path.join(SESSIONS_FOLDER, selected_session)
                    success = load_settings(filepath)
                    if success:
                        st.success("Session loaded successfully!")
                        st.session_state.show_load_ui = False
                        st.rerun()
                    else:
                        st.error("Failed to load session")
            else:
                st.info("No saved sessions found")
                
            if st.button("Cancel"):
                st.session_state.show_load_ui = False
                st.rerun()
        
        # Model selection section
        with st.sidebar.expander("🔄 Model Selection", expanded=True):
            # Get available models
            available_models = get_all_models()
            logger.info(f"CHECKPOINT: Found {len(available_models)} models for selection UI")
            
            # Use the new categorized model selection UI
            try:
                # Import model_categorization to get descriptions if available
                try:
                    from model_categorization import MODEL_DESCRIPTIONS as model_descriptions
                    logger.info("CHECKPOINT: Using model descriptions from model_categorization module")
                except ImportError:
                    # Fallback model descriptions
                    model_descriptions = {
                        "llama3": "Meta's Llama 3 model - general purpose, instruction-following",
                        "llama2": "Meta's Llama 2 model - general purpose, instruction-following",
                        "mistral": "Mistral 7B - efficient, high-quality instruction model",
                        "gemma": "Google's Gemma model - lightweight, efficient model",
                        "phi": "Microsoft's Phi model - compact, efficient model",
                        "codellama": "Code-specialized Llama model for programming tasks",
                        "orca-mini": "Lightweight model optimized for efficiency",
                        "vicuna": "Fine-tuned LLaMA model with improved instruction following",
                        "stable-diffusion": "Image generation model (multimodal)",
                        "llava": "Multimodal model supporting vision and language",
                        "gpt-4": "OpenAI's most powerful model with strong reasoning",
                        "gpt-4-turbo": "Faster version of GPT-4 with lower latency",
                        "gpt-4-vision": "GPT-4 with vision capabilities (multimodal)",
                        "gpt-3.5-turbo": "Efficient model balancing performance and speed",
                    }
                    logger.info("CHECKPOINT: Using fallback model descriptions")
                
                # Use the categorized model selection UI
                model_selection, changed = create_model_selection_ui(
                    available_models, 
                    st.session_state.selected_model
                )
                
                # Update model if changed
                if changed:
                    previous_model = st.session_state.selected_model
                    st.session_state.selected_model = model_selection
                    st.session_state.current_model = model_selection
                    
                    # Save to settings file
                    try:
                        from model_categorization import save_model_to_settings
                        save_model_to_settings(model_selection)
                    except ImportError:
                        # Fallback to saving in session state only
                        pass
                    
                    logger.info(f"CHECKPOINT: Model changed from '{previous_model}' to '{model_selection}'")
                    st.success(f"Model changed to {model_selection}")
                    st.rerun()
            except Exception as e:
                # Fallback to simple model selection if the categorized UI fails
                logger.error(f"CHECKPOINT: Error in model selection UI: {str(e)}")
                st.error(f"Error in model selection UI: {str(e)}")
                
                # Show current model
                st.write(f"**Current model:** {st.session_state.selected_model}")
                
                # Simple model selection with apply button
                with st.form(key="model_selection_form_fallback"):
                    try:
                        model_index = available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
                    except (ValueError, IndexError):
                        model_index = 0
                        
                    model_selection = st.selectbox(
                        "📦 Model:",
                        available_models,
                        index=model_index
                    )
                    
                    # Apply button
                    apply_button = st.form_submit_button("Apply")
                
                # Show model description
                description = model_descriptions.get(model_selection, 'No description available')
                st.markdown(f"**Description:** {description}")
                
                # Update model on apply
                if apply_button and model_selection != st.session_state.selected_model:
                    st.session_state.selected_model = model_selection
                    st.session_state.current_model = model_selection
                    st.success(f"Model changed to {model_selection}")
                    st.rerun()
                st.rerun()
        
        # Organized settings sections with collapsible expanders
        with st.sidebar:
            # Model Parameters Section
            with st.expander("⚙️ Model Parameters", expanded=True):
                # Temperature setting
                if "temperature_slider_chat" not in st.session_state:
                    st.session_state.temperature_slider_chat = 0.7
                    
                st.session_state.temperature_slider_chat = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.temperature_slider_chat,
                    step=0.1,
                    help="Higher values make output more random, lower values more deterministic"
                )
                
                # Max tokens setting
                if "max_tokens_slider_chat" not in st.session_state:
                    st.session_state.max_tokens_slider_chat = 4000
                    
                st.session_state.max_tokens_slider_chat = st.slider(
                    "Max Tokens",
                    min_value=100,
                    max_value=8000,
                    value=st.session_state.max_tokens_slider_chat,
                    step=100,
                    help="Maximum number of tokens to generate"
                )
                
                # Presence penalty setting
                if "presence_penalty_slider_chat" not in st.session_state:
                    st.session_state.presence_penalty_slider_chat = 0.0
                    
                st.session_state.presence_penalty_slider_chat = st.slider(
                    "Presence Penalty",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.presence_penalty_slider_chat,
                    step=0.1,
                    help="Higher values discourage repetition of topics"
                )
                
                # Frequency penalty setting
                if "frequency_penalty_slider_chat" not in st.session_state:
                    st.session_state.frequency_penalty_slider_chat = 0.0
                    
                st.session_state.frequency_penalty_slider_chat = st.slider(
                    "Frequency Penalty",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.frequency_penalty_slider_chat,
                    step=0.1,
                    help="Higher values discourage repetition of specific phrases"
                )
            
            # Prompt Engineering Section
            with st.expander("🧠 Prompt Engineering", expanded=False):
                # Agent type selection
                if "agent_type" not in st.session_state:
                    st.session_state.agent_type = "None"
                
                # Define available agent types
                agent_types = ["None", "Researcher", "Programmer", "Scientist", "Writer", "Critic"]
                
                # Check if current agent type is in the list, if not set to None
                if st.session_state.agent_type not in agent_types:
                    logger.info(f"CHECKPOINT: Invalid agent type '{st.session_state.agent_type}', resetting to None")
                    st.session_state.agent_type = "None"
                
                st.session_state.agent_type = st.selectbox(
                    "Agent Type",
                    agent_types,
                    index=agent_types.index(st.session_state.agent_type)
                )
                
                # Metacognitive type selection
                if "metacognitive_type" not in st.session_state:
                    st.session_state.metacognitive_type = "None"
                
                # Define available metacognitive types
                metacognitive_types = ["None", "Analytical", "Creative", "Balanced", "Skeptical", "Thorough"]
                
                # Check if current metacognitive type is in the list, if not set to None
                if st.session_state.metacognitive_type not in metacognitive_types:
                    logger.info(f"CHECKPOINT: Invalid metacognitive type '{st.session_state.metacognitive_type}', resetting to None")
                    st.session_state.metacognitive_type = "None"
                
                st.session_state.metacognitive_type = st.selectbox(
                    "Metacognitive Type",
                    metacognitive_types,
                    index=metacognitive_types.index(st.session_state.metacognitive_type)
                )
                
                # Voice type selection
                if "voice_type" not in st.session_state:
                    st.session_state.voice_type = "None"
                
                # Define available voice types
                voice_types = ["None", "Friendly", "Professional", "Academic", "Concise", "Detailed", "Formal"]
                
                # Check if current voice type is in the list, if not set to None
                if st.session_state.voice_type not in voice_types:
                    logger.info(f"CHECKPOINT: Invalid voice type '{st.session_state.voice_type}', resetting to None")
                    st.session_state.voice_type = "None"
                
                st.session_state.voice_type = st.selectbox(
                    "Voice Type",
                    voice_types,
                    index=voice_types.index(st.session_state.voice_type)
                )
            
            # Advanced Thinking Section
            with st.expander("🤔 Advanced Thinking", expanded=False):
                # Advanced thinking
                if "advanced_thinking_enabled" not in st.session_state:
                    st.session_state.advanced_thinking_enabled = False
                    
                st.session_state.advanced_thinking_enabled = st.checkbox(
                    "Enable Advanced Thinking", 
                    value=st.session_state.advanced_thinking_enabled,
                    help="Enable advanced thinking steps for complex reasoning"
                )
                
                # Advanced thinking settings if enabled
                if st.session_state.advanced_thinking_enabled:
                    if "thinking_steps" not in st.session_state:
                        st.session_state.thinking_steps = [
                            "1. Analyzing the problem",
                            "2. Breaking down into subtasks",
                            "3. Exploring potential solutions",
                            "4. Evaluating approaches",
                            "5. Formulating a comprehensive answer"
                        ]
                    
                    thinking_steps = st.text_area(
                        "Thinking Steps (one per line)",
                        value="\n".join(st.session_state.thinking_steps)
                    )
                    
                    # Update thinking steps
                    if thinking_steps:
                        st.session_state.thinking_steps = [step.strip() for step in thinking_steps.split("\n") if step.strip()]
                
                # Instance Adaptive CoT
                if "instance_adaptive_cot_enabled" not in st.session_state:
                    st.session_state.instance_adaptive_cot_enabled = False
                
                st.session_state.instance_adaptive_cot_enabled = st.checkbox(
                    "Enable Instance-Adaptive CoT", 
                    value=st.session_state.instance_adaptive_cot_enabled,
                    help="Uses dynamic Chain-of-Thought prompting strategies based on problem complexity"
                )
                
                # CoT settings if enabled
                if st.session_state.instance_adaptive_cot_enabled:
                    if "cot_strategy" not in st.session_state:
                        st.session_state.cot_strategy = "IAP-ss"
                    if "cot_threshold" not in st.session_state:
                        st.session_state.cot_threshold = 0.5
                    if "cot_top_n" not in st.session_state:
                        st.session_state.cot_top_n = 3
                    
                    st.session_state.cot_strategy = st.selectbox(
                        "CoT Strategy",
                        ["IAP-ss", "IAP-mv"],
                        index=0 if st.session_state.cot_strategy == "IAP-ss" else 1,
                        help="IAP-ss: Single-step selection, IAP-mv: Majority voting"
                    )
                    
                    if st.session_state.cot_strategy == "IAP-ss":
                        st.session_state.cot_threshold = st.slider(
                            "Saliency Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=st.session_state.cot_threshold,
                            step=0.05
                        )
                    else:  # IAP-mv
                        st.session_state.cot_top_n = st.slider(
                            "Top N Prompts",
                            min_value=1,
                            max_value=9,
                            value=st.session_state.cot_top_n,
                            step=1
                        )
            
            # Memory & Context Section
            with st.expander("🧩 Memory & Context", expanded=False):
                # Episodic memory
                if "episodic_memory_enabled" not in st.session_state:
                    st.session_state.episodic_memory_enabled = False
                    
                st.session_state.episodic_memory_enabled = st.checkbox(
                    "Enable Episodic Memory", 
                    value=st.session_state.episodic_memory_enabled,
                    help="Enable episodic memory for better context retention"
                )
                
                # Corpus selection for RAG
                if "selected_corpus" not in st.session_state:
                    st.session_state.selected_corpus = "None"
                
                # Get available corpora
                try:
                    from chat_workspace import get_available_corpora
                    available_corpora = ["None"] + get_available_corpora()
                except Exception as e:
                    logger.error(f"Error getting available corpora: {str(e)}")
                    available_corpora = ["None"]
                
                st.session_state.selected_corpus = st.selectbox(
                    "Knowledge Base",
                    available_corpora,
                    index=available_corpora.index(st.session_state.selected_corpus) if st.session_state.selected_corpus in available_corpora else 0
                )
            
            # Accessibility Section
            with st.expander("🔊 Accessibility", expanded=False):
                # TTS settings
                if "tts_enabled" not in st.session_state:
                    st.session_state.tts_enabled = False
                    
                st.session_state.tts_enabled = st.checkbox(
                    "Enable Text-to-Speech", 
                    value=st.session_state.tts_enabled,
                    help="Enable text-to-speech for assistant messages"
                )
            
            # Save all settings button
            if st.button("💾 Save All Settings"):
                # Prevent rerun during save
                st.session_state.avoid_rerun = True
                
                # Save settings
                save_settings()
                
                # Show success message
                st.success("All settings saved successfully!")
            
            # Reset Button with confirmation
            if st.button("⚠️ Reset All Settings"):
                # Reset all settings to defaults
                st.session_state.selected_model = "llama2"
                st.session_state.current_model = "llama2"  # Also reset current_model
                st.session_state.agent_type = "None"
                st.session_state.metacognitive_type = "None"
                st.session_state.voice_type = "None"
                st.session_state.selected_corpus = "None"
                st.session_state.temperature_slider_chat = 0.7
                st.session_state.max_tokens_slider_chat = 4000
                st.session_state.presence_penalty_slider_chat = 0.0
                st.session_state.frequency_penalty_slider_chat = 0.0
                st.session_state.episodic_memory_enabled = False
                st.session_state.advanced_thinking_enabled = False
                st.session_state.instance_adaptive_cot_enabled = False
                
                # Save settings but prevent rerun
                try:
                    save_settings()
                except st.script_runner.RerunException:
                    pass
                    
                # Show success message
                st.success("All settings reset to defaults!")
                
                # Now we do want to rerun after reset
                st.rerun()

# Run the app
if __name__ == "__main__":
    modern_chat_interface()
    