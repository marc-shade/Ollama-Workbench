#!/usr/bin/env python3

"""
Script to fix the "Object of type Message is not JSON serializable" error in the tool_playground.py.
This script makes the necessary modifications to ensure proper JSON serialization.
"""

import os
import shutil
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to the tool playground file
TOOL_PLAYGROUND_PATH = os.path.join(os.path.dirname(__file__), "tool_playground.py")

# Make sure the file exists
if not os.path.exists(TOOL_PLAYGROUND_PATH):
    logger.error(f"Cannot find tool_playground.py at {TOOL_PLAYGROUND_PATH}")
    exit(1)

# Create a backup
backup_file = f"tool_playground.py.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
backup_path = os.path.join(os.path.dirname(__file__), backup_file)
shutil.copy2(TOOL_PLAYGROUND_PATH, backup_path)
logger.info(f"Created backup at {backup_path}")

# Read the content of the file
with open(TOOL_PLAYGROUND_PATH, "r") as f:
    content = f.read()

# Insert the JSON serialization helper function
serialization_helper = """
# Helper function to ensure objects are JSON serializable
def sanitize_for_json(obj):
    # Helper to convert Python objects to JSON serializable types
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items() if k != "_client"}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif hasattr(obj, 'to_dict'):  # Handle objects with to_dict method
        return sanitize_for_json(obj.to_dict())
    elif hasattr(obj, '__dict__'):  # Handle custom objects
        return sanitize_for_json({k: v for k, v in obj.__dict__.items() 
                                if not k.startswith('_')})
    else:
        try:
            # Test if it's JSON serializable
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            # Convert to string if not serializable
            return str(obj)
"""

# Find the import section to add the function after it
import_section_end = content.find("# Predefined tool templates with example implementations")
if import_section_end > 0:
    modified_content = content[:import_section_end] + serialization_helper + content[import_section_end:]
    logger.info("Added JSON serialization helper function")
else:
    logger.error("Could not find appropriate location to insert helper function")
    exit(1)

# Update the API call to use the sanitization
api_call_original = """                    # Make the API call
                    response = client.chat(
                        model=model_to_use,
                        messages=messages,
                        tools=tools,
                        options={
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    )"""

api_call_fixed = """                    # Make the API call
                    # Sanitize objects to ensure JSON serialization works
                    safe_messages = sanitize_for_json(messages)
                    safe_tools = sanitize_for_json(tools)
                    
                    response = client.chat(
                        model=model_to_use,
                        messages=safe_messages,
                        tools=safe_tools,
                        options={
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    )"""

modified_content = modified_content.replace(api_call_original, api_call_fixed)
logger.info("Updated initial API call to use sanitized objects")

# Update the follow-up call to also use sanitization
followup_call_original = """                            follow_up_response = client.chat(
                                model=model_to_use,  # Use the same model we used for the initial request
                                messages=follow_up_messages,
                                tools=tools,  # Include tools in follow-up request
                                options={
                                    "temperature": temperature,
                                    "num_predict": max_tokens
                                }
                            )"""

followup_call_fixed = """                            # Sanitize follow-up messages and tools
                            safe_followup_messages = sanitize_for_json(follow_up_messages)
                            
                            follow_up_response = client.chat(
                                model=model_to_use,  # Use the same model we used for the initial request
                                messages=safe_followup_messages,
                                tools=safe_tools,  # Use the already sanitized tools
                                options={
                                    "temperature": temperature,
                                    "num_predict": max_tokens
                                }
                            )"""

modified_content = modified_content.replace(followup_call_original, followup_call_fixed)
logger.info("Updated follow-up API call to use sanitized objects")

# Fix the logging to avoid JSON serialization errors
log_original = """                    # Log the response for debugging
                    logger.info(f"Received response from model: {json.dumps(response.get('message', {}), indent=2)}")"""

log_fixed = """                    # Log the response for debugging (sanitized)
                    try:
                        logger.info(f"Received response from model: {json.dumps(sanitize_for_json(response.get('message', {})), indent=2)}")
                    except Exception as log_err:
                        logger.error(f"Error logging response: {log_err}")
                        logger.info(f"Response received but could not be serialized")"""

modified_content = modified_content.replace(log_original, log_fixed)
logger.info("Updated response logging to handle serialization errors")

# Also fix the follow-up response logging
followup_log_original = """                            # Log the follow-up response
                            logger.info(f"Received follow-up response: {json.dumps(follow_up_response.get('message', {}), indent=2)}")"""

followup_log_fixed = """                            # Log the follow-up response (sanitized)
                            try:
                                logger.info(f"Received follow-up response: {json.dumps(sanitize_for_json(follow_up_response.get('message', {})), indent=2)}")
                            except Exception as log_err:
                                logger.error(f"Error logging follow-up response: {log_err}")
                                logger.info(f"Follow-up response received but could not be serialized")"""

modified_content = modified_content.replace(followup_log_original, followup_log_fixed)
logger.info("Updated follow-up response logging to handle serialization errors")

# Fix the model selection issue
model_selection_original = """                    # Use the model name directly from session state
                    model_to_use = st.session_state.selected_tool_model
                    
                    # DEBUG: Log which model we're actually using
                    logger.info(f"Using model for tool call: {model_to_use}")"""

model_selection_fixed = """                    # Use the model name directly from session state
                    model_to_use = st.session_state.selected_tool_model
                    
                    # Save the model specifically for this request to ensure consistency
                    # This prevents other UI components from changing the model mid-request
                    st.session_state.last_tool_call_model = model_to_use
                    
                    # DEBUG: Log which model we're actually using for better diagnostics
                    logger.info(f"Using model for tool call: {model_to_use}")
                    
                    # Double-check that it's actually installed
                    try:
                        available = get_available_models()
                        if model_to_use not in available:
                            logger.warning(f"Model {model_to_use} not found in available models: {available}")
                            # Fall back to a model that exists
                            for fallback in ['llama3', 'mistral', 'mixtral']:
                                if fallback in available:
                                    logger.info(f"Falling back to model: {fallback}")
                                    model_to_use = fallback
                                    break
                    except Exception as model_check_error:
                        logger.error(f"Error checking available models: {model_check_error}")"""

modified_content = modified_content.replace(model_selection_original, model_selection_fixed)
logger.info("Updated model selection to be more robust and handle fallbacks")

# Fix the follow-up model selection to use the same model
followup_model_original = """                                model=model_to_use,  # Use the same model we used for the initial request"""

followup_model_fixed = """                                model=st.session_state.last_tool_call_model or model_to_use,  # Ensure consistent model between requests"""

modified_content = modified_content.replace(followup_model_original, followup_model_fixed)
logger.info("Updated follow-up model selection to ensure consistency")

# Write the modified content back to the file
with open(TOOL_PLAYGROUND_PATH, "w") as f:
    f.write(modified_content)

logger.info(f"Successfully updated {TOOL_PLAYGROUND_PATH}")
logger.info(f"Original file backed up at {backup_path}")
logger.info("The tool playground should now handle JSON serialization properly")
logger.info("Run the test_tool_calling.py script to verify the fix works")