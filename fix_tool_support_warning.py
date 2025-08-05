#!/usr/bin/env python3
"""
Fix for tool support detection and warnings in Tool Playground.

This module improves how the Tool Playground handles models that don't support tools,
providing better error messages and proactive warnings.
"""

import logging
import re

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_tool_support_warning")

def fix_tool_support_warnings():
    """Improve error handling and warnings for tool support in tool_playground.py."""
    try:
        file_path = "tool_playground.py"
        
        # Read the current content
        with open(file_path, "r") as f:
            content = f.read()
        
        # Find and replace the simple tool capability check with a more robust one
        simple_check_pattern = r"""# Check if model is known to support tools before sending request
\s+if not is_tools_capable\(selected_model\):
\s+# Model is not in our tools-capable list, show a warning
\s+logger\.warning\(f"Model '\{selected_model\}' is not known to support tools. Attempting anyway but may fail."\)
\s+st\.warning\(f"Model '\{selected_model\}' is not officially known to support tools. We'll try anyway, but it may not work correctly."\)"""
        
        robust_check = """# Check if model is known to support tools before sending request
                    try:
                        if not is_tools_capable(selected_model):
                            # Model is not in our tools-capable list, show a warning with more information
                            logger.warning(f"Model '{selected_model}' is not known to support tools. Attempting anyway but may fail.")
                            
                            warning_msg = f\"\"\"
                            ⚠️ **Important:** Model '{selected_model}' is not officially known to support tools.
                            
                            This may result in an error. For best results, use a model designed for function calling:
                            - llama3 (best support)
                            - mistral
                            - qwen
                            - phi3
                            
                            Documentation: [Ollama Function Calling Models](https://ollama.com/search?c=tools)
                            \"\"\"
                            st.warning(warning_msg)
                    except Exception as check_error:
                        # If the capability check fails, log it but continue
                        logger.error(f"Error checking tool capability: {check_error}")
                        # Don't show a warning to avoid confusing the user"""
                        
        content = re.sub(simple_check_pattern, robust_check, content)
        
        # Enhance the error handling for "does not support tools" errors
        error_pattern = r"""# Check for common tool-related error messages
\s+if "does not support tools" in error_message\.lower\(\) or "function calling" in error_message\.lower\(\):
\s+# This is a tool support error
\s+friendly_error = f"Error: This model does not support tool/function calling\. Please select a different model like llama3, mistral, or qwen that supports tools\."
\s+message_placeholder\.error\(friendly_error\)
\s+
\s+# Log the error for diagnostics
\s+logger\.error\(f"Tool support error with model '\{selected_model\}': \{error_message\}"\)
\s+
\s+# Add error message to chat history
\s+st\.session_state\.tool_chat_history\.append\(\{
\s+"role": "assistant",
\s+"content": friendly_error
\s+\}\)"""
        
        enhanced_error = """# Check for common tool-related error messages
                    if "does not support tools" in error_message.lower() or "function calling" in error_message.lower() or "status code: 400" in error_message.lower():
                        # This is a tool support error
                        friendly_error = f"Error: Model '{selected_model}' does not support tool/function calling."
                        
                        # Provide more helpful instructions
                        recommendations = \"\"\"
                        To use tools/function calling with Ollama, you need a model that supports this capability.
                        
                        Recommended models:
                        - llama3 (best support for tools)
                        - mistral
                        - qwen
                        - phi3
                        
                        You can pull one of these models using: `ollama pull llama3`
                        
                        Learn more: https://ollama.com/search?c=tools
                        \"\"\"
                        
                        message_placeholder.error(friendly_error)
                        message_placeholder.info(recommendations)
                        
                        # Log the error for diagnostics
                        logger.error(f"Tool support error with model '{selected_model}': {error_message}")
                        
                        # Add error message to chat history
                        st.session_state.tool_chat_history.append({
                            "role": "assistant",
                            "content": f"{friendly_error}\\n\\n{recommendations}"
                        })"""
        
        content = re.sub(error_pattern, enhanced_error, content)
        
        # Write the updated content back to the file
        with open(file_path, "w") as f:
            f.write(content)
        
        logger.info("Successfully fixed tool support warnings in tool_playground.py")
        return True
    except Exception as e:
        logger.error(f"Error fixing tool support warnings: {e}")
        return False

if __name__ == "__main__":
    success = fix_tool_support_warnings()
    if success:
        print("Successfully fixed tool support warnings")
    else:
        print("Failed to fix tool support warnings")