#!/usr/bin/env python3
"""
Fix for the Tool Playground module to prevent session state conflicts.

This module fixes the StreamlitAPIException that occurs when trying to modify
st.session_state.tool_prompt after a widget with key tool_prompt is instantiated.
"""

import logging
import re

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_tool_playground")

def fix_tool_playground():
    """Fix the session state conflict in tool_playground.py."""
    try:
        file_path = "tool_playground.py"
        
        with open(file_path, "r") as f:
            content = f.read()
        
        # Replace all instances of st.session_state.tool_prompt with st.session_state.selected_tool_prompt
        # except in the chat_input widget definition
        new_content = re.sub(
            r'st\.session_state\.tool_prompt(\s*=\s*|\s*\))',
            r'st.session_state.selected_tool_prompt\1',
            content
        )
        
        # Update the condition checking for hasattr(st.session_state, "tool_prompt")
        new_content = re.sub(
            r'hasattr\(st\.session_state,\s*"tool_prompt"\)',
            r'hasattr(st.session_state, "selected_tool_prompt")',
            new_content
        )
        
        # Write the updated content back to the file
        with open(file_path, "w") as f:
            f.write(new_content)
        
        logger.info("Successfully fixed tool_playground.py session state conflict")
        return True
    except Exception as e:
        logger.error(f"Error fixing tool_playground.py: {e}")
        return False

if __name__ == "__main__":
    success = fix_tool_playground()
    if success:
        print("Successfully fixed Tool Playground session state conflict")
    else:
        print("Failed to fix Tool Playground session state conflict")