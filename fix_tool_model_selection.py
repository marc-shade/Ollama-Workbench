#!/usr/bin/env python3
"""
Fix for the Tool Playground model selection issue.

This script fixes the issue where the model selection in Tool Playground
doesn't persist when a new model is selected from the dropdown.
"""

import logging
import os

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_tool_model_selection")

def fix_tool_model_selection():
    """Fix the model selection issue in tool_playground.py by adding session state persistence."""
    try:
        file_path = "tool_playground.py"
        
        # Make sure file exists
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} does not exist")
            return False
        
        # Read the file
        with open(file_path, "r") as f:
            content = f.read()
        
        # Add session state initialization for the selected tool model
        if "selected_tool_model" not in content:
            # Add session state initialization at the beginning of the file
            init_section = "# Initialize session state with defensive programming\n    if \"tool_chat_history\" not in st.session_state:"
            init_code = "# Initialize session state with defensive programming\n    if \"tool_chat_history\" not in st.session_state:\n        st.session_state.tool_chat_history = []\n    if \"selected_tool_model\" not in st.session_state:\n        st.session_state.selected_tool_model = \"\""
            
            content = content.replace(init_section, init_code)
            
            logger.info("Added selected_tool_model session state initialization")
        
        # Modify the model selection logic
        # Look for both versions of model selection
        if "model_options.index(st.session_state.selected_tool_model)" not in content:
            # Fix the model selection in the first case (with tool capabilities)
            if "selected_model_display = st.selectbox(\n                    \"Select Model:\"," in content:
                # Find the section with the model selection
                selection_code = "                # Model selection\n                selected_model_display = st.selectbox(\n                    \"Select Model:\",\n                    model_options,\n                    index=0 if model_options else 0,\n                    key=\"tool_model_selector\"\n                )"
                
                # Replace with improved version
                fixed_code = """                # Ensure we have a session state variable for the selected tool model
                if "selected_tool_model" not in st.session_state:
                    st.session_state.selected_tool_model = model_options[0] if model_options else "llama3"
                
                # Function to update the selected model
                def update_selected_tool_model():
                    # Remove the suffix if present
                    selected_display = st.session_state.tool_model_selector
                    if " (tools)" in selected_display:
                        st.session_state.selected_tool_model = selected_display.split(" (tools)")[0]
                    elif " (likely tools)" in selected_display:
                        st.session_state.selected_tool_model = selected_display.split(" (likely tools)")[0]
                    else:
                        st.session_state.selected_tool_model = selected_display
                
                # Model selection with persistence
                selected_model_display = st.selectbox(
                    "Select Model:",
                    model_options,
                    index=model_options.index(st.session_state.selected_tool_model) if st.session_state.selected_tool_model in model_options else 0,
                    key="tool_model_selector",
                    on_change=update_selected_tool_model
                )
                
                # Use the selected model from session state
                selected_model = st.session_state.selected_tool_model"""
                
                content = content.replace(selection_code, fixed_code)
                logger.info("Updated tool capabilities model selection")
            
            # Fix the standard model selection (without tool capabilities)
            if "selected_model = st.selectbox(\n                        \"Select Model:\"," in content:
                selection_code2 = "                    selected_model = st.selectbox(\n                        \"Select Model:\",\n                        available_models,\n                        index=0 if available_models else 0,\n                        key=\"tool_model_selector\"\n                    )"
                
                fixed_code2 = """                    # Ensure we have a session state variable for the selected tool model
                    if "selected_tool_model" not in st.session_state:
                        st.session_state.selected_tool_model = available_models[0] if available_models else "llama3"
                    
                    # Function to update the selected model
                    def update_selected_tool_model():
                        st.session_state.selected_tool_model = st.session_state.tool_model_selector
                    
                    # Model selection with persistence
                    selected_model = st.selectbox(
                        "Select Model:",
                        available_models,
                        index=available_models.index(st.session_state.selected_tool_model) if st.session_state.selected_tool_model in available_models else 0,
                        key="tool_model_selector",
                        on_change=update_selected_tool_model
                    )"""
                
                content = content.replace(selection_code2, fixed_code2)
                logger.info("Updated standard model selection")
        else:
            logger.info("Model selection already fixed")
        
        # Write the updated content back to the file
        with open(file_path, "w") as f:
            f.write(content)
        
        logger.info("Successfully updated tool_playground.py with model selection persistence")
        return True
    except Exception as e:
        logger.error(f"Error fixing model selection in tool_playground.py: {e}")
        return False

if __name__ == "__main__":
    success = fix_tool_model_selection()
    if success:
        print("Successfully fixed Tool Playground model selection")
    else:
        print("Failed to fix Tool Playground model selection")