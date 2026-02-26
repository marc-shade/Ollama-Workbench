"""
Chat Workspace Integration for Ollama Workbench

This module provides integration between the chat interface and the canvas system,
allowing chat-generated content to be stored and edited in a Canvas-like workspace.
"""

import streamlit as st
import json
import os
import re
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from canvas import DocumentState, BlockType

logger = logging.getLogger(__name__)

def extract_content_blocks(text: str) -> tuple:
    """
    Extract code and article blocks from text.
    
    Args:
        text: The text to extract blocks from
        
    Returns:
        Tuple of (code_blocks, article_blocks)
    """
    if text is None:
        return [], []
    
    code_blocks = re.findall(r'```[\s\S]*?```', text)
    article_blocks = re.findall(r'^Title:.*?(?=\n^Title:|\Z)', text, re.MULTILINE | re.DOTALL)
    
    return [block.strip('`').strip() for block in code_blocks], [block.strip() for block in article_blocks]

def process_ai_response(response_text: str, doc_state: DocumentState) -> int:
    """
    Process an AI response and extract content to add to the document.
    
    Args:
        response_text: The text response from the AI
        doc_state: The document state to add content to
        
    Returns:
        int: Number of blocks added to the document
    """
    logger.info("CHECKPOINT: Starting process_ai_response")
    blocks_added = 0
    
    # Extract code blocks and article blocks
    code_blocks, article_blocks = extract_content_blocks(response_text)
    logger.info(f"CHECKPOINT: Extracted {len(code_blocks)} code blocks and {len(article_blocks)} article blocks")
    
    # Add code blocks to document
    for i, code_block in enumerate(code_blocks):
        # Try to detect language from code fence
        language = "python"  # Default
        language_match = re.search(r'```(\w+)', code_block)
        if language_match:
            language = language_match.group(1)
            logger.info(f"CHECKPOINT: Detected language for code block {i+1}: {language}")
        
        # Clean code (remove markdown code fences)
        clean_code = re.sub(r'```\w+\n', '', code_block)
        clean_code = re.sub(r'```', '', clean_code)
        
        # Add to document
        doc_state.add_block(
            BlockType.CODE,
            clean_code,
            {"language": language}
        )
        blocks_added += 1
        logger.info(f"CHECKPOINT: Added code block {i+1} to workspace")
    
    # Add article blocks to document
    for i, article_block in enumerate(article_blocks):
        lines = article_block.split('\n')
        if lines and lines[0].startswith("Title:"):
            # Format as markdown with title
            title = lines[0].replace("Title:", "").strip()
            content = "\n".join(lines[1:])
            logger.info(f"CHECKPOINT: Found titled article block: {title}")
            
            doc_state.add_block(
                BlockType.MARKDOWN,
                f"# {title}\n\n{content}",
                {"original_title": title}
            )
            blocks_added += 1
            logger.info(f"CHECKPOINT: Added titled markdown block {i+1} to workspace")
        else:
            # Just add as plain markdown
            doc_state.add_block(BlockType.MARKDOWN, article_block)
            blocks_added += 1
            logger.info(f"CHECKPOINT: Added plain markdown block {i+1} to workspace")
    
    logger.info(f"CHECKPOINT: Total blocks added to workspace: {blocks_added}")
    return blocks_added

def ensure_chat_workspace(chat_id: str = None) -> DocumentState:
    """
    Ensure a workspace document exists for the current chat session.
    
    Args:
        chat_id: Optional chat ID to use for the workspace
        
    Returns:
        DocumentState instance for the chat workspace
    """
    # Use provided chat_id or generate from session state
    if not chat_id:
        if "chat_workspace_id" not in st.session_state:
            # Generate a unique ID for this chat session
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.chat_workspace_id = f"chat_{timestamp}"
        
        chat_id = st.session_state.chat_workspace_id
    
    # Create or load document state
    doc_state = DocumentState(chat_id, f"Chat Workspace {chat_id}")
    
    return doc_state

def chat_workspace_ui() -> None:
    """
    Render the chat workspace UI.
    
    This is a simplified version of the canvas UI specifically for the chat interface.
    """
    # Initialize session state if needed
    if "workspace_initialized" not in st.session_state:
        st.session_state.workspace_initialized = True
        st.session_state.clear_custom_block = False
    # Get the document state for this chat session
    doc_state = ensure_chat_workspace()
    
    # Canvas-style workspace view
    if not doc_state.blocks:
        st.info("This workspace is empty. Content from AI responses will appear here.")
    
    # Render all blocks in the document
    for i, block in enumerate(doc_state.blocks):
        with st.expander(f"Item {i+1}: {block['type'].capitalize()} - {datetime.fromisoformat(block['updated_at']).strftime('%Y-%m-%d %H:%M')}", expanded=False):
            # Block content
            if block["type"] == BlockType.CODE.value:
                language = block.get("metadata", {}).get("language", "python")
                st.code(block["content"], language=language)
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Edit", key=f"edit_{block['id']}"):
                        st.session_state[f"edit_block_{block['id']}"] = True
                with col2:
                    if st.button("Remove", key=f"remove_{block['id']}"):
                        doc_state.delete_block(block["id"])
                        st.success(f"Removed item {i+1}")
                        st.rerun()
                with col3:
                    if st.button("Copy to Clipboard", key=f"copy_{block['id']}"):
                        # This is just a placeholder - in a real app you'd need JavaScript
                        st.info("Code copied to clipboard")
                
                # Edit mode
                if st.session_state.get(f"edit_block_{block['id']}", False):
                    new_content = st.text_area(
                        "Edit code:",
                        value=block["content"],
                        height=min(100 + (block["content"].count('\n') * 20), 400),
                        key=f"edit_content_{block['id']}"
                    )
                    
                    # Language selection
                    new_language = st.selectbox(
                        "Language:",
                        ["python", "javascript", "java", "c", "cpp", "csharp", "go", "rust", "html", "css", "bash", "sql", "json"],
                        index=["python", "javascript", "java", "c", "cpp", "csharp", "go", "rust", "html", "css", "bash", "sql", "json"].index(language) 
                              if language in ["python", "javascript", "java", "c", "cpp", "csharp", "go", "rust", "html", "css", "bash", "sql", "json"] else 0,
                        key=f"edit_language_{block['id']}"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Save", key=f"save_edit_{block['id']}"):
                            # Update block
                            doc_state.update_block(
                                block["id"],
                                new_content,
                                {"language": new_language}
                            )
                            st.session_state[f"edit_block_{block['id']}"] = False
                            st.success("Changes saved")
                            st.rerun()
                    
                    with col2:
                        if st.button("Cancel", key=f"cancel_edit_{block['id']}"):
                            st.session_state[f"edit_block_{block['id']}"] = False
                            st.rerun()
            
            elif block["type"] == BlockType.MARKDOWN.value:
                # Display markdown
                st.markdown(block["content"])
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Edit", key=f"edit_{block['id']}"):
                        st.session_state[f"edit_block_{block['id']}"] = True
                with col2:
                    if st.button("Remove", key=f"remove_{block['id']}"):
                        doc_state.delete_block(block["id"])
                        st.success(f"Removed item {i+1}")
                        st.rerun()
                
                # Edit mode
                if st.session_state.get(f"edit_block_{block['id']}", False):
                    new_content = st.text_area(
                        "Edit markdown:",
                        value=block["content"],
                        height=min(100 + (block["content"].count('\n') * 20), 400),
                        key=f"edit_content_{block['id']}"
                    )
                    
                    # Preview
                    st.subheader("Preview")
                    st.markdown(new_content)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Save", key=f"save_edit_{block['id']}"):
                            # Update block
                            doc_state.update_block(block["id"], new_content)
                            st.session_state[f"edit_block_{block['id']}"] = False
                            st.success("Changes saved")
                            st.rerun()
                    
                    with col2:
                        if st.button("Cancel", key=f"cancel_edit_{block['id']}"):
                            st.session_state[f"edit_block_{block['id']}"] = False
                            st.rerun()
            
            else:  # TEXT or other types
                st.write(block["content"])
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Edit", key=f"edit_{block['id']}"):
                        st.session_state[f"edit_block_{block['id']}"] = True
                with col2:
                    if st.button("Remove", key=f"remove_{block['id']}"):
                        doc_state.delete_block(block["id"])
                        st.success(f"Removed item {i+1}")
                        st.rerun()
                
                # Edit mode
                if st.session_state.get(f"edit_block_{block['id']}", False):
                    new_content = st.text_area(
                        "Edit text:",
                        value=block["content"],
                        height=min(100 + (block["content"].count('\n') * 20), 400),
                        key=f"edit_content_{block['id']}"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Save", key=f"save_edit_{block['id']}"):
                            # Update block
                            doc_state.update_block(block["id"], new_content)
                            st.session_state[f"edit_block_{block['id']}"] = False
                            st.success("Changes saved")
                            st.rerun()
                    
                    with col2:
                        if st.button("Cancel", key=f"cancel_edit_{block['id']}"):
                            st.session_state[f"edit_block_{block['id']}"] = False
                            st.rerun()
    
    # Add custom content with proper form approach
    st.subheader("➕ Add Custom Content")
    # Use a form pattern to avoid session state issues
    with st.form(key="add_content_form"):
        # Content type selection
        block_type = st.selectbox(
            "Content Type:",
            ["text", "code", "markdown"]
        )
        
        # Language selection for code blocks
        language = None
        if block_type == "code":
            language = st.selectbox(
                "Language:",
                ["python", "javascript", "java", "c", "cpp", "csharp", "go", "rust", "html", "css", "bash", "sql", "json"],
                index=0
            )
        
        # Content input
        custom_content = st.text_area(
            f"Enter {block_type} content:",
            height=200
        )
        
        # Form submission button
        submitted = st.form_submit_button("Add to Workspace")
            
    # Process form submission outside of the form
    if submitted:
        if custom_content:
            # Create metadata based on block type
            metadata = {}
            if block_type == "code" and language:
                metadata["language"] = language
            
            # Add to document
            doc_state.add_block(block_type, custom_content, metadata)
            
            # Display success message
            st.success(f"Added new {block_type} block!")
        else:
            st.warning("Please enter some content.")

def save_ai_content_to_workspace(content: str) -> bool:
    """
    Save AI-generated content to the workspace.
    
    Args:
        content: The content to save
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("CHECKPOINT: Starting save_ai_content_to_workspace")
    
    if not content or not isinstance(content, str):
        logger.warning(f"CHECKPOINT: Invalid content type: {type(content)}")
        return False
    
    try:
        # Get document state
        doc_state = ensure_chat_workspace()
        logger.info(f"CHECKPOINT: Got document state with ID {doc_state.doc_id}")
        
        # Process the content
        blocks_added = process_ai_response(content, doc_state)
        logger.info(f"CHECKPOINT: Processed AI response, added {blocks_added} blocks")
        
        # Log success
        logger.info(f"CHECKPOINT: Successfully saved AI content to workspace {doc_state.doc_id}")
        return True
    except Exception as e:
        logger.error(f"CHECKPOINT: Error saving AI content to workspace: {str(e)}")
        logger.exception(e)
        return False

# For testing the module directly
if __name__ == "__main__":
    st.title("Chat Workspace Test")
    
    # Test content
    test_content = '''
    Here\'s an example of how to implement a simple web server in Python:

    ```python
    from flask import Flask, jsonify

    app = Flask(__name__)

    @app.route('/api/hello', methods=['GET'])
    def hello_world():
        return jsonify({"message": "Hello, World!"})

    if __name__ == '__main__':
        app.run(debug=True, port=5000)
    ```

    Title: Building a RESTful API
    
    RESTful APIs follow these key principles:
    1. Client-Server architecture
    2. Statelessness
    3. Cacheability
    4. Layered system
    5. Uniform interface
    
    Make sure your API endpoints use proper HTTP methods (GET, POST, PUT, DELETE) for CRUD operations.
    '''
    
    # Test saving content
    if st.button("Test: Save AI Content"):
        save_ai_content_to_workspace(test_content)
        st.success("Test content saved to workspace")
    
    # Container for workspace UI
    with st.container():
        chat_workspace_ui()