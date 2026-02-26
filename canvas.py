"""
Canvas Module for Ollama Workbench

This module provides a Canvas-like collaborative document editing system where both the user and AI 
can edit content in real-time, similar to the Canvas feature in ChatGPT and Claude.
"""

import streamlit as st
import json
import os
import re
import time
import difflib
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

# Constants
CANVAS_DIR = "canvas"
os.makedirs(CANVAS_DIR, exist_ok=True)

class EditOperation(Enum):
    """Types of edit operations that can be performed on a document."""
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"

class EditStatus(Enum):
    """Status of an edit suggestion."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"

class BlockType(Enum):
    """Types of content blocks that can be created."""
    TEXT = "text"
    CODE = "code"
    MARKDOWN = "markdown"
    TABLE = "table"
    IMAGE = "image"

class DocumentState:
    """
    Manages the state of a document in the collaborative canvas.
    
    This class handles:
    - Document content storage and retrieval
    - Version history tracking
    - Edit suggestions from AI
    - Conflict resolution
    - Real-time synchronization
    """
    
    def __init__(self, doc_id: str = None, name: str = "Untitled Document"):
        """
        Initialize a new document state.
        
        Args:
            doc_id: Unique identifier for the document (generated if None)
            name: Human-readable name for the document
        """
        self.doc_id = doc_id or str(uuid.uuid4())
        self.name = name
        self.blocks = []
        self.version = 0
        self.edit_history = []
        self.pending_ai_edits = []
        self.last_modified = datetime.now().isoformat()
        self.created_at = datetime.now().isoformat()
        self.metadata = {}
        
        # Load existing document if doc_id is provided and exists
        if doc_id:
            self.load()
    
    def add_block(self, block_type: Union[BlockType, str], content: str = "", metadata: Dict = None) -> str:
        """
        Add a new block to the document.
        
        Args:
            block_type: Type of block (text, code, markdown, etc.)
            content: Initial content for the block
            metadata: Additional metadata for the block (e.g., language for code blocks)
            
        Returns:
            The ID of the newly created block
        """
        # Convert string to enum if needed
        if isinstance(block_type, str):
            try:
                block_type = BlockType[block_type.upper()]
            except KeyError:
                block_type = BlockType.TEXT
        
        # Create block with unique ID
        block_id = str(uuid.uuid4())
        
        new_block = {
            "id": block_id,
            "type": block_type.value,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version_history": []
        }
        
        # Add to document
        self.blocks.append(new_block)
        self.version += 1
        self.last_modified = datetime.now().isoformat()
        
        # Record in edit history
        self._add_to_history("add_block", block_id, metadata={
            "block_type": block_type.value,
            "initial_content": content
        })
        
        # Save document
        self.save()
        
        return block_id
    
    def update_block(self, block_id: str, new_content: str, metadata: Dict = None) -> bool:
        """
        Update the content of an existing block.
        
        Args:
            block_id: ID of the block to update
            new_content: New content for the block
            metadata: Optional metadata to update
            
        Returns:
            True if block was found and updated, False otherwise
        """
        # Find the block
        for block in self.blocks:
            if block["id"] == block_id:
                # Store previous version in history
                block["version_history"].append({
                    "content": block["content"],
                    "updated_at": block["updated_at"],
                    "metadata": block.get("metadata", {})
                })
                
                # Limit version history to last 10 versions
                if len(block["version_history"]) > 10:
                    block["version_history"] = block["version_history"][-10:]
                
                # Update block
                old_content = block["content"]
                block["content"] = new_content
                block["updated_at"] = datetime.now().isoformat()
                
                # Update metadata if provided
                if metadata:
                    if "metadata" not in block:
                        block["metadata"] = {}
                    block["metadata"].update(metadata)
                
                # Update document
                self.version += 1
                self.last_modified = datetime.now().isoformat()
                
                # Record in edit history
                self._add_to_history("update_block", block_id, metadata={
                    "old_content": old_content,
                    "new_content": new_content,
                    "diff": self.generate_diff(old_content, new_content)
                })
                
                # Save document
                self.save()
                
                return True
        
        return False
    
    def delete_block(self, block_id: str) -> bool:
        """
        Delete a block from the document.
        
        Args:
            block_id: ID of the block to delete
            
        Returns:
            True if block was found and deleted, False otherwise
        """
        for i, block in enumerate(self.blocks):
            if block["id"] == block_id:
                # Remove the block
                deleted_block = self.blocks.pop(i)
                
                # Update document
                self.version += 1
                self.last_modified = datetime.now().isoformat()
                
                # Record in edit history
                self._add_to_history("delete_block", block_id, metadata={
                    "block_type": deleted_block["type"],
                    "content": deleted_block["content"]
                })
                
                # Save document
                self.save()
                
                return True
        
        return False
    
    def move_block(self, block_id: str, new_position: int) -> bool:
        """
        Move a block to a new position in the document.
        
        Args:
            block_id: ID of the block to move
            new_position: New index for the block
            
        Returns:
            True if block was found and moved, False otherwise
        """
        # Find the block
        block_index = None
        for i, block in enumerate(self.blocks):
            if block["id"] == block_id:
                block_index = i
                break
        
        if block_index is None:
            return False
        
        # Ensure new_position is within bounds
        if new_position < 0:
            new_position = 0
        elif new_position >= len(self.blocks):
            new_position = len(self.blocks) - 1
        
        # Move the block
        if block_index != new_position:
            block = self.blocks.pop(block_index)
            self.blocks.insert(new_position, block)
            
            # Update document
            self.version += 1
            self.last_modified = datetime.now().isoformat()
            
            # Record in edit history
            self._add_to_history("move_block", block_id, metadata={
                "old_position": block_index,
                "new_position": new_position
            })
            
            # Save document
            self.save()
            
            return True
        
        return False
    
    def suggest_ai_edit(self, block_id: str, new_content: str, edit_type: EditOperation, 
                     explanation: str = "") -> str:
        """
        Add a suggested edit from the AI.
        
        Args:
            block_id: ID of the block to edit
            new_content: New content for the block
            edit_type: Type of edit (insert, delete, replace)
            explanation: Explanation for the edit
            
        Returns:
            The ID of the suggestion
        """
        # Get current content for diff
        current_content = ""
        block_type = None
        for block in self.blocks:
            if block["id"] == block_id:
                current_content = block["content"]
                block_type = block["type"]
                break
        
        # Create suggestion
        suggestion_id = str(uuid.uuid4())
        
        suggestion = {
            "id": suggestion_id,
            "block_id": block_id,
            "block_type": block_type,
            "current_content": current_content,
            "new_content": new_content,
            "edit_type": edit_type.value,
            "explanation": explanation,
            "created_at": datetime.now().isoformat(),
            "status": EditStatus.PENDING.value,
            "diff": self.generate_diff(current_content, new_content)
        }
        
        # Add to pending edits
        self.pending_ai_edits.append(suggestion)
        
        # Save document
        self.save()
        
        return suggestion_id
    
    def apply_ai_suggestion(self, suggestion_id: str) -> bool:
        """
        Apply a pending AI suggestion.
        
        Args:
            suggestion_id: ID of the suggestion to apply
            
        Returns:
            True if suggestion was found and applied, False otherwise
        """
        # Find the suggestion
        suggestion = None
        for edit in self.pending_ai_edits:
            if edit["id"] == suggestion_id:
                suggestion = edit
                break
        
        if suggestion is None:
            return False
        
        # Check if suggestion is pending
        if suggestion["status"] != EditStatus.PENDING.value:
            return False
        
        # Apply the suggestion based on edit type
        if suggestion["edit_type"] == EditOperation.REPLACE.value:
            success = self.update_block(
                suggestion["block_id"], 
                suggestion["new_content"]
            )
        elif suggestion["edit_type"] == EditOperation.INSERT.value:
            # For an insert, we'll add the new content to the existing content
            for block in self.blocks:
                if block["id"] == suggestion["block_id"]:
                    success = self.update_block(
                        suggestion["block_id"],
                        block["content"] + suggestion["new_content"]
                    )
                    break
        elif suggestion["edit_type"] == EditOperation.DELETE.value:
            # For delete, the new_content should be an empty string or the content to keep
            success = self.update_block(
                suggestion["block_id"],
                suggestion["new_content"]
            )
        
        # Update suggestion status
        suggestion["status"] = EditStatus.ACCEPTED.value
        
        # Save document
        self.save()
        
        return success
    
    def reject_ai_suggestion(self, suggestion_id: str) -> bool:
        """
        Reject a pending AI suggestion.
        
        Args:
            suggestion_id: ID of the suggestion to reject
            
        Returns:
            True if suggestion was found and rejected, False otherwise
        """
        # Find the suggestion
        for edit in self.pending_ai_edits:
            if edit["id"] == suggestion_id:
                # Update status
                edit["status"] = EditStatus.REJECTED.value
                
                # Save document
                self.save()
                
                return True
        
        return False
    
    def get_block(self, block_id: str) -> Optional[Dict]:
        """
        Get a block by ID.
        
        Args:
            block_id: ID of the block to retrieve
            
        Returns:
            The block dictionary or None if not found
        """
        for block in self.blocks:
            if block["id"] == block_id:
                return block
        return None
    
    def get_document_content(self) -> str:
        """
        Get the full document content as a string.
        
        Returns:
            The document content with all blocks concatenated
        """
        content = []
        for block in self.blocks:
            if block["type"] == BlockType.CODE.value:
                lang = block.get("metadata", {}).get("language", "")
                content.append(f"```{lang}\n{block['content']}\n```")
            elif block["type"] == BlockType.MARKDOWN.value:
                content.append(block["content"])
            else:
                content.append(block["content"])
        
        return "\n\n".join(content)
    
    def save(self) -> None:
        """Save the document state to disk."""
        file_path = os.path.join(CANVAS_DIR, f"{self.doc_id}.json")
        
        doc_data = {
            "doc_id": self.doc_id,
            "name": self.name,
            "blocks": self.blocks,
            "version": self.version,
            "edit_history": self.edit_history,
            "pending_ai_edits": self.pending_ai_edits,
            "last_modified": self.last_modified,
            "created_at": self.created_at,
            "metadata": self.metadata
        }
        
        with open(file_path, "w") as f:
            json.dump(doc_data, f, indent=2)
    
    def load(self) -> bool:
        """
        Load the document state from disk.
        
        Returns:
            True if document was loaded successfully, False otherwise
        """
        file_path = os.path.join(CANVAS_DIR, f"{self.doc_id}.json")
        
        if not os.path.exists(file_path):
            return False
        
        try:
            with open(file_path, "r") as f:
                doc_data = json.load(f)
            
            self.name = doc_data.get("name", self.name)
            self.blocks = doc_data.get("blocks", [])
            self.version = doc_data.get("version", 0)
            self.edit_history = doc_data.get("edit_history", [])
            self.pending_ai_edits = doc_data.get("pending_ai_edits", [])
            self.last_modified = doc_data.get("last_modified", self.last_modified)
            self.created_at = doc_data.get("created_at", self.created_at)
            self.metadata = doc_data.get("metadata", {})
            
            return True
        except Exception as e:
            logger.error(f"Error loading document {self.doc_id}: {e}")
            return False
    
    def _add_to_history(self, action: str, block_id: str, metadata: Dict = None) -> None:
        """
        Add an entry to the document edit history.
        
        Args:
            action: Type of action performed
            block_id: ID of the block affected
            metadata: Additional information about the edit
        """
        entry = {
            "action": action,
            "block_id": block_id,
            "timestamp": datetime.now().isoformat(),
            "version": self.version,
            "metadata": metadata or {}
        }
        
        self.edit_history.append(entry)
        
        # Limit history to last 100 edits
        if len(self.edit_history) > 100:
            self.edit_history = self.edit_history[-100:]
    
    @staticmethod
    def generate_diff(old_content: str, new_content: str) -> str:
        """
        Generate a unified diff between old and new content.
        
        Args:
            old_content: Original content
            new_content: Modified content
            
        Returns:
            String representation of the diff
        """
        diff = difflib.unified_diff(
            old_content.splitlines(),
            new_content.splitlines(),
            lineterm='',
            n=3  # Context lines
        )
        return '\n'.join(diff)
    
    @staticmethod
    def highlight_diff_html(old_content: str, new_content: str) -> str:
        """
        Generate HTML with highlighted differences.
        
        Args:
            old_content: Original content
            new_content: Modified content
            
        Returns:
            HTML string with highlighted differences
        """
        differ = difflib.HtmlDiff()
        return differ.make_file(
            old_content.splitlines(),
            new_content.splitlines(),
            context=True,
            numlines=3
        )

def get_available_documents() -> List[Dict]:
    """
    Get list of available documents in the canvas directory.
    
    Returns:
        List of document metadata dictionaries
    """
    documents = []
    
    try:
        for file in os.listdir(CANVAS_DIR):
            if file.endswith(".json"):
                file_path = os.path.join(CANVAS_DIR, file)
                
                try:
                    with open(file_path, "r") as f:
                        doc_data = json.load(f)
                    
                    # Extract basic metadata
                    documents.append({
                        "doc_id": doc_data.get("doc_id", file.split(".")[0]),
                        "name": doc_data.get("name", "Untitled"),
                        "last_modified": doc_data.get("last_modified", ""),
                        "block_count": len(doc_data.get("blocks", [])),
                        "created_at": doc_data.get("created_at", "")
                    })
                except Exception as e:
                    logger.error(f"Error reading document file {file}: {e}")
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
    
    # Sort by last modified (newest first)
    documents.sort(key=lambda x: x["last_modified"], reverse=True)
    return documents

def delete_document(doc_id: str) -> bool:
    """
    Delete a document.
    
    Args:
        doc_id: ID of the document to delete
        
    Returns:
        True if document was deleted successfully, False otherwise
    """
    file_path = os.path.join(CANVAS_DIR, f"{doc_id}.json")
    
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
    
    return False

def render_document_ui(doc_state: DocumentState, model_callback: Callable = None) -> None:
    """
    Render the document UI in Streamlit.
    
    Args:
        doc_state: DocumentState instance
        model_callback: Optional callback function for AI assistance
    """
    # Document header with name and metadata
    col1, col2 = st.columns([3, 1])
    
    with col1:
        document_name = st.text_input("Document Name:", value=doc_state.name, key="doc_name_input")
        if document_name != doc_state.name:
            doc_state.name = document_name
            doc_state.save()
    
    with col2:
        st.write(f"Version: {doc_state.version}")
        # Format the last modified date
        try:
            last_modified = datetime.fromisoformat(doc_state.last_modified)
            st.write(f"Last modified: {last_modified.strftime('%Y-%m-%d %H:%M')}")
        except:
            st.write(f"Last modified: Recently")
    
    # Create a container for all blocks
    blocks_container = st.container()
    
    # Add new block UI
    with st.expander("➕ Add New Block", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            block_type = st.selectbox(
                "Block Type:",
                [t.value for t in BlockType],
                index=0,
                key="new_block_type"
            )
        
        with col2:
            if block_type == BlockType.CODE.value:
                language = st.selectbox(
                    "Language:",
                    ["python", "javascript", "java", "c", "cpp", "csharp", "go", "rust", "html", "css", "bash", "sql", "json"],
                    index=0,
                    key="new_block_language"
                )
        
        # Block content editor
        st.write(f"Enter {block_type} content:")
        
        new_block_content = st.text_area(
            "",
            key="new_block_content",
            height=200
        )
        
        # Add button
        if st.button("Add Block"):
            if new_block_content:
                # Create metadata based on block type
                metadata = {}
                if block_type == BlockType.CODE.value:
                    metadata["language"] = language
                
                # Add the block
                doc_state.add_block(block_type, new_block_content, metadata)
                
                # Clear the input
                st.session_state.new_block_content = ""
                st.rerun()
            else:
                st.warning("Please enter some content for the block.")
    
    # Render AI Suggestions if there are any pending
    pending_suggestions = [s for s in doc_state.pending_ai_edits if s["status"] == EditStatus.PENDING.value]
    if pending_suggestions:
        with st.expander("🤖 AI Suggestions", expanded=True):
            st.write(f"You have {len(pending_suggestions)} pending AI suggestions:")
            
            for suggestion in pending_suggestions:
                with st.container():
                    st.subheader(f"Suggestion for Block {suggestion['block_type']}")
                    
                    # Show explanation
                    if suggestion["explanation"]:
                        st.write(suggestion["explanation"])
                    
                    # Show diff or preview
                    if suggestion["edit_type"] == EditOperation.REPLACE.value:
                        # For replace, show side-by-side diff
                        st.write("Changes:")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.text_area("Current Content", suggestion["current_content"], height=150)
                        with col2:
                            st.text_area("Suggested Content", suggestion["new_content"], height=150)
                    elif suggestion["edit_type"] == EditOperation.INSERT.value:
                        # For insert, show what will be added
                        st.write("Content to insert:")
                        st.text_area("", suggestion["new_content"], height=100)
                    
                    # Accept/Reject buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Accept", key=f"accept_{suggestion['id']}"):
                            doc_state.apply_ai_suggestion(suggestion["id"])
                            st.success("Suggestion applied!")
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ Reject", key=f"reject_{suggestion['id']}"):
                            doc_state.reject_ai_suggestion(suggestion["id"])
                            st.success("Suggestion rejected.")
                            st.rerun()
    
    # Render all blocks
    with blocks_container:
        if not doc_state.blocks:
            st.info("This document is empty. Add a block to get started!")
        
        for i, block in enumerate(doc_state.blocks):
            with st.container():
                # Block header with drag handle and controls
                col1, col2, col3 = st.columns([7, 2, 1])
                
                with col1:
                    st.subheader(f"Block {i+1}: {block['type'].capitalize()}")
                
                with col2:
                    # Display last updated time
                    if "updated_at" in block:
                        try:
                            dt = datetime.fromisoformat(block["updated_at"])
                            st.caption(f"Updated: {dt.strftime('%Y-%m-%d %H:%M')}")
                        except:
                            st.caption("Recently updated")
                
                with col3:
                    # Delete button
                    if st.button("🗑️", key=f"delete_block_{block['id']}"):
                        doc_state.delete_block(block["id"])
                        st.success(f"Block {i+1} deleted!")
                        st.rerun()
                
                # Block content rendering and editing
                with st.container():
                    # Get unique keys for this block
                    block_key = f"block_{block['id']}"
                    edit_key = f"edit_{block['id']}"
                    
                    # Initialize edit state if not present
                    if f"{edit_key}_active" not in st.session_state:
                        st.session_state[f"{edit_key}_active"] = False
                    
                    # Render based on block type
                    if block["type"] == BlockType.CODE.value:
                        render_code_block(block, doc_state, edit_key, block_key)
                    elif block["type"] == BlockType.MARKDOWN.value:
                        render_markdown_block(block, doc_state, edit_key, block_key)
                    elif block["type"] == BlockType.TABLE.value:
                        render_table_block(block, doc_state, edit_key, block_key)
                    else:
                        render_text_block(block, doc_state, edit_key, block_key)
                
                # Version history
                if block.get("version_history"):
                    with st.expander("Version History", expanded=False):
                        for j, version in enumerate(reversed(block["version_history"])):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                try:
                                    dt = datetime.fromisoformat(version.get("updated_at", ""))
                                    st.caption(f"Version {len(block['version_history'])-j}: {dt.strftime('%Y-%m-%d %H:%M')}")
                                except:
                                    st.caption(f"Version {len(block['version_history'])-j}")
                            
                            with col2:
                                if st.button(f"Restore", key=f"restore_{block['id']}_{j}"):
                                    # Restore this version
                                    doc_state.update_block(block["id"], version["content"])
                                    st.success("Version restored!")
                                    st.rerun()
                            
                            # Show diff when requested
                            diff_key = f"diff_{block['id']}_{j}"
                            if f"{diff_key}_show" not in st.session_state:
                                st.session_state[f"{diff_key}_show"] = False
                            
                            if st.button(f"View Diff", key=diff_key):
                                st.session_state[f"{diff_key}_show"] = not st.session_state[f"{diff_key}_show"]
                            
                            if st.session_state[f"{diff_key}_show"]:
                                diff = doc_state.generate_diff(version["content"], block["content"])
                                st.code(diff)
                
                st.divider()
    
    # AI assistance for the document
    if model_callback:
        with st.expander("🤖 AI Assistance", expanded=False):
            st.write("Ask the AI to help you with this document.")
            
            # Input for the AI
            ai_query = st.text_area("Enter your question or request:", key="ai_doc_query")
            
            if st.button("Ask AI"):
                if ai_query:
                    # Construct context from the document
                    document_context = doc_state.get_document_content()
                    
                    # Prepare the prompt
                    full_prompt = f"""
                    This is a collaborative document editor called Canvas where we're working together. 
                    Here's the current content of the document named "{doc_state.name}":
                    
                    {document_context}
                    
                    The user's request is: {ai_query}
                    
                    Please assist with this document. You can:
                    1. Suggest specific edits to existing blocks
                    2. Create new content blocks
                    3. Answer questions about the content
                    4. Explain concepts related to the document
                    
                    For any content suggestions, please mark them clearly with ### BEGIN CONTENT ### and ### END CONTENT ### tags.
                    """
                    
                    # Call the model callback function
                    with st.spinner("AI is thinking..."):
                        ai_response = model_callback(full_prompt)
                        
                        # Display AI response
                        st.markdown("### AI Response")
                        st.markdown(ai_response)
                        
                        # Extract content blocks from the AI response
                        content_blocks = re.findall(r'### BEGIN CONTENT ###(.*?)### END CONTENT ###', ai_response, re.DOTALL)
                        
                        if content_blocks:
                            st.markdown("### Suggested Content")
                            
                            for i, content in enumerate(content_blocks):
                                content = content.strip()
                                suggestion_content = st.text_area(f"Suggestion {i+1}", value=content, height=min(len(content.split('\n')) * 20 + 50, 300))
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if st.button(f"Add as New Block", key=f"add_suggestion_{i}"):
                                        # Determine block type based on content
                                        if re.search(r'```\w+', content):
                                            block_type = BlockType.CODE.value
                                            # Extract language from markdown code fence
                                            language_match = re.search(r'```(\w+)', content)
                                            language = language_match.group(1) if language_match else "python"
                                            # Remove markdown code fences
                                            clean_content = re.sub(r'```\w+\n', '', content)
                                            clean_content = re.sub(r'```', '', clean_content)
                                            metadata = {"language": language}
                                        elif '|' in content and '-|-' in content:
                                            block_type = BlockType.TABLE.value
                                            clean_content = content
                                            metadata = {}
                                        elif '#' in content or '**' in content or '*' in content:
                                            block_type = BlockType.MARKDOWN.value
                                            clean_content = content
                                            metadata = {}
                                        else:
                                            block_type = BlockType.TEXT.value
                                            clean_content = content
                                            metadata = {}
                                        
                                        # Create and add the block
                                        doc_state.add_block(block_type, clean_content, metadata)
                                        st.success(f"Added new {block_type} block!")
                                        st.rerun()
                                
                                with col2:
                                    # Suggest as replacement for existing block
                                    if doc_state.blocks:
                                        block_options = [f"Block {j+1} ({block['type']})" for j, block in enumerate(doc_state.blocks)]
                                        target_block = st.selectbox(
                                            "Select block:",
                                            block_options,
                                            key=f"replace_target_{i}"
                                        )
                                        
                                        if st.button(f"Replace Block", key=f"replace_block_{i}"):
                                            block_idx = int(target_block.split()[1]) - 1
                                            block_id = doc_state.blocks[block_idx]["id"]
                                            
                                            # Create a suggestion instead of direct replacement
                                            suggestion_id = doc_state.suggest_ai_edit(
                                                block_id=block_id,
                                                new_content=suggestion_content,
                                                edit_type=EditOperation.REPLACE,
                                                explanation="AI-suggested replacement based on your request."
                                            )
                                            
                                            st.success(f"Created suggestion to replace Block {block_idx+1}!")
                                            st.rerun()
                                
                                with col3:
                                    # Quick accept option to apply suggestion immediately
                                    if doc_state.blocks and st.button(f"Quick Apply to Selected", key=f"quick_apply_{i}"):
                                        block_idx = int(target_block.split()[1]) - 1
                                        block_id = doc_state.blocks[block_idx]["id"]
                                        
                                        # Direct update
                                        doc_state.update_block(block_id, suggestion_content)
                                        st.success(f"Updated Block {block_idx+1}!")
                                        st.rerun()
                else:
                    st.warning("Please enter a question or request for the AI.")

def render_text_block(block: Dict, doc_state: DocumentState, edit_key: str, block_key: str) -> None:
    """Render a text block with editing capabilities."""
    # Edit state
    is_editing = st.session_state.get(f"{edit_key}_active", False)
    
    # Edit and display controls
    col1, col2 = st.columns([9, 1])
    
    with col1:
        if is_editing:
            # Edit mode
            new_content = st.text_area(
                "Edit text:", 
                value=block["content"],
                key=f"{block_key}_editor",
                height=min(100 + (block["content"].count('\n') * 20), 400)
            )
            
            if st.button("Save", key=f"{block_key}_save"):
                # Update block
                doc_state.update_block(block["id"], new_content)
                st.session_state[f"{edit_key}_active"] = False
                st.rerun()
        else:
            # Display mode
            st.write(block["content"])
    
    with col2:
        if st.button("✏️" if not is_editing else "Cancel", key=f"{block_key}_edit"):
            st.session_state[f"{edit_key}_active"] = not is_editing
            st.rerun()

def render_code_block(block: Dict, doc_state: DocumentState, edit_key: str, block_key: str) -> None:
    """Render a code block with syntax highlighting and editing capabilities."""
    # Edit state
    is_editing = st.session_state.get(f"{edit_key}_active", False)
    
    # Language selection and controls
    language = block.get("metadata", {}).get("language", "python")
    
    # Create columns for controls
    col1, col2, col3 = st.columns([7, 2, 1])
    
    with col1:
        if is_editing:
            selected_language = st.selectbox(
                "Language:",
                ["python", "javascript", "java", "c", "cpp", "csharp", "go", "rust", "html", "css", "bash", "sql", "json"],
                index=["python", "javascript", "java", "c", "cpp", "csharp", "go", "rust", "html", "css", "bash", "sql", "json"].index(language) 
                      if language in ["python", "javascript", "java", "c", "cpp", "csharp", "go", "rust", "html", "css", "bash", "sql", "json"] else 0,
                key=f"{block_key}_language"
            )
            
            # Update metadata
            block["metadata"] = block.get("metadata", {})
            block["metadata"]["language"] = selected_language
            language = selected_language
    
    with col2:
        # Show run button for code blocks
        if not is_editing and language in ["python", "javascript"]:
            if st.button("▶️ Run", key=f"{block_key}_run"):
                st.info("Code execution is not implemented in this demo.")
    
    with col3:
        if st.button("✏️" if not is_editing else "Cancel", key=f"{block_key}_edit"):
            st.session_state[f"{edit_key}_active"] = not is_editing
            st.rerun()
    
    # Display or edit the code
    if is_editing:
        # Edit mode
        new_content = st.text_area(
            "Edit code:", 
            value=block["content"],
            key=f"{block_key}_editor",
            height=min(100 + (block["content"].count('\n') * 20), 400)
        )
        
        if st.button("Save", key=f"{block_key}_save"):
            # Update block with new content and language
            doc_state.update_block(
                block["id"], 
                new_content, 
                {"language": language}
            )
            st.session_state[f"{edit_key}_active"] = False
            st.rerun()
    else:
        # Display mode with syntax highlighting
        st.code(block["content"], language=language)

def render_markdown_block(block: Dict, doc_state: DocumentState, edit_key: str, block_key: str) -> None:
    """Render a markdown block with preview and editing capabilities."""
    # Edit state
    is_editing = st.session_state.get(f"{edit_key}_active", False)
    
    # Edit and display controls
    col1, col2 = st.columns([9, 1])
    
    with col1:
        if is_editing:
            # Edit mode
            new_content = st.text_area(
                "Edit markdown:", 
                value=block["content"],
                key=f"{block_key}_editor",
                height=min(100 + (block["content"].count('\n') * 20), 400)
            )
            
            # Preview toggle
            show_preview = st.checkbox("Show Preview", value=True, key=f"{block_key}_preview_toggle")
            
            if show_preview:
                st.write("Preview:")
                st.markdown(new_content)
            
            if st.button("Save", key=f"{block_key}_save"):
                # Update block
                doc_state.update_block(block["id"], new_content)
                st.session_state[f"{edit_key}_active"] = False
                st.rerun()
        else:
            # Display mode with markdown rendering
            st.markdown(block["content"])
    
    with col2:
        if st.button("✏️" if not is_editing else "Cancel", key=f"{block_key}_edit"):
            st.session_state[f"{edit_key}_active"] = not is_editing
            st.rerun()

def render_table_block(block: Dict, doc_state: DocumentState, edit_key: str, block_key: str) -> None:
    """Render a table block (uses markdown tables)."""
    # Just reuse markdown renderer
    render_markdown_block(block, doc_state, edit_key, block_key)

def canvas_ui(model_callback: Callable = None) -> None:
    """
    Main UI for the Canvas document editor.
    
    Args:
        model_callback: Optional callback function for AI assistance
    """
    st.title("📄 Canvas Document Editor")
    
    # Sidebar for document management
    with st.sidebar:
        # New document button
        if st.button("➕ New Document"):
            # Create new document with unique ID
            doc_id = str(uuid.uuid4())
            doc_state = DocumentState(doc_id, "Untitled Document")
            doc_state.save()
            
            # Store in session state
            st.session_state.canvas_doc_id = doc_id
            st.success("New document created!")
            st.rerun()
        
        # Document list
        st.subheader("Your Documents")
        documents = get_available_documents()
        
        for doc in documents:
            col1, col2 = st.columns([4, 1])
            with col1:
                # Format date for better display
                last_modified = "Recently"
                if doc.get("last_modified"):
                    try:
                        dt = datetime.fromisoformat(doc["last_modified"])
                        last_modified = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        pass
                
                doc_label = f"{doc['name']} ({doc['block_count']} blocks, {last_modified})"
                if st.button(doc_label, key=f"load_{doc['doc_id']}"):
                    st.session_state.canvas_doc_id = doc["doc_id"]
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"delete_{doc['doc_id']}"):
                    if delete_document(doc["doc_id"]):
                        if st.session_state.get("canvas_doc_id") == doc["doc_id"]:
                            # If we deleted the current document, clear selection
                            st.session_state.canvas_doc_id = None
                        st.success(f"Deleted document: {doc['name']}")
                        st.rerun()
                    else:
                        st.error("Failed to delete document.")
    
    # Main document area
    if "canvas_doc_id" in st.session_state and st.session_state.canvas_doc_id:
        # Load or create document
        doc_state = DocumentState(st.session_state.canvas_doc_id)
        
        # Render document UI
        render_document_ui(doc_state, model_callback)
    else:
        # No document selected
        st.info("Select a document from the sidebar or create a new one to get started.")
        
        # Show demo button
        if st.button("Create Demo Document"):
            # Create a demo document with sample content
            doc_id = str(uuid.uuid4())
            doc_state = DocumentState(doc_id, "Demo Document")
            
            # Add sample blocks
            doc_state.add_block(
                BlockType.MARKDOWN, 
                "# Welcome to Canvas\n\nThis is a collaborative document editor where you can work with AI to create and edit content."
            )
            
            doc_state.add_block(
                BlockType.CODE,
                "def hello_world():\n    print('Hello, Canvas!')\n\nhello_world()",
                {"language": "python"}
            )
            
            doc_state.add_block(
                BlockType.TEXT,
                "You can add different types of content blocks, including text, code, and markdown."
            )
            
            doc_state.save()
            
            # Set as current document
            st.session_state.canvas_doc_id = doc_id
            st.success("Demo document created!")
            st.rerun()

# For testing the module directly
if __name__ == "__main__":
    def dummy_model_callback(prompt):
        return f"""
        I've analyzed your document. Here are some suggestions:
        
        ### BEGIN CONTENT ###
        # Improved Heading
        
        This is some improved markdown content that I'm suggesting as a replacement or new block.
        
        - Point 1
        - Point 2
        - Point 3
        ### END CONTENT ###
        
        I also noticed you might want to add some code:
        
        ### BEGIN CONTENT ###
        ```python
        def hello_world():
            print("Hello, Canvas!")
            
        hello_world()
        ```
        ### END CONTENT ###
        
        Let me know if you'd like further assistance with your document!
        """
    
    canvas_ui(model_callback=dummy_model_callback)