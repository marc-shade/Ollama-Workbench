# file_management.py
import streamlit as st
import os
import shutil
import re
import tiktoken
from typing import List, Optional, Dict, Any
import json
import time
import logging

# Setup logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        # Return a rough estimate if tiktoken fails
        return len(text) // 4  # Very rough estimate

def split_file(file_path: str, chunk_size: int, include_extension: bool = True) -> List[str]:
    """
    Split a text file into multiple chunks based on the specified size.
    
    Args:
        file_path: Path to the file to split
        chunk_size: Size of each chunk in bytes
        include_extension: Whether to include the original file extension in chunk filenames
        
    Returns:
        List of created chunk file paths
    """
    file_number = 1
    file_base, file_ext = os.path.splitext(file_path)
    chunk_files = []
    
    try:
        # Handle text files with various encodings
        encodings = ['utf-8', 'latin-1', 'windows-1252', 'ascii']
        file_content = None
        
        # Try different encodings
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    file_content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if file_content is None:
            # If all text encodings fail, try binary mode for PDFs
            if file_path.lower().endswith('.pdf'):
                return split_pdf_file(file_path, chunk_size, include_extension)
            logger.error(f"Failed to decode file {file_path} with any encoding")
            return []
            
        # Now split the content
        for i in range(0, len(file_content), chunk_size):
            chunk = file_content[i:i+chunk_size]
            if include_extension:
                chunk_file_path = f"{file_base}.part{file_number}{file_ext}"
            else:
                chunk_file_path = f"{file_base}.part{file_number}"
            
            with open(chunk_file_path, 'w', encoding='utf-8') as chunk_file:
                chunk_file.write(chunk)
                
            chunk_files.append(chunk_file_path)
            file_number += 1
            
        return chunk_files
    except Exception as e:
        logger.error(f"Error splitting file {file_path}: {str(e)}")
        return []

def split_pdf_file(file_path: str, chunk_size: int, include_extension: bool = True) -> List[str]:
    """
    Split a PDF file into multiple chunks (STUB - actual implementation would require PyPDF2)
    """
    # This is a placeholder for PDF splitting logic
    logger.warning(f"PDF splitting not fully implemented for {file_path}")
    return []
        
def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """Get metadata for a file including size, type, and modification time."""
    try:
        file_size = os.path.getsize(file_path)
        mod_time = os.path.getmtime(file_path)
        file_type = os.path.splitext(file_path)[1].lower()[1:]  # Remove the dot
        
        # Convert bytes to human-readable format
        size_labels = ['B', 'KB', 'MB', 'GB', 'TB']
        size_index = 0
        human_size = file_size
        
        while human_size > 1024 and size_index < len(size_labels) - 1:
            human_size /= 1024
            size_index += 1
            
        # Format modification time
        mod_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
        
        return {
            "name": os.path.basename(file_path),
            "path": file_path,
            "size_bytes": file_size,
            "size_human": f"{human_size:.2f} {size_labels[size_index]}",
            "type": file_type,
            "modified": mod_time_str
        }
    except Exception as e:
        logger.error(f"Error getting metadata for {file_path}: {str(e)}")
        return {
            "name": os.path.basename(file_path),
            "path": file_path,
            "error": str(e)
        }

def files_tab():
    """Main function for the Files tab UI."""
    st.title("📂 Files")
    
    # Create files folder if it doesn't exist
    files_folder = "files"
    if not os.path.exists(files_folder):
        os.makedirs(files_folder)
        
    # Define allowed file extensions
    allowed_extensions = ['.json', '.txt', '.pdf', '.gif', '.jpg', '.jpeg', '.png', '.md', '.csv']
    
    # Get list of files
    try:
        files = [f for f in os.listdir(files_folder) 
                if os.path.isfile(os.path.join(files_folder, f)) 
                and (os.path.splitext(f)[1].lower() in allowed_extensions or not os.path.splitext(f)[1])]
    except Exception as e:
        st.error(f"Error reading files directory: {str(e)}")
        files = []

    for file in files:
        col1, col2, col3, col4 = st.columns([20, 1, 1, 1])
        with col1:
            st.write(file)
        with col2:
            if file.endswith('.pdf'):
                st.button("📥", key=f"download_{file}")
            else:
                st.button("👀", key=f"view_{file}")
        with col3:
            if not file.endswith('.pdf'):
                st.button("✏️", key=f"edit_{file}")
        with col4:
            st.button("🗑️", key=f"delete_{file}")

    for file in files:
        file_path = os.path.join(files_folder, file)
        
        if st.session_state.get(f"view_{file}", False):
            try:
                with open(file_path, "r", encoding='utf-8') as f:
                    file_content = f.read()
                st.text_area("File Content:", value=file_content, height=200, key=f"view_content_{file}")
            except UnicodeDecodeError:
                st.error(f"Unable to decode file {file}. It may be a binary file.")
        
        if st.session_state.get(f"edit_{file}", False):
            try:
                with open(file_path, "r", encoding='utf-8') as f:
                    file_content = f.read()
                new_content = st.text_area("Edit File Content:", value=file_content, height=200, key=f"edit_content_{file}")
                if st.button("Save Changes", key=f"save_{file}"):
                    with open(file_path, "w", encoding='utf-8') as f:
                        f.write(new_content)
                    st.success(f"Changes saved to {file}")
            except UnicodeDecodeError:
                st.error(f"Unable to decode file {file}. It may be a binary file.")
        
        if st.session_state.get(f"download_{file}", False):
            if file.endswith('.pdf'):
                with open(file_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name=file,
                    mime='application/pdf',
                )
            else:
                with open(file_path, "r", encoding='utf-8') as f:
                    file_content = f.read()
                st.download_button(
                    label="Download File",
                    data=file_content,
                    file_name=file,
                    mime='text/plain',
                )
        
        if st.session_state.get(f"delete_{file}", False):
            os.remove(file_path)
            st.success(f"File {file} deleted.")
            st.rerun()

    uploaded_file = st.file_uploader("Upload a file", type=['txt', 'pdf', 'json', 'gif', 'jpg', 'jpeg', 'png', 'md'])
    if uploaded_file is not None:
        file_path = os.path.join(files_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        st.rerun()

    st.subheader("✂️ Split File")
    text_files = [f for f in files if f.endswith(('.txt', '.md'))]
    selected_file = st.selectbox("Select a text file to split", text_files)
    chunk_size_mb = st.slider("Chunk Size (MB)", 1, 100, 20)
    chunk_size = chunk_size_mb * 1024 * 1024  # Convert MB to bytes
    include_extension = st.checkbox("Include original file extension in chunk filenames", value=True)
    if st.button("✂️ Split File"):
        if selected_file:
            file_path = os.path.join(files_folder, selected_file)
            split_file(file_path, chunk_size, include_extension)
            st.success(f"File '{selected_file}' split into chunks.")
            st.rerun()
        else:
            st.warning("Please select a file to split.")
