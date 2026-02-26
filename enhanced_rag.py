"""
Enhanced RAG Interface for Ollama Workbench

This module provides a more sophisticated Retrieval Augmented Generation interface
similar to the RAG capabilities in Open WebUI, including:
- Document upload and processing
- Web content integration
- Inline citations
- Configurable embedding models
"""

import streamlit as st
import os
import json
import uuid
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from enum import Enum

from ollama_utils import get_available_models, get_ollama_client, call_ollama_endpoint
from enhanced_corpus import GraphRAGCorpus, OllamaEmbedder
from styles import apply_styles
from session_utils import initialize_session_state

# Define document types locally to avoid dependency issues
class DocumentTypes(str, Enum):
    """Enum for document types."""
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    CSV = "csv"
    HTML = "html"
    JSON = "json"
    CODE = "code"

logger = logging.getLogger(__name__)

# Constants
RAGTEST_DIR = "ragtest"
UPLOADS_DIR = "uploads"
SUPPORTED_FILE_TYPES = [".txt", ".md", ".pdf", ".docx", ".csv", ".html", ".json"]

# Ensure directories exist
os.makedirs(RAGTEST_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

def get_corpus_path(corpus_name: str) -> str:
    """Get the path to a corpus directory."""
    return os.path.join(RAGTEST_DIR, corpus_name)

def get_available_corpora() -> List[str]:
    """Get list of available corpora."""
    return [d for d in os.listdir(RAGTEST_DIR) if os.path.isdir(os.path.join(RAGTEST_DIR, d))]

def create_new_corpus(corpus_name: str) -> bool:
    """Create a new corpus directory."""
    if not corpus_name:
        return False
    
    corpus_path = get_corpus_path(corpus_name)
    try:
        os.makedirs(corpus_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating corpus: {e}")
        return False

def upload_document(file, corpus_name: str) -> Optional[str]:
    """Process and add an uploaded document to the corpus."""
    if not file or not corpus_name:
        return None
    
    # Save the uploaded file
    file_name = file.name
    file_path = os.path.join(UPLOADS_DIR, file_name)
    
    try:
        # Save the file
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        # Get the file extension to determine document type
        _, file_ext = os.path.splitext(file_name)
        file_ext = file_ext.lower()
        
        # Set document type based on extension
        if file_ext == ".pdf":
            doc_type = DocumentTypes.PDF
        elif file_ext in [".md", ".txt"]:
            doc_type = DocumentTypes.TEXT
        elif file_ext == ".docx":
            doc_type = DocumentTypes.DOCX
        elif file_ext == ".csv":
            doc_type = DocumentTypes.CSV
        elif file_ext == ".html":
            doc_type = DocumentTypes.HTML
        elif file_ext == ".json":
            doc_type = DocumentTypes.JSON
        else:
            doc_type = DocumentTypes.TEXT
        
        # Add to corpus
        embedder = OllamaEmbedder(model=st.session_state.rag_embedding_model)
        
        # Check if corpus exists, if not create it
        corpus_path = get_corpus_path(corpus_name)
        if not os.path.exists(corpus_path):
            create_new_corpus(corpus_name)
        
        # Load or create corpus
        try:
            corpus = GraphRAGCorpus.load(corpus_name, embedder)
        except:
            corpus = GraphRAGCorpus(embedder)
        
        # Add document to corpus
        document_id = corpus.add_document(file_path, doc_type=doc_type, filename=file_name)
        
        # Save corpus
        corpus.save(corpus_name)
        
        return document_id
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return None

def add_web_content(url: str, corpus_name: str) -> Optional[str]:
    """Add web content to the corpus."""
    if not url or not corpus_name:
        return None
    
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Fetch the web page
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text content
        text = soup.get_text(separator='\n\n')
        
        # Create a temporary file
        file_name = f"web_{int(time.time())}.html"
        file_path = os.path.join(UPLOADS_DIR, file_name)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Add to corpus
        embedder = OllamaEmbedder(model=st.session_state.rag_embedding_model)
        
        # Load or create corpus
        try:
            corpus = GraphRAGCorpus.load(corpus_name, embedder)
        except:
            corpus = GraphRAGCorpus(embedder)
        
        # Add document to corpus
        document_id = corpus.add_document(file_path, doc_type=DocumentTypes.HTML, filename=url)
        
        # Save corpus
        corpus.save(corpus_name)
        
        return document_id
    except Exception as e:
        logger.error(f"Error adding web content: {e}")
        return None

def query_corpus(query: str, corpus_name: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """Query the corpus for relevant documents."""
    if not query or not corpus_name:
        return []
    
    try:
        # Initialize embedder and corpus
        embedder = OllamaEmbedder(model=st.session_state.rag_embedding_model)
        
        # Load corpus
        try:
            corpus = GraphRAGCorpus.load(corpus_name, embedder)
        except:
            # No corpus found
            return []
        
        # Query corpus
        results = corpus.query(query, n_results=n_results)
        
        return results
    except Exception as e:
        logger.error(f"Error querying corpus: {e}")
        return []

def get_corpus_stats(corpus_name: str) -> Dict[str, Any]:
    """Get statistics about a corpus."""
    stats = {
        "document_count": 0,
        "chunk_count": 0,
        "document_types": {}
    }
    
    try:
        # Initialize embedder
        embedder = OllamaEmbedder(model=st.session_state.rag_embedding_model)
        
        # Load corpus
        corpus = GraphRAGCorpus.load(corpus_name, embedder)
        
        # Get document count
        stats["document_count"] = len(corpus.documents)
        
        # Get chunk count
        stats["chunk_count"] = len(corpus.chunks)
        
        # Get document types
        for doc in corpus.documents:
            doc_type = doc.get("doc_type", "unknown")
            if doc_type in stats["document_types"]:
                stats["document_types"][doc_type] += 1
            else:
                stats["document_types"][doc_type] = 1
        
        return stats
    except Exception as e:
        logger.error(f"Error getting corpus stats: {e}")
        return stats

def enhanced_rag_interface():
    """Main UI for the enhanced RAG interface."""
    # Apply styling
    colors, theme = apply_styles()
    
    # Initialize session state
    initialize_session_state()
    
    # Page header
    st.markdown(f"<h1 style='font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem;'>📚 Enhanced RAG Interface</h1>", unsafe_allow_html=True)
    
    # Create sidebar
    with st.sidebar:
        st.markdown(
            '<div class="logo-title">🦙 Ollama <span>RAG</span></div>',
            unsafe_allow_html=True
        )
        
        # Corpus Management
        st.subheader("Corpus Management")
        
        # Create new corpus
        new_corpus_name = st.text_input("New Corpus Name:")
        if st.button("Create Corpus", key="create_corpus") and new_corpus_name:
            if create_new_corpus(new_corpus_name):
                st.success(f"Created corpus: {new_corpus_name}")
                st.session_state.rag_corpus_name = new_corpus_name
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to create corpus")
        
        # Select existing corpus
        available_corpora = get_available_corpora()
        if available_corpora:
            st.selectbox(
                "Select Corpus",
                available_corpora,
                index=available_corpora.index(st.session_state.rag_corpus_name) if st.session_state.rag_corpus_name in available_corpora else 0,
                key="rag_corpus_name"
            )
        else:
            st.info("No corpora available. Create one to get started.")
        
        # Model settings
        st.subheader("Models")
        
        # Get available models
        models = get_available_models()
        if not models:
            st.error("No models available. Please make sure Ollama is running.")
        else:
            # LLM Model selection
            st.selectbox(
                "LLM Model",
                models,
                index=models.index(st.session_state.rag_llm_model) if st.session_state.rag_llm_model in models else 0,
                key="rag_llm_model"
            )
            
            # Embedding model selection
            st.selectbox(
                "Embedding Model",
                models,
                index=models.index(st.session_state.rag_embedding_model) if st.session_state.rag_embedding_model in models else 0,
                key="rag_embedding_model"
            )
            
            # Temperature
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.rag_temperature,
                step=0.1,
                key="rag_temperature"
            )
    
    # Main interface with tabs
    tab1, tab2 = st.tabs(["Chat", "Knowledge Base"])
    
    with tab1:
        # Chat interface
        st.subheader("RAG Chat")
        
        # Display chat history
        for i, message in enumerate(st.session_state.rag_chat_history):
            role = message.get("role", "assistant")
            content = message.get("content", "")
            
            with st.chat_message(role):
                # Display message content
                st.markdown(content)
                
                # Display sources if available and this is an assistant message
                if role == "assistant" and "sources" in message:
                    with st.expander("Sources"):
                        for source in message["sources"]:
                            st.markdown(f"**{source.get('filename', 'Document')}** ({source.get('similarity', 0):.2f})")
                            st.markdown(f"```\n{source.get('content', '')}\n```")
        
        # Chat input
        query = st.chat_input("Ask a question about your documents...")
        
        if query:
            # Display user message
            st.session_state.rag_chat_history.append({"role": "user", "content": query})
            st.session_state.rag_last_query = query
            
            # Rerun to show user message
            st.rerun()
    
    with tab2:
        # Knowledge base management
        st.subheader("Document Management")
        
        # Document upload
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=[ext[1:] for ext in SUPPORTED_FILE_TYPES],
            help="Upload a document to add to the corpus."
        )
        
        if uploaded_file is not None and st.session_state.rag_corpus_name:
            st.info(f"Uploaded: {uploaded_file.name}")
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    doc_id = upload_document(uploaded_file, st.session_state.rag_corpus_name)
                    if doc_id:
                        st.success(f"Added document to corpus: {uploaded_file.name}")
                    else:
                        st.error("Failed to process document")
        
        # Web content
        st.subheader("Add Web Content")
        web_url = st.text_input("Website URL:")
        if web_url and st.button("Add Web Content"):
            with st.spinner("Fetching and processing web content..."):
                doc_id = add_web_content(web_url, st.session_state.rag_corpus_name)
                if doc_id:
                    st.success(f"Added web content to corpus: {web_url}")
                else:
                    st.error("Failed to process web content")
        
        # Corpus statistics
        if st.session_state.rag_corpus_name in get_available_corpora():
            st.subheader("Corpus Statistics")
            stats = get_corpus_stats(st.session_state.rag_corpus_name)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents", stats["document_count"])
            with col2:
                st.metric("Chunks", stats["chunk_count"])
            with col3:
                st.metric("Document Types", len(stats["document_types"]))
            
            # Document types
            if stats["document_types"]:
                st.subheader("Document Types")
                for doc_type, count in stats["document_types"].items():
                    st.text(f"{doc_type}: {count}")
    
    # Process the query if one was submitted
    if st.session_state.rag_last_query and st.session_state.rag_chat_history and st.session_state.rag_chat_history[-1]["role"] == "user":
        query = st.session_state.rag_last_query
        st.session_state.rag_last_query = ""  # Reset to prevent reprocessing
        
        with st.spinner("Searching knowledge base..."):
            # Query the corpus
            results = query_corpus(query, st.session_state.rag_corpus_name)
            
            if not results:
                # No results found
                with st.chat_message("assistant"):
                    st.markdown("I don't have enough information in my knowledge base to answer this question. Try uploading more relevant documents.")
                
                # Add to chat history
                st.session_state.rag_chat_history.append({
                    "role": "assistant",
                    "content": "I don't have enough information in my knowledge base to answer this question. Try uploading more relevant documents."
                })
                st.rerun()
            
            # Construct context from results
            context = ""
            for i, result in enumerate(results):
                context += f"[{i+1}] {result.get('content', '')}\n\n"
            
            # Construct prompt
            prompt = f"""
            Please answer the following question based ONLY on the provided context. If the information cannot be found in the context, say "I don't have enough information to answer this question."
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:
            """
            
            # Call the model
            try:
                client = get_ollama_client()
                
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    if client:
                        # Stream response
                        for response_chunk in client.generate(
                            model=st.session_state.rag_llm_model,
                            prompt=prompt,
                            stream=True,
                            options={
                                "temperature": st.session_state.rag_temperature
                            }
                        ):
                            content = response_chunk["response"]
                            full_response += content
                            message_placeholder.markdown(full_response + "▌")
                        
                        message_placeholder.markdown(full_response)
                    else:
                        # Use non-streaming fallback
                        full_response, _, _, _ = call_ollama_endpoint(
                            model=st.session_state.rag_llm_model,
                            prompt=prompt,
                            temperature=st.session_state.rag_temperature
                        )
                        message_placeholder.markdown(full_response)
                    
                    # Show sources
                    with st.expander("Sources"):
                        for i, result in enumerate(results):
                            st.markdown(f"**Source {i+1}:** {result.get('filename', 'Document')} ({result.get('similarity', 0):.2f})")
                            st.markdown(f"```\n{result.get('content', '')}\n```")
                
                # Add to chat history
                st.session_state.rag_chat_history.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": results
                })
                
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                with st.chat_message("assistant"):
                    st.error(error_message)
                
                # Add error to chat history
                st.session_state.rag_chat_history.append({
                    "role": "assistant",
                    "content": error_message
                })
            
            # Reset the query
            st.session_state.rag_last_query = ""
            st.rerun()

if __name__ == "__main__":
    enhanced_rag_interface()