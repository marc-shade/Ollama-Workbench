"""
Simplified Enhanced RAG Interface for Ollama Workbench

This module provides a more basic RAG interface that's compatible with the existing codebase.
"""

import streamlit as st
import os
import json
import logging
import time
from datetime import datetime

from ollama_utils import get_available_models, get_ollama_client, call_ollama_endpoint

# Try to import from enhanced_corpus, fall back to fallback modules if needed
try:
    from enhanced_corpus import GraphRAGCorpus, OllamaEmbedder
except ImportError:
    from fallback_modules import get_enhanced_corpus_modules
    GraphRAGCorpus, OllamaEmbedder = get_enhanced_corpus_modules()

# Try to import styles, use fallback if needed
try:
    from styles import apply_styles
except ImportError:
    from fallback_modules import apply_fallback_styles as apply_styles

# Set up logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
RAGTEST_DIR = "ragtest"
UPLOADS_DIR = "uploads"

# Ensure directories exist
os.makedirs(RAGTEST_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

def initialize_session_state():
    """Initialize session state variables for RAG interface."""
    if "rag_corpus_name" not in st.session_state:
        st.session_state.rag_corpus_name = "default"
    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []
    if "rag_last_query" not in st.session_state:
        st.session_state.rag_last_query = ""
    
    # Get available Ollama models first
    models = get_available_models()
    
    if "rag_embedding_model" not in st.session_state:
        # Set a default embedding model, use first available or fallback to llama2
        st.session_state.rag_embedding_model = models[0] if models else "llama2"
    elif st.session_state.rag_embedding_model not in models and models:
        # If current model isn't available, reset to first available
        st.session_state.rag_embedding_model = models[0]
        
    if "rag_llm_model" not in st.session_state:
        # Default to first available model
        st.session_state.rag_llm_model = models[0] if models else "llama2"
    elif st.session_state.rag_llm_model not in models and models:
        # If current model isn't available, reset to first available
        st.session_state.rag_llm_model = models[0]
    if "rag_temperature" not in st.session_state:
        st.session_state.rag_temperature = 0.7

def get_corpus_path(corpus_name: str) -> str:
    """Get the path to a corpus directory."""
    return os.path.join(RAGTEST_DIR, corpus_name)

def get_available_corpora():
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

def upload_document(file, corpus_name: str):
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
        
        # Determine doc_type from file extension
        _, ext = os.path.splitext(file_name)
        doc_type = ext[1:] if ext else "text"
        
        # Add document to corpus
        document_id = corpus.add_document(file_path, doc_type=doc_type, filename=file_name)
        
        # Save corpus
        corpus.save(corpus_name)
        
        return document_id
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return None

def query_corpus(query: str, corpus_name: str, n_results: int = 5):
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

def enhanced_rag_interface():
    """Main UI for the simplified RAG interface."""
    # Apply styling
    colors, theme = apply_styles()
    
    # Ensure full width
    st.markdown("""
        <style>
        .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Page header
    st.title("📚 Enhanced RAG Interface")
    st.write("Retrieval Augmented Generation for better answers from your documents")
    
    # Create sidebar 
    with st.sidebar:
        st.header("RAG Settings")
        
        # Corpus Management
        st.subheader("Knowledge Base")
        
        # Create new corpus
        new_corpus_name = st.text_input("New Knowledge Base Name:")
        if st.button("Create") and new_corpus_name:
            if create_new_corpus(new_corpus_name):
                st.success(f"Created knowledge base: {new_corpus_name}")
                st.session_state.rag_corpus_name = new_corpus_name
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to create knowledge base")
        
        # Select existing corpus
        available_corpora = get_available_corpora()
        if available_corpora:
            st.selectbox(
                "Select Knowledge Base",
                available_corpora,
                index=available_corpora.index(st.session_state.rag_corpus_name) if st.session_state.rag_corpus_name in available_corpora else 0,
                key="rag_corpus_name"
            )
        else:
            st.info("No knowledge bases available. Create one to get started.")
        
        # Model settings
        st.subheader("Models")
        
        # Get available models
        models = get_available_models()
        if not models:
            st.error("No models available. Please make sure Ollama is running.")
        else:
            # LLM Model selection
            if models:
                default_llm_index = 0
                if st.session_state.rag_llm_model in models:
                    default_llm_index = models.index(st.session_state.rag_llm_model)
                st.selectbox(
                    "LLM Model",
                    models,
                    index=default_llm_index,
                    key="rag_llm_model"
                )
                
                # Embedding model selection
                default_embed_index = 0
                if st.session_state.rag_embedding_model in models:
                    default_embed_index = models.index(st.session_state.rag_embedding_model)
                st.selectbox(
                    "Embedding Model",
                    models,
                    index=default_embed_index,
                    key="rag_embedding_model"
                )
            else:
                st.error("No models available. Please make sure Ollama is running.")
                # Use text inputs as fallback
                st.text_input("LLM Model (fallback)", value=st.session_state.rag_llm_model, key="rag_llm_model")
                st.text_input("Embedding Model (fallback)", value=st.session_state.rag_embedding_model, key="rag_embedding_model")
            
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
    tab1, tab2 = st.tabs(["Chat", "Documents"])
    
    with tab1:
        # Chat interface
        st.header("RAG Chat")
        
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
        # Document management
        st.header("Document Management")
        
        # Document upload
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["pdf", "txt", "md", "docx", "csv", "html", "json"],
            help="Upload a document to add to the corpus."
        )
        
        if uploaded_file is not None and st.session_state.rag_corpus_name:
            st.info(f"Uploaded: {uploaded_file.name}")
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    doc_id = upload_document(uploaded_file, st.session_state.rag_corpus_name)
                    if doc_id:
                        st.success(f"Added document to knowledge base: {uploaded_file.name}")
                    else:
                        st.error("Failed to process document")
    
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
            Answer the following question based ONLY on the provided context. If the information cannot be found in the context, say "I don't have enough information to answer this question."
            
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