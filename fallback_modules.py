"""
Fallback modules for Ollama Workbench to ensure compatibility

This module provides fallback implementations for modules that might be missing
in some environments. This helps maintain compatibility across different setups.
"""

import streamlit as st
import logging

logger = logging.getLogger(__name__)

# Fallback for GraphRAGCorpus
class FallbackGraphRAGCorpus:
    def __init__(self, embedder=None):
        self.documents = []
        self.chunks = []
        self.embedder = embedder
    
    @classmethod
    def load(cls, name, embedder=None):
        logger.warning(f"Using fallback GraphRAGCorpus instead of real implementation")
        return cls(embedder)
    
    def add_document(self, file_path, doc_type="text", filename=None):
        doc_id = f"fallback-doc-{len(self.documents)}"
        self.documents.append({
            "id": doc_id,
            "file_path": file_path,
            "doc_type": doc_type,
            "filename": filename or file_path
        })
        return doc_id
    
    def save(self, name):
        logger.warning(f"Saving fallback corpus '{name}' (does nothing)")
        return True
    
    def query(self, query_text, n_results=3):
        logger.warning(f"Querying fallback corpus (returns empty results)")
        return []

# Fallback for OllamaEmbedder
class FallbackOllamaEmbedder:
    def __init__(self, model=None):
        self.model = model or "fallback-model"
    
    def embed(self, text):
        logger.warning(f"Using fallback embedder (returns empty embedding)")
        import numpy as np
        return np.zeros(384)  # Return a zero vector of typical embedding size

# Function to provide fallback modules
def get_enhanced_corpus_modules():
    try:
        from enhanced_corpus import GraphRAGCorpus, OllamaEmbedder
        return GraphRAGCorpus, OllamaEmbedder
    except ImportError:
        logger.warning("Failed to import enhanced_corpus modules, using fallbacks")
        return FallbackGraphRAGCorpus, FallbackOllamaEmbedder

# Function to get text-to-speech utilities
def get_tts_utils():
    try:
        from tts_utils import text_to_speech, play_speech
        return text_to_speech, play_speech
    except ImportError:
        logger.warning("Failed to import tts_utils, using fallbacks")
        
        def fallback_text_to_speech(text, voice="en-US-Neural2-F"):
            logger.warning(f"Using fallback text_to_speech (does nothing)")
            return None
        
        def fallback_play_speech(file_path):
            logger.warning(f"Using fallback play_speech (does nothing)")
            return None
        
        return fallback_text_to_speech, fallback_play_speech

# Apply styles function if styles module is not available
def apply_fallback_styles():
    logger.warning("Using fallback styles (does nothing special)")
    return {}, "light"  # Return empty colors dict and default theme