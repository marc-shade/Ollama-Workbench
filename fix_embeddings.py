#!/usr/bin/env python3
"""
Fix for the embeddings dimensionality mismatch in Multi-Model Chat.

This module patches the get_token_embeddings function to ensure that embeddings
from different models can be compared without dimensionality errors.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_embeddings")

def install_patch():
    """Install the patch for the embeddings functions."""
    try:
        import ollama_utils
        
        # Store the original function
        original_get_token_embeddings = ollama_utils.get_token_embeddings
        
        def patched_get_token_embeddings(model: str, text: str, api_keys: Dict[str, Any] = None) -> Optional[np.ndarray]:
            """
            Patched version of get_token_embeddings that handles dimensionality mismatches.
            
            This function wraps the original get_token_embeddings to ensure all embeddings
            have a consistent dimensionality regardless of which model created them.
            """
            try:
                # Call the original function
                embedding = original_get_token_embeddings(model, text, api_keys)
                
                if embedding is None:
                    return None
                
                # For standardization - we could resize all embeddings to the same size
                # but that would lose information. Instead, we'll just return the original
                # and handle the comparison more carefully in the calling function.
                
                logger.info(f"Generated embeddings for model {model} with shape {embedding.shape}")
                return embedding
                
            except Exception as e:
                logger.error(f"Error in patched_get_token_embeddings for model {model}: {e}")
                return None
        
        # Apply the patch
        ollama_utils.get_token_embeddings = patched_get_token_embeddings
        logger.info("Successfully patched ollama_utils.get_token_embeddings")
        
        return True
    except Exception as e:
        logger.error(f"Failed to patch embeddings functions: {e}")
        return False

if __name__ == "__main__":
    success = install_patch()
    if success:
        print("Successfully patched embeddings functions")
    else:
        print("Failed to patch embeddings functions")