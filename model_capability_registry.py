# model_capability_registry.py

"""
Central registry for model capabilities based on Ollama's official categorization.
This module provides functions to determine model capabilities such as tool/function calling,
vision/multimodal capabilities, and embedding support.
"""

import re
import logging
from typing import Dict, List, Set, Optional

logger = logging.getLogger(__name__)

# Official Ollama model capabilities based on their website categories
# These are models explicitly listed on https://ollama.com/search?c=X
TOOLS_CAPABLE_MODELS = {
    "athene-v2", "aya-expanse", "cogito", "command-a", "command-r", 
    "command-r-plus", "command-r7b", "command-r7b-arabic", "firefunction-v2", 
    "granite3-dense", "granite3-moe", "granite3.1-dense", "granite3.1-moe", 
    "granite3.2", "granite3.2-vision", "granite3.3", "hermes3", 
    "llama3-groq-tool-use", "llama3.1", "llama3.2", "llama3.3", "llama4", 
    "mistral", "mistral-large", "mistral-nemo", "mistral-small", 
    "mistral-small3.1", "mixtral", "nemotron", "nemotron-mini", "phi4-mini", 
    "qwen2", "qwen2.5", "qwen2.5-coder", "qwen3", "qwq", "smollm2"
}

VISION_CAPABLE_MODELS = {
    "bakllava", "gemma3", "granite3.2-vision", "llama3.2-vision", "llama4", 
    "llava", "llava-llama3", "llava-phi3", "minicpm-v", "mistral-small3.1", 
    "moondream", "qwen2.5vl"
}

EMBEDDING_CAPABLE_MODELS = {
    "all-minilm", "bge-large", "bge-m3", "granite-embedding", 
    "mxbai-embed-large", "nomic-embed-text", "paraphrase-multilingual", 
    "snowflake-arctic-embed", "snowflake-arctic-embed2"
}

# Additional model families and patterns
COMMON_FUNCTION_FAMILIES = [
    "llama3", "mistral", "qwen", "phi3", "phi-3", "phi4", "phi-4", 
    "gemma", "mixtral", "command", "granite"
]

COMMON_VISION_FAMILIES = [
    "llava", "bakllava", "llama3.1", "llama3.2", "llama4", "vision", 
    "gemma3", "qwen-vl", "qwen2-vl", "qwen2.5vl", "cogvlm", "fuyu",
    "moondream", "persee", "minigpt"
]

COMMON_EMBEDDING_PATTERNS = [
    "embed", "embedding", "text-embedding", "all-minilm", "bge", "nomic", 
    "arctic-embed", "sentence-transformer"
]

def normalize_model_name(model_name: str) -> str:
    """
    Normalize a model name by removing specific tags, versions and quantizations.
    This helps match capabilities across model variants.
    
    Args:
        model_name: The raw model name
        
    Returns:
        Normalized model name for capability matching
    """
    # Convert to lowercase
    name = model_name.lower()
    
    # Handle model:tag format (e.g., llama3:8b)
    if ":" in name:
        # Extract base name before the colon
        name = name.split(":")[0]
    
    # Remove common quantization suffixes
    quant_patterns = ["-q2_k", "-q3_k", "-q4_0", "-q4_k", "-q5_0", "-q5_k", 
                     "-q6_k", "-q8_0", "q2k", "q3k", "q4k", "q5k", "q6k", "q8_0"]
    for pattern in quant_patterns:
        name = name.replace(pattern, "")
    
    # Remove size indicators when not part of the core name
    # This preserves names like llama3.1, llama3.2 while removing size suffixes
    size_patterns = ["-8b", "-70b", "-7b", "-13b", "-34b", "-3b"]
    for pattern in size_patterns:
        name = name.replace(pattern, "")
        
    # Handle special cases
    if "llava" in name and name != "llava":
        name = "llava"  # Normalize all llava variants
        
    return name

def is_tools_capable(model_name: str) -> bool:
    """
    Check if a model is capable of using tools/function calling.
    
    Args:
        model_name: The model name to check
        
    Returns:
        True if the model supports tool/function calling
    """
    # First check exact matches in our known tools models
    if model_name in TOOLS_CAPABLE_MODELS:
        return True
    
    # Normalize the name for better matching
    normalized_name = normalize_model_name(model_name)
    
    # Check if normalized name is in the known set
    if normalized_name in TOOLS_CAPABLE_MODELS:
        return True
    
    # Check for common function-calling model families
    for family in COMMON_FUNCTION_FAMILIES:
        if family in normalized_name:
            return True
    
    # Special cases based on model architecture
    if "function" in normalized_name:
        return True
    
    # Default to false if no matches
    return False

def is_vision_capable(model_name: str) -> bool:
    """
    Check if a model is capable of processing images (multimodal).
    
    Args:
        model_name: The model name to check
        
    Returns:
        True if the model supports image processing
    """
    # First check exact matches in our known vision models
    if model_name in VISION_CAPABLE_MODELS:
        return True
    
    # Normalize the name for better matching
    normalized_name = normalize_model_name(model_name)
    
    # Check if normalized name is in the known set
    if normalized_name in VISION_CAPABLE_MODELS:
        return True
    
    # Check for common vision model patterns
    for family in COMMON_VISION_FAMILIES:
        if family in normalized_name:
            return True
            
    # Special cases based on known naming patterns
    if any(pattern in normalized_name for pattern in ["vision", "vl", "visual", "multimodal"]):
        return True
    
    # Default to false if no matches
    return False

def is_embedding_capable(model_name: str) -> bool:
    """
    Check if a model is designed for generating embeddings.
    
    Args:
        model_name: The model name to check
        
    Returns:
        True if the model supports generating embeddings
    """
    # First check exact matches
    if model_name in EMBEDDING_CAPABLE_MODELS:
        return True
    
    # Normalize the name for better matching
    normalized_name = normalize_model_name(model_name)
    
    # Check if normalized name is in the known set
    if normalized_name in EMBEDDING_CAPABLE_MODELS:
        return True
    
    # Check for common embedding patterns
    for pattern in COMMON_EMBEDDING_PATTERNS:
        if pattern in normalized_name:
            return True
    
    # Default to false if no matches
    return False

def get_model_capabilities(model_name: str) -> Dict[str, bool]:
    """
    Get a complete set of capabilities for a given model.
    
    Args:
        model_name: The model name to check
        
    Returns:
        Dictionary of capability names to boolean values
    """
    return {
        "tools": is_tools_capable(model_name),
        "vision": is_vision_capable(model_name),
        "embedding": is_embedding_capable(model_name),
    }

def filter_models_by_capability(models: List[str], capability: str) -> List[str]:
    """
    Filter a list of models to only those with the specified capability.
    
    Args:
        models: List of model names
        capability: Capability to filter by ('tools', 'vision', or 'embedding')
        
    Returns:
        Filtered list of models with the specified capability
    """
    capability_checkers = {
        "tools": is_tools_capable,
        "vision": is_vision_capable,
        "embedding": is_embedding_capable,
    }
    
    checker = capability_checkers.get(capability)
    if not checker:
        logger.error(f"Unknown capability: {capability}")
        return models  # Return all models if capability unknown
        
    return [model for model in models if checker(model)]