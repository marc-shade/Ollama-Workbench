"""
Model Categorization Module

This module provides functionality to categorize models by provider and type,
and offers a consistent UI for model selection with proper categorization.

CHECKPOINT: Model categorization initialization
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any

# Set up logging with detailed checkpoints
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.info("CHECKPOINT: Loading model_categorization module")

# Import dependencies safely
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logger.warning("CHECKPOINT: Streamlit not available, UI functions will be limited")

# Model category definitions
MODEL_CATEGORIES = {
    "ollama": {
        "display_name": "Ollama Models",
        "prefix_matchers": [],  # All models without other prefixes
        "description": "Local models running on your Ollama server"
    },
    "openai": {
        "display_name": "OpenAI Models",
        "prefix_matchers": ["gpt-", "text-", "dall-e", "whisper"],
        "description": "OpenAI hosted models (requires API key)"
    },
    "groq": {
        "display_name": "Groq Models",
        "prefix_matchers": ["llama2-", "mixtral-", "gemma-"],
        "description": "Groq hosted models (requires API key)"
    },
    "mistral": {
        "display_name": "Mistral Models",
        "prefix_matchers": ["mistral-", "mixtral-"],
        "description": "Mistral AI hosted models (requires API key)"
    },
    "anthropic": {
        "display_name": "Anthropic Models",
        "prefix_matchers": ["claude-"],
        "description": "Anthropic Claude models (requires API key)"
    }
}

# Model descriptions with capabilities
MODEL_DESCRIPTIONS = {
    # Ollama models
    "llama3": "Meta's Llama 3 model - general purpose, instruction-following",
    "llama2": "Meta's Llama 2 model - general purpose, instruction-following",
    "mistral": "Mistral 7B - efficient, high-quality instruction model",
    "gemma": "Google's Gemma model - lightweight, efficient model",
    "phi": "Microsoft's Phi model - compact, efficient model",
    "codellama": "Code-specialized Llama model for programming tasks",
    "orca-mini": "Lightweight model optimized for efficiency",
    "vicuna": "Fine-tuned LLaMA model with improved instruction following",
    "stable-diffusion": "Image generation model (multimodal)",
    "llava": "Multimodal model supporting vision and language",
    
    # OpenAI models
    "gpt-4": "OpenAI's most powerful model with strong reasoning",
    "gpt-4-turbo": "Faster version of GPT-4 with lower latency",
    "gpt-4-vision": "GPT-4 with vision capabilities (multimodal)",
    "gpt-3.5-turbo": "Efficient model balancing performance and speed",
    
    # Groq models
    "llama2-70b-4096": "Llama 2 70B hosted on Groq's LPU platform",
    "mixtral-8x7b-32768": "Mixtral model with extended context window on Groq",
    
    # Mistral models
    "mistral-small": "Mistral AI's compact model",
    "mistral-medium": "Mistral AI's medium-sized model",
    "mistral-large": "Mistral AI's most powerful model",
    
    # Anthropic models
    "claude-3-opus": "Anthropic's most capable Claude model",
    "claude-3-sonnet": "Balanced Claude model for most use cases",
    "claude-3-haiku": "Fast, efficient Claude model"
}

def categorize_models(models: List[str]) -> Dict[str, List[str]]:
    """
    Categorize a list of models by provider based on naming patterns.
    
    Args:
        models: List of model names
        
    Returns:
        Dict mapping category names to lists of model names
    """
    logger.info(f"CHECKPOINT: Categorizing {len(models)} models")
    
    # Initialize categories with empty lists
    categorized = {cat: [] for cat in MODEL_CATEGORIES.keys()}
    
    # Track models that have been categorized
    categorized_models = set()
    
    # First pass: categorize based on prefix matchers
    for model in models:
        for category, info in MODEL_CATEGORIES.items():
            if category == "ollama":
                continue  # Skip ollama for now, it's our fallback
                
            # Check if model matches any prefix for this category
            for prefix in info["prefix_matchers"]:
                if model.startswith(prefix):
                    categorized[category].append(model)
                    categorized_models.add(model)
                    break
    
    # Second pass: add remaining models to ollama category
    for model in models:
        if model not in categorized_models:
            categorized["ollama"].append(model)
    
    # Log results
    for category, model_list in categorized.items():
        logger.info(f"CHECKPOINT: Category '{category}' has {len(model_list)} models")
    
    return categorized

def get_model_description(model_name: str) -> str:
    """
    Get the description for a model, with fallback for unknown models.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Description string
    """
    return MODEL_DESCRIPTIONS.get(
        model_name, 
        f"Model running on {'Ollama' if not any(model_name.startswith(p) for c in MODEL_CATEGORIES.values() for p in c.get('prefix_matchers', []))  else 'external'} server"
    )

def get_model_category_display_name(category: str) -> str:
    """
    Get the display name for a model category.
    
    Args:
        category: Category key
        
    Returns:
        Display name for the category
    """
    return MODEL_CATEGORIES.get(category, {}).get("display_name", category.title())

def create_categorized_model_ui(available_models: List[str], current_model: Optional[str] = None) -> Tuple[str, bool]:
    """
    Create a UI for selecting models with proper categorization.
    
    Args:
        available_models: List of available model names
        current_model: Currently selected model
        
    Returns:
        Tuple of (selected_model, changed)
    """
    if not STREAMLIT_AVAILABLE:
        logger.warning("CHECKPOINT: Streamlit not available, cannot create UI")
        return current_model or (available_models[0] if available_models else "llama3"), False
    
    logger.info("CHECKPOINT: Creating categorized model UI")
    
    # Ensure we have a valid current model
    if not current_model or current_model not in available_models:
        current_model = available_models[0] if available_models else "llama3"
    
    # Categorize models
    categorized_models = categorize_models(available_models)
    
    # Create UI
    st.subheader("Model Selection")
    
    # Find which category the current model belongs to
    current_category = None
    for category, models in categorized_models.items():
        if current_model in models:
            current_category = category
            break
    
    if not current_category and categorized_models["ollama"]:
        current_category = "ollama"
    elif not current_category:
        # If we can't find the category, use the first non-empty category
        for category, models in categorized_models.items():
            if models:
                current_category = category
                current_model = models[0]
                break
    
    # Create category tabs
    non_empty_categories = [cat for cat, models in categorized_models.items() if models]
    if non_empty_categories:
        tabs = st.tabs([get_model_category_display_name(cat) for cat in non_empty_categories])
        
        selected_model = current_model
        changed = False
        
        # Fill each tab with its models
        for i, category in enumerate(non_empty_categories):
            with tabs[i]:
                models = categorized_models[category]
                if not models:
                    st.info(f"No {category} models available")
                    continue
                
                # Find index of current model in this category
                try:
                    index = models.index(current_model) if current_model in models else 0
                except ValueError:
                    index = 0
                
                # Create model selector
                with st.form(key=f"model_selection_form_{category}"):
                    model_choice = st.selectbox(
                        "📦 Model:",
                        models,
                        index=index,
                        key=f"model_selector_{category}",
                        help=get_model_description(models[index])
                    )
                    
                    # Show model description
                    st.markdown(f"**Description:** {get_model_description(model_choice)}")
                    
                    # Form submit button
                    submit_button = st.form_submit_button(label="Select Model")
                    
                    if submit_button and model_choice != current_model:
                        selected_model = model_choice
                        changed = True
                        st.success(f"Model changed to {model_choice}")
                        
                        # Log the change
                        logger.info(f"CHECKPOINT: Model changed from '{current_model}' to '{model_choice}'")
        
        return selected_model, changed
    else:
        st.warning("No models available. Please ensure Ollama is running and models are installed.")
        return current_model, False

def save_model_to_settings(model_name: str) -> bool:
    """
    Save the selected model to settings file.
    
    Args:
        model_name: Name of the model to save
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"CHECKPOINT: Saving model '{model_name}' to settings")
    
    try:
        settings_file = "app_settings.json"
        
        # Load existing settings
        if os.path.exists(settings_file):
            with open(settings_file, "r") as f:
                settings = json.load(f)
        else:
            settings = {}
        
        # Update model settings
        settings["selected_model"] = model_name
        settings["current_model"] = model_name
        
        # Save settings
        with open(settings_file, "w") as f:
            json.dump(settings, f, indent=2)
            
        logger.info(f"CHECKPOINT: Successfully saved model '{model_name}' to settings")
        return True
    except Exception as e:
        logger.error(f"CHECKPOINT: Error saving model to settings: {str(e)}")
        return False

# Test function to verify model categorization
def test_model_categorization():
    """Test the model categorization functionality."""
    logger.info("CHECKPOINT: Running model categorization tests")
    
    # Test case 1: Mixed models
    test_models = [
        "llama3", "mistral", "gemma", "phi", 
        "gpt-4", "gpt-3.5-turbo", 
        "llama2-70b-4096", "mixtral-8x7b-32768",
        "mistral-small", "mistral-medium",
        "claude-3-opus"
    ]
    
    categorized = categorize_models(test_models)
    
    # Verify categorization
    assert "llama3" in categorized["ollama"], "llama3 should be in ollama category"
    assert "gpt-4" in categorized["openai"], "gpt-4 should be in openai category"
    assert "llama2-70b-4096" in categorized["groq"], "llama2-70b-4096 should be in groq category"
    assert "mistral-small" in categorized["mistral"], "mistral-small should be in mistral category"
    assert "claude-3-opus" in categorized["anthropic"], "claude-3-opus should be in anthropic category"
    
    logger.info("CHECKPOINT: Model categorization tests passed")
    return True

# Run tests if this module is executed directly
if __name__ == "__main__":
    test_model_categorization()
