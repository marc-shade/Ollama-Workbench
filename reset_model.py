#!/usr/bin/env python
"""
Reset the model settings to fix the model selection issue.
This script will remove the saved settings file and create a new one with default values.
"""

import json
import os
import sys

# Settings file path
SETTINGS_FILE = "chat-settings.json"

def reset_model_settings():
    """Reset the model settings file."""
    # Check if settings file exists
    if os.path.exists(SETTINGS_FILE):
        print(f"Removing existing settings file: {SETTINGS_FILE}")
        os.remove(SETTINGS_FILE)
    
    # Create new settings with default values
    default_settings = {
        "selected_model": None,  # Allow the app to select first available model
        "agent_type": "None",
        "metacognitive_type": "None",
        "voice_type": "None",
        "selected_corpus": "None",
        "temperature_slider_chat": 0.7,
        "max_tokens_slider_chat": 4000,
        "presence_penalty_slider_chat": 0.0,
        "frequency_penalty_slider_chat": 0.0,
        "episodic_memory_enabled": False,
        "advanced_thinking_enabled": False,
        "thinking_steps": [
            "1. Analyzing the problem",
            "2. Breaking down into subtasks",
            "3. Exploring potential solutions",
            "4. Evaluating approaches",
            "5. Formulating a comprehensive answer"
        ],
        "instance_adaptive_cot_enabled": False,
        "cot_strategy": "IAP-ss",
        "cot_threshold": 0.5,
        "cot_top_n": 3
    }
    
    # Write new settings file
    with open(SETTINGS_FILE, "w") as f:
        json.dump(default_settings, f, indent=2)
    
    print(f"Created new settings file with default values.")
    print(f"Next time you start the app, you'll need to select and save your preferred model.")
    
    return True

if __name__ == "__main__":
    reset_model_settings()