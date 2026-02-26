"""
Database initialization script for Ollama Workbench.
Creates and populates the necessary database tables.
"""
import sqlite3
import json
import os

def init_db():
    """Initialize the database with required tables."""
    conn = sqlite3.connect('ollama_models.db')
    cursor = conn.cursor()

    # Create models table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS models (
        model_name TEXT PRIMARY KEY,
        description TEXT,
        capabilities TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Default model descriptions
    default_models = {
        "llama2": {
            "description": "Meta's Llama 2 model, optimized for chat and general text generation",
            "capabilities": "General text generation, chat, code completion, reasoning"
        },
        "mistral": {
            "description": "Mistral AI's powerful language model",
            "capabilities": "Chat, text generation, analysis, code completion"
        },
        "codellama": {
            "description": "Specialized variant of Llama 2 focused on code generation",
            "capabilities": "Code completion, code explanation, debugging, technical documentation"
        },
        "neural-chat": {
            "description": "Intel's optimized chat model",
            "capabilities": "Conversational AI, text generation, task assistance"
        },
        "starling-lm": {
            "description": "Berkeley's instruction-tuned language model",
            "capabilities": "Chat, instruction following, reasoning, analysis"
        },
        "dolphin-phi": {
            "description": "Microsoft's Phi-2 model fine-tuned for chat",
            "capabilities": "Chat, reasoning, code generation, task completion"
        }
    }

    # Insert default models
    for model_name, info in default_models.items():
        cursor.execute('''
        INSERT OR REPLACE INTO models (model_name, description, capabilities)
        VALUES (?, ?, ?)
        ''', (model_name, info["description"], info["capabilities"]))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
