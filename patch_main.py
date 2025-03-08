#!/usr/bin/env python3
"""
Script to patch main.py to handle the case where streamlit_option_menu is not available.
This creates a fallback implementation of option_menu that uses streamlit's built-in
selectbox component instead.
"""

import sys
import os

def create_backup(file_path):
    """Create a backup of the file."""
    backup_path = f"{file_path}.bak"
    if not os.path.exists(backup_path):
        with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
        print(f"Created backup at {backup_path}")
    else:
        print(f"Backup already exists at {backup_path}")

def patch_main_py(file_path):
    """Patch main.py to handle missing streamlit_option_menu."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if the file has already been patched
    if "# Fallback implementation for streamlit_option_menu" in content:
        print("main.py has already been patched for streamlit_option_menu")
    else:
        # Find the import line
        import_line = "from streamlit_option_menu import option_menu"
        if import_line not in content:
            print(f"Could not find import line: {import_line}")
        else:
            # Replace the import line with the fallback implementation
            fallback_code = """try:
    from streamlit_option_menu import option_menu
except ImportError:
    print("Warning: streamlit_option_menu not found, using fallback implementation")
    # Fallback implementation for streamlit_option_menu
    def option_menu(menu_title, options, icons=None, menu_icon=None, default_index=0, styles=None):
        import streamlit as st
        selected = st.selectbox(
            menu_title if menu_title else "Menu",
            options,
            index=default_index,
        )
        return selected"""
            
            content = content.replace(import_line, fallback_code)
            print("Patched main.py for streamlit_option_menu")
    
    # Write the patched content back to the file
    with open(file_path, 'w') as f:
        f.write(content)

def patch_openai_utils(file_path):
    """Patch openai_utils.py to handle missing openai package."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return
    
    # Create a backup of the file
    create_backup(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if the file has already been patched
    if "# Fallback implementation for openai" in content:
        print("openai_utils.py has already been patched")
        return
    
    # Find the import line
    import_line = "from openai import OpenAI"
    if import_line not in content:
        print(f"Could not find import line: {import_line}")
        return
    
    # Replace the import line with the fallback implementation
    fallback_code = """try:
    from openai import OpenAI
except ImportError:
    print("Warning: openai package not found, using fallback implementation")
    # Fallback implementation for openai
    class OpenAI:
        def __init__(self, api_key=None):
            self.api_key = api_key
            
        class Completion:
            @staticmethod
            def create(*args, **kwargs):
                return {"choices": [{"text": "OpenAI API not available - please install the openai package"}]}
        
        class ChatCompletion:
            @staticmethod
            def create(*args, **kwargs):
                return {"choices": [{"message": {"content": "OpenAI API not available - please install the openai package"}}]}
                
        class Embedding:
            @staticmethod
            def create(*args, **kwargs):
                return {"data": [{"embedding": [0.0] * 1536}]}
                
        completion = Completion()
        chat = ChatCompletion()
        embeddings = Embedding()"""
    
    patched_content = content.replace(import_line, fallback_code)
    
    # Write the patched content back to the file
    with open(file_path, 'w') as f:
        f.write(patched_content)
    
    print(f"Successfully patched {file_path} for openai")

def patch_groq_utils(file_path):
    """Patch groq_utils.py to handle missing sentence_transformers package."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return
    
    # Create a backup of the file
    create_backup(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if the file has already been patched
    if "# Fallback implementation for sentence_transformers" in content:
        print("groq_utils.py has already been patched")
        return
    
    # Find the import line
    import_line = "from sentence_transformers import SentenceTransformer"
    if import_line not in content:
        print(f"Could not find import line: {import_line}")
        return
    
    # Replace the import line with the fallback implementation
    fallback_code = """try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence_transformers package not found, using fallback implementation")
    # Fallback implementation for sentence_transformers
    class SentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name
            print(f"Warning: Using fallback SentenceTransformer with model: {model_name}")
            
        def encode(self, text, **kwargs):
            import numpy as np
            print(f"Warning: Using fallback encoding for text: {text[:50]}...")
            # Return a random embedding vector of size 384 (same as all-MiniLM-L6-v2)
            return np.random.rand(384)"""
    
    patched_content = content.replace(import_line, fallback_code)
    
    # Write the patched content back to the file
    with open(file_path, 'w') as f:
        f.write(patched_content)
    
    print(f"Successfully patched {file_path} for sentence_transformers")

def patch_tts_utils(file_path):
    """Patch tts_utils.py to handle missing gtts package."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return
    
    # Create a backup of the file
    create_backup(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if the file has already been patched
    if "# Fallback implementation for gtts" in content:
        print("tts_utils.py has already been patched")
        return
    
    # Find the import line
    import_line = "from gtts import gTTS"
    if import_line not in content:
        print(f"Could not find import line: {import_line}")
        return
    
    # Replace the import line with the fallback implementation
    fallback_code = """try:
    from gtts import gTTS
except ImportError:
    print("Warning: gtts package not found, using fallback implementation")
    # Fallback implementation for gtts
    class gTTS:
        def __init__(self, text="", lang="en", **kwargs):
            self.text = text
            self.lang = lang
            print(f"Warning: Using fallback gTTS with text: {text[:50]}...")
            
        def save(self, filename):
            print(f"Warning: Saving fallback speech to {filename}")
            # Create an empty file
            with open(filename, 'w') as f:
                f.write("# This is a placeholder file created by the fallback gTTS implementation")
            return filename"""
    
    patched_content = content.replace(import_line, fallback_code)
    
    # Also patch pygame import
    pygame_import = "import pygame"
    if pygame_import in content:
        pygame_fallback = """try:
    import pygame
except ImportError:
    print("Warning: pygame package not found, using fallback implementation")
    # Fallback implementation for pygame
    class pygame:
        class mixer:
            @staticmethod
            def init():
                print("Warning: Using fallback pygame.mixer.init()")
                
            @staticmethod
            def quit():
                print("Warning: Using fallback pygame.mixer.quit()")
                
            class music:
                @staticmethod
                def load(filename):
                    print(f"Warning: Using fallback pygame.mixer.music.load({filename})")
                    
                @staticmethod
                def play():
                    print("Warning: Using fallback pygame.mixer.music.play()")
                    
                @staticmethod
                def get_busy():
                    # Return False to exit the playback loop immediately
                    return False
                    
        class time:
            class Clock:
                def tick(self, framerate):
                    pass"""
        
        patched_content = patched_content.replace(pygame_import, pygame_fallback)
    
    # Write the patched content back to the file
    with open(file_path, 'w') as f:
        f.write(patched_content)
    
    print(f"Successfully patched {file_path} for gtts and pygame")

def patch_brainstorm(file_path):
    """Patch brainstorm.py to handle missing autogen package."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return
    
    # Create a backup of the file
    create_backup(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if the file has already been patched
    if "# Fallback implementation for autogen" in content:
        print("brainstorm.py has already been patched")
        return
    
    # Find the import line
    import_line = "from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager"
    if import_line not in content:
        print(f"Could not find import line: {import_line}")
        return
    
    # Replace the import line with the fallback implementation
    fallback_code = """try:
    from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager
except ImportError:
    print("Warning: autogen package not found, using fallback implementation")
    # Fallback implementation for autogen
    class ConversableAgent:
        def __init__(self, name, llm_config=None, human_input_mode=None, **kwargs):
            self.name = name
            self.llm_config = llm_config
            print(f"Warning: Using fallback ConversableAgent with name: {name}")
            
        def generate_reply(self, messages, sender, config=None):
            print(f"Warning: Using fallback generate_reply for {self.name}")
            return f"This is a fallback response from {self.name}. The autogen package is not installed."
    
    class UserProxyAgent(ConversableAgent):
        def __init__(self, name, human_input_mode=None, code_execution_config=None, **kwargs):
            super().__init__(name=name, **kwargs)
            self.human_input_mode = human_input_mode
            self.code_execution_config = code_execution_config
    
    class GroupChat:
        def __init__(self, agents, messages=None, speaker_selection_method=None):
            self.agents = agents
            self.messages = messages or []
            self.speaker_selection_method = speaker_selection_method
    
    class GroupChatManager:
        def __init__(self, groupchat):
            self.groupchat = groupchat"""
    
    patched_content = content.replace(import_line, fallback_code)
    
    # Also patch the Teachability import
    teachability_import = "from autogen.agentchat.contrib.capabilities.teachability import Teachability"
    if teachability_import in patched_content:
        teachability_fallback = """try:
    from autogen.agentchat.contrib.capabilities.teachability import Teachability
except ImportError:
    print("Warning: autogen.agentchat.contrib.capabilities.teachability package not found, using fallback implementation")
    # Fallback implementation for Teachability
    class Teachability:
        def __init__(self, path_to_db_dir=None):
            self.path_to_db_dir = path_to_db_dir
            print(f"Warning: Using fallback Teachability with path: {path_to_db_dir}")
            
        def add_to_agent(self, agent):
            print(f"Warning: Using fallback add_to_agent for {agent.name}")
            pass"""
        
        patched_content = patched_content.replace(teachability_import, teachability_fallback)
    
    # Write the patched content back to the file
    with open(file_path, 'w') as f:
        f.write(patched_content)
    
    print(f"Successfully patched {file_path} for autogen")

def patch_web_to_corpus(file_path):
    """Patch web_to_corpus.py to handle missing fake_useragent package."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return
    
    # Create a backup of the file
    create_backup(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if the file has already been patched
    if "# Fallback implementation for fake_useragent" in content:
        print("web_to_corpus.py has already been patched")
        return
    
    # Find the import line
    import_line = "from fake_useragent import UserAgent"
    if import_line not in content:
        print(f"Could not find import line: {import_line}")
        return
    
    # Replace the import line with the fallback implementation
    fallback_code = """try:
    from fake_useragent import UserAgent
except ImportError:
    print("Warning: fake_useragent package not found, using fallback implementation")
    # Fallback implementation for fake_useragent
    class UserAgent:
        def __init__(self):
            self.user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
            ]
            print("Warning: Using fallback UserAgent")
            
        @property
        def random(self):
            import random
            return random.choice(self.user_agents)"""
    
    patched_content = content.replace(import_line, fallback_code)
    
    # Write the patched content back to the file
    with open(file_path, 'w') as f:
        f.write(patched_content)
    
    print(f"Successfully patched {file_path} for fake_useragent")

def patch_pull_model(file_path):
    """Patch pull_model.py to handle missing humanize package."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return

    # Create a backup of the file
    create_backup(file_path)

    with open(file_path, 'r') as f:
        content = f.read()

    # Check if the file has already been patched
    if "# Fallback implementation for humanize" in content:
        print("pull_model.py has already been patched")
        return

    # Find the import line
    import_line = "import humanize"
    if import_line not in content:
        print(f"Could not find import line: {import_line}")
        return

    # Replace the import line with the fallback implementation
    fallback_code = """try:
    import humanize
except ImportError:
    print("Warning: humanize package not found, using fallback implementation")
    # Fallback implementation for humanize
    def format_size(size_bytes: int) -> str:
        \"\"\"Format the file size in a human-readable format (fallback).\"\"\"
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    humanize = type('humanize', (object,), {'naturalsize': format_size})()"""

    patched_content = content.replace(import_line, fallback_code)

    # Write the patched content back to the file
    with open(file_path, 'w') as f:
        f.write(patched_content)

    print(f"Successfully patched {file_path} for humanize")
    
def patch_file(file_path):
    """Patch the file based on its name."""
    if os.path.basename(file_path) == "main.py":
        patch_main_py(file_path)
    elif os.path.basename(file_path) == "openai_utils.py":
        patch_openai_utils("openai_utils.py")
    elif os.path.basename(file_path) == "groq_utils.py":
        patch_groq_utils("groq_utils.py")
    elif os.path.basename(file_path) == "tts_utils.py":
        patch_tts_utils("tts_utils.py")
    elif os.path.basename(file_path) == "brainstorm.py":
        patch_brainstorm("brainstorm.py")
    elif os.path.basename(file_path) == "web_to_corpus.py":
        patch_web_to_corpus("web_to_corpus.py")
    else:
        print(f"No specific patch available for {file_path}")

def main():
    """Main function."""
    main_py_path = "main.py"
    openai_utils_path = "openai_utils.py"
    groq_utils_path = "groq_utils.py"
    tts_utils_path = "tts_utils.py"
    brainstorm_path = "brainstorm.py"
    web_to_corpus_path = "web_to_corpus.py"
    
    if not os.path.exists(main_py_path):
        print(f"Error: {main_py_path} not found")
        return False
    
    # Create a backup of main.py
    create_backup(main_py_path)
    
    # Patch main.py
    patch_file(main_py_path)
    
    # Check if openai_utils.py exists
    if os.path.exists(openai_utils_path):
        # Create a backup of openai_utils.py
        create_backup(openai_utils_path)
        
        # Patch openai_utils.py
        patch_file(openai_utils_path)
    else:
        print(f"Warning: {openai_utils_path} not found, skipping")
    
    # Check if groq_utils.py exists
    if os.path.exists(groq_utils_path):
        # Create a backup of groq_utils.py
        create_backup(groq_utils_path)
        
        # Patch groq_utils.py
        patch_file(groq_utils_path)
    else:
        print(f"Warning: {groq_utils_path} not found, skipping")
    
    # Check if tts_utils.py exists
    if os.path.exists(tts_utils_path):
        # Create a backup of tts_utils.py
        create_backup(tts_utils_path)
        
        # Patch tts_utils.py
        patch_file(tts_utils_path)
    else:
        print(f"Warning: {tts_utils_path} not found, skipping")
    
    # Check if brainstorm.py exists
    if os.path.exists(brainstorm_path):
        # Create a backup of brainstorm.py
        create_backup(brainstorm_path)
        
        # Patch brainstorm.py
        patch_file(brainstorm_path)
    else:
        print(f"Warning: {brainstorm_path} not found, skipping")
    
    # Check if web_to_corpus.py exists
    if os.path.exists(web_to_corpus_path):
        # Create a backup of web_to_corpus.py
        create_backup(web_to_corpus_path)
        
        # Patch web_to_corpus.py
        patch_file(web_to_corpus_path)
    else:
        print(f"Warning: {web_to_corpus_path} not found, skipping")
    
    # Check if web_to_corpus.py exists
    if os.path.exists(web_to_corpus_path):
        # Create a backup of web_to_corpus.py
        create_backup(web_to_corpus_path)
        
        # Patch web_to_corpus.py
        patch_file(web_to_corpus_path)
    else:
        print(f"Warning: {web_to_corpus_path} not found, skipping")
    
    # Check if web_to_corpus.py exists
    if os.path.exists(web_to_corpus_path):
        # Create a backup of web_to_corpus.py
        create_backup(web_to_corpus_path)
        
        # Patch web_to_corpus.py
        patch_file(web_to_corpus_path)
    else:
        print(f"Warning: {web_to_corpus_path} not found, skipping")
    
    pull_model_path = "pull_model.py"
    # Check if pull_model.py exists
    if os.path.exists(pull_model_path):
        # Create a backup of pull_model.py
        create_backup(pull_model_path)

        # Patch pull_model.py
        patch_file(pull_model_path) # Or patch_pull_model(pull_model_path) - let's use patch_file for consistency
    else:
        print(f"Warning: {pull_model_path} not found, skipping")

    print("\nPatch complete. You can now run the application with:")
    print("streamlit run main.py")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)