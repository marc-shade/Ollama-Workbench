"""
Integrated Main Application for Ollama Workbench

This file integrates the modern_chat_interface with the full Ollama Workbench functionality,
preserving all original features while using the modern UI.
"""

import os
import json
import logging
import subprocess
import threading
import importlib.util

import streamlit as st
from flask import Flask, jsonify

# Function to check if module is available without importing it
def is_module_available(module_name):
    """Check if a module can be imported without importing it"""
    return importlib.util.find_spec(module_name) is not None

# Setup logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("integrated_main")

# Set page config for wide layout with maximum width
st.set_page_config(layout="wide", page_title="Ollama Workbench", page_icon="🦙")

# Add CSS to ensure 100% width on all pages
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

# Import core modules with error handling
try:
    from streamlit_option_menu import option_menu
    logger.info("CHECKPOINT: Successfully imported streamlit_option_menu")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing streamlit_option_menu: {str(e)}")
    st.error("Missing required dependency: streamlit-option-menu")
    st.info("Please install it with: pip install streamlit-option-menu")

# Import our robust ollama utilities module first
try:
    # Try to import from the robust module first
    from robust_ollama_utils import (
        load_api_keys, get_ollama_resource_usage,
        call_ollama_endpoint, get_available_models, get_all_models,
        save_ai_content_to_workspace, pull_model, remove_model, show_model_info,
        get_server_logs, start_server, stop_server
    )
    logger.info("CHECKPOINT: Successfully imported robust_ollama_utils with complete functionality")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing simplified_ollama_utils: {str(e)}")
    
    # Fall back to the original module
    try:
        from ollama_utils import (
            load_api_keys, get_ollama_resource_usage,
            call_ollama_endpoint, get_available_models, get_all_models
        )
        logger.info("CHECKPOINT: Successfully imported ollama_utils")
    except ImportError as e2:
        logger.error(f"CHECKPOINT: Error importing ollama_utils: {str(e2)}")
        st.error("Missing required ollama utilities module. Using fallback implementations.")
        
        # Create fallback implementations for critical functions
        def load_api_keys():
            """Fallback implementation for loading API keys."""
            logger.info("CHECKPOINT: Using fallback load_api_keys function")
            try:
                import json
                import os
                api_keys_file = "api_keys.json"
                if os.path.exists(api_keys_file):
                    with open(api_keys_file, "r") as f:
                        return json.load(f)
                return {}
            except Exception as e:
                logger.error(f"CHECKPOINT: Error in fallback load_api_keys: {str(e)}")
                return {}
        
        def get_ollama_resource_usage():
            """Fallback implementation for getting Ollama resource usage."""
            logger.info("CHECKPOINT: Using fallback get_ollama_resource_usage function")
            try:
                import psutil
                return {
                    "cpu": psutil.cpu_percent(),
                    "memory": psutil.virtual_memory().percent
                }
            except Exception as e:
                logger.error(f"CHECKPOINT: Error in fallback get_ollama_resource_usage: {str(e)}")
                return {"cpu": 0, "memory": 0}
        
        def get_available_models():
            """Fallback implementation for getting available models."""
            logger.info("CHECKPOINT: Using fallback get_available_models function")
            return ["llama3", "llama3:8b", "mistral", "gemma", "phi"]
            
        def get_all_models():
            """Fallback implementation for getting all models."""
            logger.info("CHECKPOINT: Using fallback get_all_models function")
            return get_available_models()
        
        def call_ollama_endpoint(model, prompt=None, image=None, temperature=0.5, max_tokens=150, **kwargs):
            """Fallback implementation for calling Ollama endpoint."""
            logger.info("CHECKPOINT: Using fallback call_ollama_endpoint function")
            try:
                import ollama
                import requests
                
                # Try to use the ollama client library
                try:
                    client = ollama.Client(host="http://localhost:11434")
                    response = client.generate(model=model, prompt=prompt, options={"temperature": temperature})
                    return response.get("response", "Fallback response: Unable to connect to Ollama"), None, None, None
                except Exception as client_error:
                    logger.error(f"CHECKPOINT: Error using ollama client: {str(client_error)}")
                    
                    # Fallback to direct API call
                    try:
                        response = requests.post(
                            "http://localhost:11434/api/generate",
                            json={"model": model, "prompt": prompt, "temperature": temperature}
                        )
                        if response.status_code == 200:
                            return response.json().get("response", "Fallback response: Unable to parse Ollama response"), None, None, None
                    except Exception as api_error:
                        logger.error(f"CHECKPOINT: Error in direct API call: {str(api_error)}")
                
                return "Fallback response: Unable to connect to Ollama server. Please ensure Ollama is running.", None, None, None
            except Exception as e:
                logger.error(f"CHECKPOINT: Error in fallback call_ollama_endpoint: {str(e)}")
                return "Fallback response: Critical error in Ollama communication.", None, None, None

# Import model tests
try:
    # Import specific functions instead of using wildcard imports
    from model_tests import (
        test_model_json, test_model_function_calling,
        test_model_context_window
    )
    logger.info("CHECKPOINT: Successfully imported model_tests")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing model_tests: {str(e)}")

# Check for critical dependencies before importing modules
autogen_available = is_module_available("autogen")
pdfkit_available = is_module_available("pdfkit")
json_schema_available = is_module_available("json_schema_for_humans")

logger.info(f"CHECKPOINT: Critical dependencies check - autogen: {autogen_available}, pdfkit: {pdfkit_available}, json_schema: {json_schema_available}")

# Import UI elements with special handling for dependencies
try:
    # First try importing autogen if needed
    if not autogen_available:
        logger.error("CHECKPOINT: autogen module not available")
    
    # Check for pdfkit (needed for PDF generation)
    if not pdfkit_available:
        logger.error("CHECKPOINT: pdfkit module not available")
        # Inform user about wkhtmltopdf requirement
        st.warning("pdfkit module requires wkhtmltopdf to be installed on your system.")
        with st.expander("How to install wkhtmltopdf"):
            st.markdown("### Installing wkhtmltopdf (required by pdfkit)")
            st.markdown("On macOS: `brew install wkhtmltopdf`")
            st.markdown("On Ubuntu/Debian: `sudo apt-get install wkhtmltopdf`")
            st.markdown("On CentOS/RHEL: `sudo yum install wkhtmltopdf`")
    
    # Check for json-schema-for-humans
    if not json_schema_available:
        logger.warning("CHECKPOINT: json-schema-for-humans module not available")
        st.warning("json-schema-for-humans package is not installed. Schema visualization will be limited.")
    
    # Now import UI elements with knowledge of dependency availability
    try:
        from ui_elements import (
            model_comparison_test, contextual_response_test, feature_test,
            list_local_models, pull_models, show_model_details, remove_model_ui,
            vision_comparison_test, update_models, files_tab,
            server_configuration, server_monitoring
        )
        logger.info("CHECKPOINT: Successfully imported ui_elements")
    except ImportError as e:
        missing_deps = []
        if not autogen_available and "autogen" in str(e): missing_deps.append("autogen")
        if not pdfkit_available and "pdfkit" in str(e): missing_deps.append("pdfkit")
        if not json_schema_available and "json_schema_for_humans" in str(e): missing_deps.append("json-schema-for-humans")
        
        if missing_deps:
            logger.error(f"CHECKPOINT: UI elements import failed due to missing dependencies: {', '.join(missing_deps)}")
            # Define fallback functions for UI elements
            def model_comparison_test(): st.error(f"Feature unavailable due to missing dependencies: {', '.join(missing_deps)}")
            def contextual_response_test(): st.error(f"Feature unavailable due to missing dependencies: {', '.join(missing_deps)}")
            def feature_test(): st.error(f"Feature unavailable due to missing dependencies: {', '.join(missing_deps)}")
            def list_local_models(): st.error(f"Feature unavailable due to missing dependencies: {', '.join(missing_deps)}")
            def pull_models(): st.error(f"Feature unavailable due to missing dependencies: {', '.join(missing_deps)}")
            def show_model_details(): st.error(f"Feature unavailable due to missing dependencies: {', '.join(missing_deps)}")
            def remove_model_ui(): st.error(f"Feature unavailable due to missing dependencies: {', '.join(missing_deps)}")
            def vision_comparison_test(): st.error(f"Feature unavailable due to missing dependencies: {', '.join(missing_deps)}")
            def update_models(): st.error(f"Feature unavailable due to missing dependencies: {', '.join(missing_deps)}")
            def files_tab(): st.error(f"Feature unavailable due to missing dependencies: {', '.join(missing_deps)}")
            def server_configuration(): st.error(f"Feature unavailable due to missing dependencies: {', '.join(missing_deps)}")
            def server_monitoring(): st.error(f"Feature unavailable due to missing dependencies: {', '.join(missing_deps)}")
        else:
            logger.error(f"CHECKPOINT: Error importing ui_elements: {str(e)}")
            raise
except Exception as e:
    logger.error(f"CHECKPOINT: Unexpected error importing UI elements: {str(e)}")
    st.error(f"Error loading UI elements: {str(e)}\n\nPlease run ./run_integrated_workbench.sh to install all required dependencies.")

# Import the modern chat interface
try:
    from modern_chat_interface import modern_chat_interface
    logger.info("CHECKPOINT: Successfully imported modern_chat_interface")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing modern_chat_interface: {str(e)}")
    def modern_chat_interface():
        st.error("Modern chat interface is unavailable.")
        st.info("Please ensure modern_chat_interface.py exists in the project directory.")

# Import styles
try:
    from styles import apply_styles
    logger.info("CHECKPOINT: Successfully imported styles")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing styles: {str(e)}")
    # Define fallback colors
    colors = {
        "primary": "#FF4B4B",
        "hover": "#FF6B6B",
        "selected": "#FF8080",
        "selected_text": "#FFFFFF"
    }
    theme = "light"
    def apply_styles():
        return colors, theme

# Import other UI modules with error handling
try:
    from simplified_rag import enhanced_rag_interface
    logger.info("CHECKPOINT: Successfully imported simplified_rag")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing simplified_rag: {str(e)}")
    def enhanced_rag_interface():
        st.error("Enhanced RAG interface is unavailable.")
        st.info("Missing required dependencies for RAG functionality.")

try:
    from model_onboarding import onboarding_test_process
    logger.info("CHECKPOINT: Successfully imported model_onboarding")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing model_onboarding: {str(e)}")
    def onboarding_test_process():
        st.error("Model onboarding is unavailable.")

try:
    from multimodal_chat import multimodal_chat_interface
    logger.info("CHECKPOINT: Successfully imported multimodal_chat")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing multimodal_chat: {str(e)}")
    def multimodal_chat_interface():
        st.error("Multimodal chat is unavailable.")
        st.info("Missing required dependencies for multimodal functionality.")

try:
    from multimodel_chat import multimodel_chat_app
    logger.info("CHECKPOINT: Successfully imported multimodel_chat")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing multimodel_chat: {str(e)}")
    def multimodel_chat_app():
        st.error("Multi-model chat is unavailable.")

try:
    from voice_interface import voice_chat_interface, voice_settings_ui
    logger.info("CHECKPOINT: Successfully imported voice_interface")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing voice_interface: {str(e)}")
    def voice_chat_interface():
        st.error("Voice chat is unavailable.")
        st.info("Missing required TTS dependencies.")
    def voice_settings_ui():
        st.error("Voice settings are unavailable.")

# Import tool_playground with error handling to prevent UI failures
try:
    from tool_playground import tool_playground
except Exception as e:
    st.error(f"Error loading Tool Playground module: {str(e)}")
    # Define a fallback function
    def tool_playground():
        st.error("Tool Playground is currently unavailable.")
        st.info("Try restarting the application or check the logs for errors.")

# Import additional feature modules with error handling
try:
    from structured_output import structured_output_ui
    logger.info("CHECKPOINT: Successfully imported structured_output")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing structured_output: {str(e)}")
    def structured_output_ui():
        st.error("Structured output UI is unavailable.")

try:
    from openai_compatibility import openai_compatibility_ui
    logger.info("CHECKPOINT: Successfully imported openai_compatibility")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing openai_compatibility: {str(e)}")
    def openai_compatibility_ui():
        st.error("OpenAI compatibility UI is unavailable.")

try:
    from model_capabilities import model_capabilities_ui
    logger.info("CHECKPOINT: Successfully imported model_capabilities")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing model_capabilities: {str(e)}")
    def model_capabilities_ui():
        st.error("Model capabilities UI is unavailable.")

try:
    from test_visualization import test_visualization_ui
    logger.info("CHECKPOINT: Successfully imported test_visualization")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing test_visualization: {str(e)}")
    def test_visualization_ui():
        st.error("Test visualization UI is unavailable.")

try:
    from repo_docs import main as repo_docs_main
    logger.info("CHECKPOINT: Successfully imported repo_docs")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing repo_docs: {str(e)}")
    def repo_docs_main():
        st.error("Repository analyzer is unavailable.")

try:
    from web_to_corpus import main as web_to_corpus_main
    logger.info("CHECKPOINT: Successfully imported web_to_corpus")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing web_to_corpus: {str(e)}")
    def web_to_corpus_main():
        st.error("Web crawler is unavailable.")

try:
    from welcome import display_welcome_message
    logger.info("CHECKPOINT: Successfully imported welcome")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing welcome: {str(e)}")
    def display_welcome_message():
        st.title("Welcome to Ollama Workbench")
        st.write("The welcome module could not be loaded.")

try:
    from projects import projects_main
    logger.info("CHECKPOINT: Successfully imported projects")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing projects: {str(e)}")
    def projects_main():
        st.error("Projects feature is unavailable.")

try:
    from prompts import manage_prompts, get_agent_prompt, get_metacognitive_prompt, get_voice_prompt, get_identity_prompt
    logger.info("CHECKPOINT: Successfully imported prompts")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing prompts: {str(e)}")
    def manage_prompts():
        st.error("Prompts management is unavailable.")
    def get_agent_prompt():
        return {}
    def get_metacognitive_prompt():
        return {}
    def get_voice_prompt():
        return {}
    def get_identity_prompt():
        return {}
# Import brainstorm with robust error handling to prevent UI failures
try:
    # Using previously set autogen_available variable from dependency check
    if autogen_available:
        logger.info("CHECKPOINT: Verified autogen is available for brainstorm module")
    
    # Only try to import brainstorm if autogen is available
    if autogen_available:
        try:
            from brainstorm import brainstorm_interface
            logger.info("CHECKPOINT: Successfully imported brainstorm module")
        except ImportError as e:
            if "autogen" in str(e):
                logger.error(f"CHECKPOINT: Error importing brainstorm module due to autogen issues: {str(e)}")
                raise ImportError("Autogen dependency issue detected")
            else:
                logger.error(f"CHECKPOINT: Error importing brainstorm module: {str(e)}")
                raise
    else:
        raise ImportError("Required dependency 'autogen' not available")
        
except Exception as e:
    logger.error(f"CHECKPOINT: Error setting up brainstorm module: {str(e)}")
    # Define a fallback function with detailed troubleshooting information
    def brainstorm_interface():
        st.error("Brainstorm feature is currently unavailable.")
        st.info("The brainstorm feature requires the 'pyautogen' package.")
        
        # Show detailed troubleshooting information
        with st.expander("Troubleshooting Information"):
            st.markdown("### Dependency Installation")
            st.markdown("Try running the following commands to install the required dependencies:")
            st.code("pip uninstall -y pyautogen\npip install --no-cache-dir pyautogen", language="bash")
            
            st.markdown("### Verify Installation")
            st.markdown("You can verify the installation with:")
            st.code("python -c 'import autogen; print(autogen.__version__)'\n", language="bash")
            
            st.markdown("### Alternative Solution")
            st.markdown("Run the integrated workbench script which will handle dependencies:")
            st.code("./run_integrated_workbench.sh", language="bash")
        
        logger.warning("CHECKPOINT: Using fallback brainstorm_interface function")
# Import research and related modules
try:
    from research import research_interface
    logger.info("CHECKPOINT: Successfully imported research")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing research: {str(e)}")
    def research_interface():
        st.error("Research interface is unavailable.")

try:
    from enhanced_corpus import enhance_corpus_ui
    logger.info("CHECKPOINT: Successfully imported enhanced_corpus")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing enhanced_corpus: {str(e)}")
    def enhance_corpus_ui():
        st.error("Enhanced corpus UI is unavailable.")

try:
    from build import build_interface
    logger.info("CHECKPOINT: Successfully imported build")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing build: {str(e)}")
    def build_interface():
        st.error("Build interface is unavailable.")

# Import API utilities
try:
    from openai_utils import call_openai_api
    logger.info("CHECKPOINT: Successfully imported openai_utils")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing openai_utils: {str(e)}")
    def call_openai_api(*args, **kwargs):
        return "OpenAI API is unavailable."

try:
    from groq_utils import call_groq_api
    logger.info("CHECKPOINT: Successfully imported groq_utils")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing groq_utils: {str(e)}")
    def call_groq_api(*args, **kwargs):
        return "Groq API is unavailable."

# Import additional UI modules
try:
    from nodes import nodes_interface
    logger.info("CHECKPOINT: Successfully imported nodes")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing nodes: {str(e)}")
    def nodes_interface():
        st.error("Nodes interface is unavailable.")

try:
    from external_providers import external_providers_ui
    logger.info("CHECKPOINT: Successfully imported external_providers")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing external_providers: {str(e)}")
    def external_providers_ui():
        st.error("External providers UI is unavailable.")

# Import database initialization
try:
    from db_init import init_db
    logger.info("CHECKPOINT: Successfully imported db_init")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing db_init: {str(e)}")
    def init_db():
        logger.warning("CHECKPOINT: Using fallback init_db function")
        st.warning("Database initialization is unavailable.")

# Import model management
try:
    from model_management import model_management_dashboard
    logger.info("CHECKPOINT: Successfully imported model_management")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing model_management: {str(e)}")
    def model_management_dashboard():
        st.error("Model management dashboard is unavailable.")

try:
    from performance_metrics import performance_metrics_interface
    logger.info("CHECKPOINT: Successfully imported performance_metrics")
except ImportError as e:
    logger.error(f"CHECKPOINT: Error importing performance_metrics: {str(e)}")
    def performance_metrics_interface():
        st.error("Performance metrics interface is unavailable.")

# Initialize the databases with error handling
try:
    logger.info("CHECKPOINT: Initializing main database")
    init_db()
    logger.info("CHECKPOINT: Main database initialized successfully")
except Exception as e:
    logger.error(f"CHECKPOINT: Error initializing main database: {str(e)}")
    st.error("Error initializing database. Some features may not work correctly.")

# Initialize model management database with error handling
try:
    logger.info("CHECKPOINT: Initializing model management database")
    from model_management import init_db as init_model_db
    init_model_db()
    logger.info("CHECKPOINT: Model management database initialized successfully")
except ImportError:
    logger.warning("CHECKPOINT: Model management module not available")
except Exception as e:
    logger.error(f"CHECKPOINT: Error initializing model management database: {str(e)}")

# Global variable to store the port number
ollama_port = None 

# Create a native messaging host manifest 
NATIVE_HOST_MANIFEST = "native_host_manifest.json" 
with open(NATIVE_HOST_MANIFEST, "w") as f:
    json.dump({
        "name": "ollama_workbench_host",
        "description": "Native messaging host for Ollama Workbench",
        "path": os.path.abspath(__file__), 
        "type": "stdio"
    }, f, indent=4)

# Apply modern styling from the styles module
colors, theme = apply_styles()

# Define constants for the sidebar sections
SIDEBAR_SECTIONS = {
    "Workflow": [
        ("Research", "Research"),
        ("Brainstorm", "Brainstorm"),
        ("Projects", "Projects"),
        ("CEF", "CEF"),
        ("Build", "Build"),
        ("Prompts", "Prompts"),
    ],
    "Document": [
        ("Repository Analyzer", "Repository Analyzer"),
        ("Web Crawler", "Web Crawler"),
        ("Corpus", "Corpus"),
        ("Manage Files", "Files"),
    ],
    "Maintain": [
        ("List Local Models", "List Local Models"),
        ("Model Management", "Model Management"),
        ("Model Information", "Show Model Information"),
        ("Pull a Model", "Pull a Model"),
        ("Remove a Model", "Remove a Model"),
        ("Update Models", "Update Models"),
        ("Server Configuration", "Server Configuration"),
        ("Server Monitoring", "Server Monitoring"),
        ("Performance Metrics", "Performance Metrics"),
        ("External Providers", "External Providers"),
        ("OpenAI Compatibility", "OpenAI Compatibility")
    ],
    "Test": [
        ("Model Onboarding", "Model Onboarding"),
        ("Model Feature Test", "Feature Test"),
        ("Response Quality", "Model Comparison"),
        ("Contextual Response", "Contextual Response"),
        ("Vision Models", "Vision"),
        ("Tool Calling", "Tool Playground"),
        ("Structured Output", "Structured Output"),
        ("Model Capabilities", "Model Capabilities"),
        ("Test Visualization", "Test Visualization"),
    ],
    "Help": [
        ("Help", "Help")
    ]
}

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "selected_test" not in st.session_state:
        st.session_state.selected_test = "Chat"
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "llama3"
    
    # Make sure current_model is synchronized with selected_model
    if "selected_model" in st.session_state and st.session_state.selected_model:
        if "current_model" not in st.session_state:
            st.session_state.current_model = st.session_state.selected_model
    elif "current_model" in st.session_state and st.session_state.current_model:
        st.session_state.selected_model = st.session_state.current_model
    
    # Initialize other session state variables
    if "show_resource_usage" not in st.session_state:
        st.session_state.show_resource_usage = False
    
    # Log initialization
    logger.info("CHECKPOINT: Session state initialized")
    for key in st.session_state:
        logger.info(f"CHECKPOINT: Session state {key} = {st.session_state[key]}")

def display_resource_usage_sidebar():
    """Display resource usage in the sidebar."""
    with st.sidebar:
        st.subheader("Resource Usage")
        try:
            usage = get_ollama_resource_usage()
            st.write(f"CPU: {usage['cpu']:.1f}%")
            st.write(f"Memory: {usage['memory']:.1f}%")
            if 'gpu' in usage:
                st.write(f"GPU: {usage['gpu']:.1f}%")
        except Exception as e:
            st.error(f"Error getting resource usage: {str(e)}")

def create_sidebar():
    """Create and populate the sidebar."""
    with st.sidebar:
        st.markdown(
            '<div style="text-align: left;">'
            '<h1 class="logo" style="font-size: 24px; font-weight: 300;">🦙 Ollama <span style="color: orange;">Workbench</span></h1>'
            "</div>",
            unsafe_allow_html=True,
        )

        if st.session_state.get("show_resource_usage", False):
            display_resource_usage_sidebar()

        # Define the main navigation menu with modern styling
        main_menu = option_menu(
            menu_title="",
            options=["Chat", "Multimodal Chat", "Multi-Model Chat", "Voice Chat", "Tool Playground", "Structured Output", "Enhanced RAG", "Collaborative Workspace"] + list(SIDEBAR_SECTIONS.keys()),
            icons=["chat", "image", "chat-square-text", "mic", "tools", "braces", "book", "pencil-square", "gear", "folder", "tools", "clipboard-check", "question-circle"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important"},
                "icon": {"font-size": "14px", "margin-right": "10px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "0.25rem 0",
                    "padding": "0.5rem 0.75rem",
                    "border-radius": "0.375rem",
                    "--primary-color": colors["primary"],
                    "--hover-color": colors["hover"],
                },
                "nav-link-selected": {
                    "font-weight": "500",
                    "background-color": colors["selected"],
                    "color": colors["selected_text"]
                },
            },
        )

        # Define the sub navigation menu based on the selected main menu
        if main_menu == "Chat":
            st.session_state.selected_test = "Chat"
        elif main_menu == "Multimodal Chat":
            st.session_state.selected_test = "Multimodal Chat"
        elif main_menu == "Multi-Model Chat":
            st.session_state.selected_test = "Multi-Model Chat"
        elif main_menu == "Voice Chat":
            st.session_state.selected_test = "Voice Chat"
        elif main_menu == "Tool Playground":
            st.session_state.selected_test = "Tool Playground"
        elif main_menu == "Structured Output":
            st.session_state.selected_test = "Structured Output"
        elif main_menu == "Enhanced RAG":
            st.session_state.selected_test = "Enhanced RAG"
        elif main_menu == "Collaborative Workspace":
            st.session_state.selected_test = "Collaborative Workspace"
        else:
            sub_menu = option_menu(
                menu_title=None,
                options=[option[1] for option in SIDEBAR_SECTIONS[main_menu]],
                default_index=0,
                styles={
                    "container": {"padding": "0!important"},
                    "icon": {"font-size": "12px"},
                    "nav-link": {
                        "font-size": "14px",
                        "text-align": "left",
                        "margin": "0px",
                        "--primary-color": "#1976D2",
                        "--hover-color": "#e16d6d",
                    },
                    "nav-link-selected": {"font-weight": "bold"},
                },
            )
            st.session_state.selected_test = sub_menu

def main_content():
    """Display the main content based on the selected test."""
    if 'bm_tasks' not in st.session_state:
        st.session_state.bm_tasks = []
    
    if st.session_state.selected_test == "Chat":
        # Use the modern chat interface that preserves all original functionality
        logger.info("CHECKPOINT: Loading modern chat interface")
        modern_chat_interface()
    elif st.session_state.selected_test == "Multimodal Chat":
        multimodal_chat_interface()
    elif st.session_state.selected_test == "Multi-Model Chat":
        multimodel_chat_app()
    elif st.session_state.selected_test == "Voice Chat":
        tab1, tab2 = st.tabs(["Voice Chat", "Voice Settings"])
        with tab1:
            voice_chat_interface()
        with tab2:
            voice_settings_ui()
    elif st.session_state.selected_test == "Tool Playground":
        tool_playground()
    elif st.session_state.selected_test == "Structured Output":
        structured_output_ui()
    elif st.session_state.selected_test == "Build":
        build_interface()
    elif st.session_state.selected_test == "Research":
        research_interface()
    elif st.session_state.selected_test == "Brainstorm":
        brainstorm_interface()
    elif st.session_state.selected_test == "Projects":
        projects_main()
    elif st.session_state.selected_test == "Prompts":
        manage_prompts()
    elif st.session_state.selected_test == "CEF":
        nodes_interface()
    elif st.session_state.selected_test == "Corpus":
        enhance_corpus_ui()
    elif st.session_state.selected_test == "Files":
        files_tab()
    elif st.session_state.selected_test == "Web Crawler":
        web_to_corpus_main()
    elif st.session_state.selected_test == "Repository Analyzer":
        repo_docs_main()
    elif st.session_state.selected_test == "List Local Models":
        list_local_models()
    elif st.session_state.selected_test == "Model Management":
        model_management_dashboard()
    elif st.session_state.selected_test == "Show Model Information":
        show_model_details()
    elif st.session_state.selected_test == "Pull a Model":
        pull_models()
    elif st.session_state.selected_test == "Remove a Model":
        remove_model_ui()
    elif st.session_state.selected_test == "Update Models":
        update_models()
    elif st.session_state.selected_test == "Server Configuration":
        server_configuration()
    elif st.session_state.selected_test == "Server Monitoring":
        server_monitoring()
    elif st.session_state.selected_test == "Performance Metrics":
        performance_metrics_interface()
    elif st.session_state.selected_test == "External Providers":
        external_providers_ui()
    elif st.session_state.selected_test == "OpenAI Compatibility":
        openai_compatibility_ui()
    elif st.session_state.selected_test == "Model Onboarding":
        onboarding_test_process()
    elif st.session_state.selected_test == "Feature Test":
        feature_test()
    elif st.session_state.selected_test == "Model Comparison":
        model_comparison_test()
    elif st.session_state.selected_test == "Contextual Response":
        contextual_response_test()
    elif st.session_state.selected_test == "Vision":
        vision_comparison_test()
    elif st.session_state.selected_test == "Tool Playground":
        tool_playground()
    elif st.session_state.selected_test == "Structured Output":
        structured_output_ui()
    elif st.session_state.selected_test == "Model Capabilities":
        model_capabilities_ui()
    elif st.session_state.selected_test == "Test Visualization":
        test_visualization_ui()
    elif st.session_state.selected_test == "Enhanced RAG":
        enhanced_rag_interface()
    elif st.session_state.selected_test == "Collaborative Workspace":
        # Import collaborative_workspace_ui here to avoid circular imports
        try:
            from collaborative_workspace import collaborative_workspace_ui
            logger.info("CHECKPOINT: Successfully imported collaborative_workspace")
            
            def model_callback(prompt):
                # Get the currently selected model
                model = st.session_state.get("selected_model", "llama3")
                # Use the model to generate a response
                try:
                    logger.info(f"CHECKPOINT: Generating response with model {model}")
                    response, _, _, _ = call_ollama_endpoint(
                        model=model,
                        prompt=prompt,
                        temperature=0.7,
                        max_tokens=4000
                    )
                    logger.info("CHECKPOINT: Response generated successfully")
                    return response
                except Exception as e:
                    logger.error(f"CHECKPOINT: Error generating response: {str(e)}")
                    return f"Error generating response: {str(e)}"
            
            collaborative_workspace_ui(model_callback=model_callback)
        except ImportError as e:
            logger.error(f"CHECKPOINT: Error importing collaborative_workspace: {str(e)}")
            st.error("Collaborative workspace is unavailable.")
            st.info("Missing required dependencies for collaborative workspace functionality.")
    elif st.session_state.selected_test == "Help":
        display_welcome_message()
    else:
        # Fallback to modern chat interface
        logger.info(f"CHECKPOINT: Unknown test selected: {st.session_state.selected_test}, falling back to modern chat interface")
        modern_chat_interface()

# Create a Flask app for the API
app = Flask(__name__)

@app.route('/prompts')
def get_prompts():
    """Returns a JSON with all prompt types."""
    return jsonify({
        "agent": get_agent_prompt(),
        "metacognitive": get_metacognitive_prompt(),
        "voice": get_voice_prompt(),
        "identity": get_identity_prompt()
    })

@app.route('/openai-key')
def get_openai_key():
    """Returns the OpenAI API key."""
    api_keys = load_api_keys()
    return jsonify({"openai_api_key": api_keys.get("openai_api_key")})

@app.route('/identifier')
def get_identifier():
    """Returns a unique identifier for Ollama Workbench."""
    return "Ollama Workbench" 

@app.route('/port')
def get_port():
    """Returns the dynamically allocated port number."""
    global ollama_port
    return str(ollama_port)

def main():
    """Main entry point for the application."""
    # Initialize session state
    initialize_session_state()
    
    # Create the sidebar
    create_sidebar()
    
    # Handle query parameters
    params = dict(st.query_params)
    web_page_content = params.get('web_page_content', [None])[0]
    if web_page_content:
        st.session_state.web_page_content = web_page_content

    # Display main content
    main_content()

    # Get parameters from URL
    params = dict(st.query_params)
    web_page_url = params.get('url', [None])[0]
    is_extension = params.get('extension', ['false'])[0].lower() == 'true'

    if is_extension and web_page_url:
        st.session_state.web_page_url = web_page_url
        st.session_state.is_extension = is_extension

def send_port_to_extension(port):
    """Sends the port number to the background script of the extension."""
    try:
        # Use the 'chrome-extension' protocol to send a message to the extension
        cmd = f'chrome-extension://{os.environ.get("EXTENSION_ID", "gddghhhklfnhijhhagfgnfiehidcdnba")}/background.js'
        message = {"message": "ollamaPort", "port": port}

        # Use subprocess to send the message. Requires 'npx' which comes with Node.js.
        process = subprocess.Popen(['npx', 'chrome-remote-interface', 'sendMessage', cmd, '--json', json.dumps(message)], 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if stderr:
            print(f"Error sending message to extension: {stderr.decode('utf-8')}") 
        else:
            print(f"Successfully sent port number to extension: {stdout.decode('utf-8')}")

    except Exception as e:
        print(f"Error sending message to extension: {e}")

def run_flask():
    """Run the Flask server."""
    global ollama_port
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    ollama_port = sock.getsockname()[1] 
    sock.close()
    app.run(port=ollama_port)
    print(f"Flask running on port: {ollama_port}") 

    # Send the port to the extension once Flask is running
    send_port_to_extension(ollama_port)

if __name__ == "__main__":
    # Start Flask before Streamlit
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start() 
    main()
