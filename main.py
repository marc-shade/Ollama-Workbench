# main.py

import streamlit as st
import threading
from flask import Flask, jsonify, request
import os
import subprocess
import json
import time
import logging

# Configure logging for the application (entry point - single basicConfig call)
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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

from streamlit_option_menu import option_menu
from ollama_workbench.chat.collaborative_workspace import collaborative_workspace_ui
from ollama_workbench.providers.ollama_utils import *
from ollama_workbench.models.model_tests import *
from ollama_workbench.core.session_utils import initialize_session_state
from ollama_workbench.models.model_comparison import model_comparison_test
from ollama_workbench.ui.contextual_response import contextual_response_test
from ollama_workbench.models.feature_test import feature_test
from ollama_workbench.models.local_models import list_local_models
from ollama_workbench.models.pull_model import pull_models
from ollama_workbench.models.show_model import show_model_details
from ollama_workbench.models.remove_model import remove_model_ui
from ollama_workbench.models.vision_comparison import vision_comparison_test
from ollama_workbench.chat.chat_interface import chat_interface
from ollama_workbench.models.update_models import update_models
from ollama_workbench.ui.file_management import files_tab
from ollama_workbench.server.server_configuration import server_configuration
from ollama_workbench.server.server_monitoring import server_monitoring
# Import the enhanced chat interface that preserves all original functionality
from ollama_workbench.chat.enhanced_chat_interface import enhanced_chat_interface
from ollama_workbench.ui.styles import apply_styles
from ollama_workbench.knowledge.simplified_rag import enhanced_rag_interface
from ollama_workbench.models.model_onboarding import onboarding_test_process
# Multimodal chat functionality is now integrated into the main Chat interface
from ollama_workbench.chat.multimodel_chat import multimodel_chat_app
# Import voice interface with error handling to prevent UI failures
try:
    from ollama_workbench.chat.voice_interface import voice_chat_interface, voice_settings_ui
    voice_interface_available = True
    print("CHECKPOINT: Voice interface successfully loaded")
except ImportError as e:
    print(f"CHECKPOINT: Voice interface not available: {str(e)}")
    voice_interface_available = False
    # Define fallback functions
    def voice_chat_interface():
        st.warning("Voice chat is not available. Please install the required dependencies: pip install SpeechRecognition pyaudio gtts pygame")
    def voice_settings_ui():
        st.warning("Voice settings are not available. Please install the required dependencies: pip install SpeechRecognition pyaudio gtts pygame")
# Import tool_playground with error handling to prevent UI failures
try:
    from ollama_workbench.ui.tool_playground import tool_playground
except Exception as e:
    st.error(f"Error loading Tool Playground module: {str(e)}")
    # Define a fallback function
    def tool_playground():
        st.error("Tool Playground is currently unavailable.")
        st.info("Try restarting the application or check the logs for errors.")
from ollama_workbench.ui.structured_output import structured_output_ui
from ollama_workbench.server.openai_compatibility import openai_compatibility_ui
from ollama_workbench.models.model_capabilities import model_capabilities_ui
from ollama_workbench.models.test_visualization import test_visualization_ui
from ollama_workbench.knowledge.repo_docs import main as repo_docs_main
from ollama_workbench.knowledge.web_to_corpus import main as web_to_corpus_main
from ollama_workbench.ui.welcome import display_welcome_message
from ollama_workbench.workflows.projects import projects_main, Task
from ollama_workbench.ui.prompts import manage_prompts, get_agent_prompt, get_metacognitive_prompt, get_voice_prompt, get_identity_prompt
from ollama_workbench.workflows.brainstorm import brainstorm_interface
from ollama_workbench.providers.ollama_utils import get_ollama_resource_usage
from ollama_workbench.workflows.research import research_interface
from ollama_workbench.knowledge.enhanced_corpus import enhance_corpus_ui
from ollama_workbench.workflows.build import build_interface
from ollama_workbench.providers.openai_utils import display_openai_settings, call_openai_api, set_openai_api_key
from ollama_workbench.providers.groq_utils import display_groq_settings, call_groq_api
from ollama_workbench.workflows.nodes import nodes_interface  
from ollama_workbench.providers.external_providers import external_providers_ui 
from streamlit_extras.stylable_container import stylable_container
from streamlit_javascript import st_javascript
from ollama_workbench.core.db_init import init_db
from ollama_workbench.chat.persona_chat import persona_group_chat
from persona_lab.persona_lab import persona_lab_interface
from ollama_workbench.models.model_management import model_management_dashboard
from ollama_workbench.server.performance_metrics import performance_metrics_interface, record_metrics

# Import enhanced observability
try:
    from observability import enhanced_observability_dashboard, configure_opik, observability_config
    ENHANCED_OBSERVABILITY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced observability not available: {e}")
    ENHANCED_OBSERVABILITY_AVAILABLE = False
    
    def enhanced_observability_dashboard():
        st.error("Enhanced observability not available. Install opik: pip install opik")
    
    def configure_opik(*args, **kwargs):
        pass

# Initialize the databases
init_db()

# Initialize model management database
try:
    from ollama_workbench.models.model_management import init_db as init_model_db
    init_model_db()
except ImportError:
    pass  # Model management module not available

# Global variable to store the port number
ollama_port = None 

# Create a native messaging host manifest (only if it doesn't already exist)
NATIVE_HOST_MANIFEST = "native_host_manifest.json"
if not os.path.exists(NATIVE_HOST_MANIFEST):
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
        ("Observability Dashboard", "Observability Dashboard"),
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

def create_sidebar():
    """Create and populate the sidebar."""
    with st.sidebar:
        st.markdown(
            '<div style="text-align: left;">'
            '<h1 class="logo" style="font-size: 24px; font-weight: 300;">🦙 Ollama <span style="color: orange;">Workbench</span></h1>'
            "</div>",
            unsafe_allow_html=True,
        )

        # Define the main navigation menu with modern styling
        main_menu = option_menu(
            menu_title="",
            options=["Chat", "Multi-Model Chat", "Voice Chat", "Tool Playground", "Structured Output", "Enhanced RAG", "Collaborative Workspace"] + list(SIDEBAR_SECTIONS.keys()),
            icons=["chat", "chat-square-text", "mic", "tools", "braces", "book", "pencil-square", "gear", "folder", "tools", "clipboard-check", "question-circle"],
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
    if 'bm_tasks' not in st.session_state:
        st.session_state.bm_tasks = []
    if st.session_state.selected_test == "Chat":
        # Use the enhanced chat interface that preserves all original functionality
        enhanced_chat_interface()
    # Multimodal chat functionality is now integrated into the main Chat interface
    elif st.session_state.selected_test == "Multi-Model Chat":
        multimodel_chat_app()
    elif st.session_state.selected_test == "Voice Chat":
        if voice_interface_available:
            tab1, tab2 = st.tabs(["Voice Chat", "Voice Settings"])
            with tab1:
                voice_chat_interface()
            with tab2:
                voice_settings_ui()
        else:
            st.warning("Voice chat features are not available. Please install the required dependencies:")
            st.code("pip install SpeechRecognition pyaudio gtts pygame", language="bash")
            st.info("Once installed, restart the application to use voice features.")
            st.info("The application is missing the 'speech_recognition' module which is required for voice functionality.")
            st.info("This does not affect other features of the Ollama Workbench.")
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
    elif st.session_state.selected_test == "Observability Dashboard":
        enhanced_observability_dashboard()
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
    elif st.session_state.selected_test == "Model Capabilities":
        model_capabilities_ui()
    elif st.session_state.selected_test == "Test Visualization":
        test_visualization_ui()
    elif st.session_state.selected_test == "Enhanced RAG":
        enhanced_rag_interface()
    elif st.session_state.selected_test == "Collaborative Workspace":
        def model_callback(prompt):
            # Get the currently selected model with dynamic fallback
            try:
                from ollama_workbench.providers.ollama_utils import get_dynamic_model_default
                dynamic_default = get_dynamic_model_default()
                model = st.session_state.get("selected_model", dynamic_default)
            except Exception:
                model = st.session_state.get("selected_model", None)
            
            # Get provider (default to Ollama)
            provider = st.session_state.get("selected_provider", "ollama")
            # Use the model to generate a response
            from ollama_workbench.providers.ollama_utils import get_ollama_client, call_ollama_endpoint
            try:
                response, _, _, _ = call_ollama_endpoint(
                    model=model,
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=4000
                )
                return response
            except Exception as e:
                return f"Error generating response: {str(e)}"
        collaborative_workspace_ui(model_callback=model_callback)
    elif st.session_state.selected_test == "Help":
        display_welcome_message()
    else:
        chat_interface()

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
    # Initialize session state
    initialize_session_state()
    
    # Initialize observability system
    if ENHANCED_OBSERVABILITY_AVAILABLE:
        try:
            # Configure Opik with default project name
            configure_opik(project_name="ollama-workbench")
            logger.info("Observability system initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize observability: {e}")
    
    # Create the sidebar
    create_sidebar()
    
    # Handle query parameters
    params = st.query_params
    web_page_content = params.get('web_page_content', None)
    if web_page_content:
        st.session_state.web_page_content = web_page_content

    # Display main content
    main_content()

    # Get parameters from URL
    web_page_url = params.get('url', None)
    is_extension = params.get('extension', 'false').lower() == 'true'

    if is_extension and web_page_url:
        st.session_state.web_page_url = web_page_url
        st.session_state.is_extension = is_extension

# Function to send a message to the Chrome extension
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