# main.py

import streamlit as st
import threading
from flask import Flask, jsonify, request
import os
import subprocess
import json 

# Set page config for wide layout
st.set_page_config(layout="wide", page_title="Ollama Workbench", page_icon="🦙")


import sys
print(f"Python executable: {sys.executable}")
try:
    from streamlit_option_menu import option_menu
except ImportError:
    # Fallback implementation for streamlit_option_menu
    def option_menu(menu_title, options, icons=None, menu_icon=None, default_index=0, styles=None):
        import streamlit as st
        selected = st.selectbox(
            menu_title if menu_title else "Menu",
            options,
            index=default_index,
        )
        return selected
from ollama_utils import *
from model_tests import *
from ui_elements import (
    model_comparison_test, contextual_response_test, feature_test,
    list_local_models, pull_models, show_model_details, remove_model_ui,
    vision_comparison_test, chat_interface, update_models, files_tab,
    server_configuration, server_monitoring
)
from repo_docs import main as repo_docs_main
from web_to_corpus import main as web_to_corpus_main
from welcome import display_welcome_message
from projects import projects_main, Task
from prompts import manage_prompts, get_agent_prompt, get_metacognitive_prompt, get_voice_prompt, get_identity_prompt
from brainstorm import brainstorm_interface
from ollama_utils import get_ollama_resource_usage
from research import research_interface
from enhanced_corpus import enhance_corpus_ui
from build import build_interface
from openai_utils import display_openai_settings, call_openai_api, set_openai_api_key
from groq_utils import display_groq_settings, call_groq_api
from nodes import nodes_interface  
from external_providers import external_providers_ui 
from streamlit_extras.stylable_container import stylable_container
from streamlit_javascript import st_javascript
from db_init import init_db
from persona_chat import persona_group_chat
from persona_lab.persona_lab import persona_lab_interface

# Initialize the database
init_db()

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

# Custom CSS
st.markdown("""
        <style>
        body, h1, h2, h3, h4, h5, h6, p {
            font-family: Open Sans, Helvetica, Arial, sans-serif!important;
            font-weight: 400;
        }
        .app-title {
            font-size: 40px!important;
            font-family: Open Sans, Helvetica, Arial, sans-serif!important;
        }
        .app-title span {
            color: orange;
        }
        .nav-link {
            display: block;
            color: inherit;
            border: 0!important;
            padding: 10px; 
            text-align: left!important;
            cursor: pointer;
            font-size: 16px;
            box-sizing: border-box;
            white-space: nowrap;
            text-decoration: none; 
        }
        .nav-link.active {
            background-color: orange!important; 
            color: white;
        }
        .nav-button {
            display: block;
            color: inherit;
            border: 0!important;
            padding: 0px;
            text-align: left!important;
            cursor: pointer;
            font-size: 16px;
            box-sizing: border-box;
            white-space: nowrap;
        }
        
        button {
            border: 0!important;
            text-align: left!important;
            justify-content: left!important;
            white-space: nowrap;
        }

        .st-emotion-cache-0, 
        .st-emotion-cache-0 details, 
        .st-emotion-cache-0 summary {
            border: 0!important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div > div[width="439"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: rgb(255,255,255,.2);
        }
        </style>
""", unsafe_allow_html=True)

# Function to inject custom CSS
def inject_custom_css(css):
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# JavaScript to detect theme
theme = st_javascript("""
    (function() {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        return prefersDark ? 'dark' : 'light';
    })();
""")

# Define CSS for light and dark modes
light_mode_css = """
/* Custom styles for light mode */
.sidebar .css-1d391kg {
    background-color: #fafafa;
}
.sidebar .css-1d391kg .nav-link {
    color: #000000;
}
.sidebar .css-1d391kg .nav-link-selected {
    background-color: #e0e0e0;
    color: #000000;
}
button.ef3psqc13, 
button.ef3psqc14 {
    background-color: #2c4755;
    color: #FFFFFF;
    border: solid 1px #FFF;
}
button.ef3psqc13:hover, 
button.ef3psqc14:hover {
    background-color: #e16d6d;
    color: #FFFFFF;
}
"""

dark_mode_css = """
/* Custom styles for dark mode */
.sidebar .css-1d391kg {
    background-color: #333333;
}
.sidebar .css-1d391kg .nav-link {
    color: #ffffff;
}
.sidebar .css-1d391kg .nav-link-selected {
    background-color: #555555;
    color: #ffffff;
}
button.ef3psqc13 {
    background-color: rgb(255,255,255,.1);
    border: 0px!important;
}
button.ef3psqc13:hover {
    background-color: rgb(255,255,255,.05);
}
"""

# Apply the appropriate CSS based on the detected theme
if theme == 'dark':
    inject_custom_css(dark_mode_css)
else:
    inject_custom_css(light_mode_css)

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
        ("Model Information", "Show Model Information"),
        ("Pull a Model", "Pull a Model"),
        ("Remove a Model", "Remove a Model"),
        ("Update Models", "Update Models"),
        ("Server Configuration", "Server Configuration"),
        ("Server Monitoring", "Server Monitoring"),
        ("External Providers", "External Providers")
    ],
    "Test": [
        ("Model Feature Test", "Feature Test"),
        ("Response Quality", "Model Comparison"),
        ("Contextual Response", "Contextual Response"),
        ("Vision Models", "Vision"),
    ],
    "Help": [
        ("Help", "Help")
    ]
}

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'selected_test' not in st.session_state:
        st.session_state.selected_test = "Chat"
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if 'bm_tasks' not in st.session_state:
        st.session_state.bm_tasks = []
    if 'show_resource_usage' not in st.session_state:
        st.session_state.show_resource_usage = False
    if 'web_page_content' not in st.session_state: 
        st.session_state.web_page_content = None
    if 'show_prompt' not in st.session_state:
        st.session_state.show_prompt = True

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

        # Define the main navigation menu
        main_menu = option_menu(
            menu_title="",
            options=["Chat"] + list(SIDEBAR_SECTIONS.keys()),
            icons=["chat", "gear", "folder", "tools", "clipboard-check", "question-circle"],
            menu_icon="cast",
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

        # Define the sub navigation menu based on the selected main menu
        if main_menu == "Chat":
            st.session_state.selected_test = "Chat"
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
        chat_interface()
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
    elif st.session_state.selected_test == "External Providers":
        external_providers_ui()
    elif st.session_state.selected_test == "Feature Test":
        feature_test()
    elif st.session_state.selected_test == "Model Comparison":
        model_comparison_test()
    elif st.session_state.selected_test == "Contextual Response":
        contextual_response_test()
    elif st.session_state.selected_test == "Vision":
        vision_comparison_test()
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
    initialize_session_state()
    create_sidebar()
    
    web_page_content = st.query_params.get('web_page_content', [None])[0] 
    if web_page_content:
        st.session_state.web_page_content = web_page_content

    main_content()

    # Get parameters from URL
    web_page_url = st.query_params.get('url', None)
    is_extension = st.query_params.get('extension', 'false').lower() == 'true'

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