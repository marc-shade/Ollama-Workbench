import streamlit as st

# Set page config for wide layout
st.set_page_config(layout="wide", page_title="Ollama Workbench", page_icon="🦙")

from streamlit_option_menu import option_menu
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
from prompts import manage_prompts
from brainstorm import brainstorm_interface
from ollama_utils import get_ollama_resource_usage
from research import research_interface
from enhanced_corpus import enhance_corpus_ui
from build import build_interface
from streamlit_extras.stylable_container import stylable_container
from streamlit_javascript import st_javascript

# Custom CSS
st.markdown("""
        <style>
        body, h1, h2, h3, h4, h5, h6, p {
            font-family: Open Sans, Helvetica, Arial, sans-serif!important;
            font-weight: 400;
        }
        .app-title {
            font-size: 40px!important; /* Adjust font size as needed */
            font-family: Open Sans, Helvetica, Arial, sans-serif!important;
        }
        .app-title span {
            color: orange;
        }
        .nav-link {
            display: block;
            color: inherit;
            border: 0!important;
            padding: 10px; /* Add padding for better visual appeal */
            text-align: left!important;
            cursor: pointer;
            font-size: 16px;
            box-sizing: border-box;
            white-space: nowrap;
            text-decoration: none; /* Remove default link underline */
        }
        .nav-link.active {
            background-color: orange!important; /* Orange background for active link */
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
        .st-emotion-cache-1itdyc2 .stButton button {
            width: 100%;
        }
        div.row-widget.stButton > button {width:100%;}
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
        .main button {
            width: auto!important;
        }
        .help-button {
            position: fixed;
            bottom: 20px;
            left: 20px;
            background-color: #f0f0f0;
            border: none;
            cursor: pointer;
            font-size: 20px;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            text-align: center;
            line-height: 40px;
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
button.ef3psqc13 {
    background-color: rgb(0,0,0,.1);
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
    ],
    "Test": [
        ("Model Feature Test", "Feature Test"),
        ("Response Quality", "Model Comparison"),
        ("Contextual Response", "Contextual Response"),
        ("Vision Models", "Vision"),
    ],
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

def display_resource_usage_sidebar():
    """Displays resource usage in the sidebar."""
    usage = get_ollama_resource_usage()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"🚦: {usage['status']}")
    with col2:
        st.markdown(f"CPU: {usage['cpu_usage']}")
    with col3:
        st.markdown(f"RAM: {usage['memory_usage']}")
    with col4:
        st.markdown(f"GPU: {usage['gpu_usage']}")

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
            icons=["chat", "gear", "folder", "tools", "clipboard-check"],
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

        # Add help button
        if st.button("?", key="help_button", help="Click for help"):
            st.session_state.selected_test = "Help"

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

def main():
    initialize_session_state()
    create_sidebar()
    main_content()

if __name__ == "__main__":
    main()
