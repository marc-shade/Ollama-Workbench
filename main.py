# main.py
import streamlit as st
import sys
import os
from streamlit_option_menu import option_menu
from streamlit_extras.buy_me_a_coffee import button
import json
from welcome import show_welcome
import sqlite3

# Ensure the current working directory is the root of the project
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'leads'))
sys.path.append(os.path.join(current_dir, 'onboarding'))
sys.path.append(os.path.join(current_dir, 'domains'))
sys.path.append(os.path.join(current_dir, 'task_management'))
sys.path.append(os.path.join(current_dir, 'prompts'))

# Set page config before any other Streamlit commands
st.set_page_config(page_title="TeamWork", page_icon="🐴", layout="wide")

# Import the necessary functions from each script
from lead_generator import run_lead_generator
from onboarding_workflow import run_onboarding_workflow
from check_domain import run_domain_checker
from domain_search import run_domain_search
from task_management import run_task_management
from prompts.weekly_prompt import run_weekly_prompt
from prompts.agent_builder import run_agent_builder
from website_critique import run_website_critique

custom_css = """
<style>
body, h1, h2, h3, h4, h5, h6, p {
    font-family: Helvetica, sans-serif!important;
    font-size: 18px;
}
</style>
"""

def check_secret_key(file_path, expected_key):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data.get('secret_key') == expected_key
    return False

def get_file_size(file_path):
    """Get the size of a file in a human-readable format."""
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
    return "File not found"

def get_db_size(db_path):
    """Get the size of a SQLite database file."""
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        size_bytes = cursor.fetchone()[0]
        conn.close()
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
    return "Database not found"

def reset_caches_and_databases():
    st.title("🔄 Reset Caches and Databases")
    st.write("Select the caches and databases you want to reset:")

    col1, col2 = st.columns(2)

    with col1:
        # Lead Generator Cache
        config_size = get_file_size('config.json')
        reset_lead_generator = st.checkbox(f"Lead Generator Cache ({config_size})")

        # Website Critique Database
        website_critique_size = get_db_size('website_critique.db')
        reset_website_critique = st.checkbox(f"Website Critique Database ({website_critique_size})")

        # Weekly Prompt Database
        weekly_prompt_size = get_db_size('prompts.db')
        reset_weekly_prompt = st.checkbox(f"Weekly Prompt Database ({weekly_prompt_size})")

        # Agent Builder Database
        agent_builder_size = get_db_size('agent_prompts.db')
        reset_agent_builder = st.checkbox(f"Agent Builder Database ({agent_builder_size})")

    with col2:
        # Onboarding Workflow Cache
        onboarding_size = get_file_size('onboarding/config.json')
        reset_onboarding = st.checkbox(f"Onboarding Workflow Cache ({onboarding_size})")

        # Domain Search Cache
        domain_search_size = get_file_size('domains/config.json')
        reset_domain_search = st.checkbox(f"Domain Search Cache ({domain_search_size})")

        # Task Management Database (in-memory, so no size)
        reset_task_management = st.checkbox("Task Management Database (In-memory)")

        # Streamlit Cache (size not directly accessible)
        reset_streamlit_cache = st.checkbox("Streamlit Cache (Size not available)")

    if st.button("Reset Selected"):
        if reset_lead_generator:
            if os.path.exists('config.json'):
                os.remove('config.json')
                st.success("Lead Generator Cache reset.")

        if reset_website_critique:
            conn = sqlite3.connect('website_critique.db')
            c = conn.cursor()
            c.execute("DROP TABLE IF EXISTS critiques")
            conn.commit()
            conn.close()
            st.success("Website Critique Database reset.")

        if reset_weekly_prompt:
            conn = sqlite3.connect('prompts.db')
            c = conn.cursor()
            c.execute("DROP TABLE IF EXISTS prompts")
            conn.commit()
            conn.close()
            st.success("Weekly Prompt Database reset.")

        if reset_agent_builder:
            conn = sqlite3.connect('agent_prompts.db')
            c = conn.cursor()
            c.execute("DROP TABLE IF EXISTS agent_prompts")
            conn.commit()
            conn.close()
            st.success("Agent Builder Database reset.")

        if reset_onboarding:
            if os.path.exists('onboarding/config.json'):
                os.remove('onboarding/config.json')
            st.success("Onboarding Workflow Cache reset.")

        if reset_domain_search:
            if os.path.exists('domains/config.json'):
                os.remove('domains/config.json')
            st.success("Domain Search Cache reset.")

        if reset_task_management:
            if 'tasks' in st.session_state:
                del st.session_state['tasks']
            st.success("Task Management Database reset.")

        if reset_streamlit_cache:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Streamlit Cache cleared.")

        st.success("Selected caches and databases have been reset. Please refresh the page.")

def main():
    # Initialize the page state
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = 'Welcome'

    # Sidebar navigation
    with st.sidebar:
        # Style the app title
        st.markdown(
            """
            <style>
            body, h1, h2, h3, h4, h5, h6, p {
            font-family: Open Sans, Helvetica, Arial, sans-serif!important;
            }
            .app-title {
                font-size: 44px!important; /* Adjust font size as needed */
                font-family: Open Sans, Helvetica, Arial, sans-serif!important;
            }
            .app-title span {
                color: orange;
            }
            </style>
            <h1 class="app-title">🐴 Team<span>Work</span></h1>
            """,
            unsafe_allow_html=True
        )

        # Create a more visually appealing navigation menu
        selected = option_menu(
            menu_title=None,
            options=["Welcome", "Leads", "Website Critique", "Prompts", "Onboarding", "Domains", "Task Management", "Reset Caches"],
            icons=["house", "bullseye", "lightbulb", "star", "rocket-takeoff", "globe", "kanban", "arrow-clockwise"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {"font-size": "16px", "color": "#999", "text-align": "left", "margin": "0px", "--hover-color": "#333"},
                "nav-link-selected": {"background-color": "#333"},
                "nav-link": {
                    "font-size": "16px", 
                    "color": "#999", 
                    "text-align": "left", 
                    "margin": "0px", 
                    "--hover-color": "#333",
                    "font-family": "Open Sans, Helvetica, Arial, sans-serif"
                },
            }
        )

        # Update session state with selected page
        st.session_state.selected_page = selected

        # Check if the secret key JSON file exists and has the correct key
        secret_key_file = 'secret_key_off.json'
        secret_key_value = 'I_am_an_honest_person'
        if not check_secret_key(secret_key_file, secret_key_value):
            # Add Buy Me a Coffee button and image in a 2-column layout
            st.markdown("---")  # Add a separator

            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(
                    '<a href="https://github.com/marc-shade" target="_blank"><img src="https://2acrestudios.com/wp-content/uploads/2024/06/marc-cyberpunk.png" '
                    'style="border-radius: 50%; max-width: 70px; object-fit: cover;" /></a>',
                    unsafe_allow_html=True,
                )
            with col2:
                button(
                    username=os.getenv("BUYMEACOFFEE_USERNAME", "marcshade"),
                    floating=False,
                    text="Support Marc",
                    emoji="☕",
                    bg_color="#FF5F5F",
                    font_color="#FFFFFF",
                )
            st.markdown(
                '<span style="font-size:17px; font-weight:normal; font-family:Courier;">Find this tool useful? Your support means a lot! Give a donation of $10 or more to remove this notice.</span>',
                unsafe_allow_html=True,
            )

    # Main content area
    selected_page = st.session_state.selected_page
    if selected_page == "Welcome":
        show_welcome()
    elif selected_page == "Leads":
        run_lead_generator()
    elif selected_page == "Website Critique":
        run_website_critique()
    elif selected_page == "Prompts":
        submenu = option_menu(
            menu_title="Prompts",
            options=["Agent Prompt Builder", "Weekly Prompts"],
            icons=["robot", "calendar"],
            menu_icon="star",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#333"},
                "icon": {"color": "orange", "font-size": "20px"},
                "nav-link": {"font-size": "14px", "color": "#FFF", "text-align": "center", "margin": "0px", "--hover-color": "#444"},
                "nav-link-selected": {"background-color": "#444"},
            }
        )

        if submenu == "Weekly Prompts":
            st.title("🗓️  Weekly Prompts")
            run_weekly_prompt()
        elif submenu == "Agent Prompt Builder":
            st.title("🤖 Agent Prompt Builder")
            run_agent_builder()
    elif selected_page == "Onboarding":
        st.title("🚀 Onboarding Workflow")
        run_onboarding_workflow()
    elif selected_page == "Domains":
        submenu = option_menu(
            menu_title="Domains",
            options=["AI-Powered Domain Search", "Domain Name Availability Checker"],
            icons=["robot", "search"],
            menu_icon="globe",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#333"},
                "icon": {"color": "orange", "font-size": "20px"},
                "nav-link": {"font-size": "14px", "color": "#FFF", "text-align": "center", "margin": "0px", "--hover-color": "#444"},
                "nav-link-selected": {"background-color": "#444"},
            }
        )

        if submenu == "AI-Powered Domain Search":
            run_domain_search()
        elif submenu == "Domain Name Availability Checker":
            run_domain_checker()
    elif selected_page == "Task Management":
        st.title("🗂️ Task Management")
        run_task_management()
    elif selected_page == "Reset Caches":
        reset_caches_and_databases()

if __name__ == "__main__":
    main()