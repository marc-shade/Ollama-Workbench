# research.py
import streamlit as st
import ollama
from typing import List, Dict
import json
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
import re
from agents import SearchManager, SearchAgent
from search_libraries import duckduckgo_search, google_search, serpapi_search, serper_search, bing_search
from ollama_utils import get_available_models
from ollama_utils import load_api_keys
from openai_utils import call_openai_api, OPENAI_MODELS
from groq_utils import call_groq_api, GROQ_MODELS
import sqlite3
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import pdfkit

# Create 'files' directory if it doesn't exist
files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')
os.makedirs(files_dir, exist_ok=True)

# Load API keys from settings file
def load_api_keys():
    if os.path.exists("api_keys.json"):
        with open("api_keys.json", "r") as f:
            return json.load(f)
    return {}

# Save API keys to settings file
def save_api_keys(api_keys):
    with open("api_keys.json", "w") as f:
        json.dump(api_keys, f, indent=4)

# Database functions
def init_db():
    conn = sqlite3.connect('research_reports.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reports
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT,
                  content TEXT,
                  date TEXT)''')
    conn.commit()
    conn.close()

def save_report(title: str, content: str):
    conn = sqlite3.connect('research_reports.db')
    c = conn.cursor()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO reports (title, content, date) VALUES (?, ?, ?)",
              (title, content, date))
    conn.commit()
    conn.close()

def get_all_reports():
    conn = sqlite3.connect('research_reports.db')
    c = conn.cursor()
    c.execute("SELECT id, title, date FROM reports")
    reports = c.fetchall()
    conn.close()
    return reports

def get_report_content(report_id: int):
    conn = sqlite3.connect('research_reports.db')
    c = conn.cursor()
    c.execute("SELECT content FROM reports WHERE id=?", (report_id,))
    content = c.fetchone()[0]
    conn.close()
    return content

def delete_report(report_id: int):
    conn = sqlite3.connect('research_reports.db')
    c = conn.cursor()
    c.execute("DELETE FROM reports WHERE id=?", (report_id,))
    conn.commit()
    conn.close()

# Export functions
def export_to_pdf(content: str, filename: str):
    pdf_path = os.path.join(files_dir, filename)
    pdf = SimpleDocTemplate(pdf_path, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    custom_style = ParagraphStyle(
        'CustomStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceBefore=6,
        spaceAfter=6
    )
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=16,
        leading=20,
        spaceBefore=12,
        spaceAfter=12
    )
    flowables = []

    # Split content into sections
    sections = content.split('\n\n')
    for section in sections:
        if section.startswith('Final Report:'):
            flowables.append(Paragraph("Final Report", title_style))
            flowables.append(Paragraph(section[14:], custom_style))
        elif section.startswith('References:'):
            flowables.append(Paragraph("References", title_style))
            references = section[12:].split('\n')
            for ref in references:
                flowables.append(Paragraph(ref, custom_style))
        elif section.startswith('Search Results:'):
            flowables.append(Paragraph("Search Results", title_style))
            search_results = section[16:].split('\n\n')
            for result in search_results:
                flowables.append(Paragraph(result, custom_style))
        else:
            flowables.append(Paragraph(section, custom_style))
        flowables.append(Spacer(1, 12))

    pdf.build(flowables)
    return pdf_path

def export_to_txt(content: str, filename: str):
    txt_path = os.path.join(files_dir, filename)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return txt_path

# Load research model settings from JSON file
def load_research_model_settings():
    if os.path.exists("research_models.json"):
        with open("research_models.json", "r") as f:
            return json.load(f)
    return {}

# Save research model settings to JSON file
def save_research_model_settings(settings):
    with open("research_models.json", "w") as f:
        json.dump(settings, f, indent=4)

def research_interface():
    st.title("🔬 Research")

    # Initialize database
    init_db()

    # Load API keys
    api_keys = load_api_keys()

    # Load research model settings
    research_model_settings = load_research_model_settings()

    # Sidebar settings
    with st.sidebar:
        # API Key Settings in a collapsed section
        with st.expander("🔑 API Key Settings", expanded=False):
            api_keys["serpapi_api_key"] = st.text_input("SerpApi API Key", value=api_keys.get("serpapi_api_key", ""), type="password")
            api_keys["serper_api_key"] = st.text_input("Serper API Key", value=api_keys.get("serper_api_key", ""), type="password")
            api_keys["google_api_key"] = st.text_input("Google Custom Search API Key", value=api_keys.get("google_api_key", ""), type="password")
            api_keys["google_cse_id"] = st.text_input("Google Custom Search Engine ID", value=api_keys.get("google_cse_id", ""), type="password")
            api_keys["bing_api_key"] = st.text_input("Bing Search API Key", value=api_keys.get("bing_api_key", ""), type="password")
            api_keys["openai_api_key"] = st.text_input("OpenAI API Key", value=api_keys.get("openai_api_key", ""), type="password")
            api_keys["groq_api_key"] = st.text_input("Groq API Key", value=api_keys.get("groq_api_key", ""), type="password")
            
            if st.button("💾 Save API Keys"):
                save_api_keys(api_keys)
                st.success("🟢 API keys saved!")

        # Model Settings in a collapsed section
        with st.expander("🤖 Model Settings", expanded=False):
            available_models = get_available_models() + OPENAI_MODELS + GROQ_MODELS

            # Load settings or defaults
            manager_model = research_model_settings.get("manager_model", available_models[0])
            manager_temperature = research_model_settings.get("manager_temperature", 0.7)
            manager_max_tokens = research_model_settings.get("manager_max_tokens", 4000)
            agent_model = research_model_settings.get("agent_model", available_models[0])
            agent_temperature = research_model_settings.get("agent_temperature", 0.7)
            agent_max_tokens = research_model_settings.get("agent_max_tokens", 4000)

            # Display model selection and settings
            manager_model = st.selectbox("Search Manager Model", available_models, index=available_models.index(manager_model))
            manager_temperature = st.slider("Search Manager Temperature", 0.0, 1.0, manager_temperature, step=0.1)
            manager_max_tokens = st.slider("Search Manager Max Tokens", 1000, 128000, manager_max_tokens, step=1000)
            agent_model = st.selectbox("Search Agent Model", available_models, index=available_models.index(agent_model))
            agent_temperature = st.slider("Search Agent Temperature", 0.0, 1.0, agent_temperature, step=0.1)
            agent_max_tokens = st.slider("Search Agent Max Tokens", 1000, 128000, agent_max_tokens, step=1000)

            if st.button("💾 Save Model Settings"):
                research_model_settings = {
                    "manager_model": manager_model,
                    "manager_temperature": manager_temperature,
                    "manager_max_tokens": manager_max_tokens,
                    "agent_model": agent_model,
                    "agent_temperature": agent_temperature,
                    "agent_max_tokens": agent_max_tokens
                }
                save_research_model_settings(research_model_settings)
                st.success("🟢 Model settings saved!")

    # User research request
    user_request = st.text_area("Enter your research request:")

    # Report length options with word count targets
    report_lengths = {
        "short": 500,
        "medium": 1000,
        "long": 2000
    }
    selected_length = st.selectbox("Report Length", list(report_lengths.keys()), format_func=lambda x: f"{x.capitalize()} (~{report_lengths[x]} words)")

    if st.button("🔬 Start Research"):
        if user_request:
            with st.spinner("Initializing Search Manager..."):
                search_manager = SearchManager(
                    name="Search Manager",
                    model=manager_model,
                    temperature=manager_temperature,
                    max_tokens=manager_max_tokens,
                    api_keys=api_keys
                )

            with st.spinner("Running Research..."):
                final_report = ""
                references = []
                agent_outputs = []
                for result_type, content in search_manager.run_research(user_request, selected_length, agent_model, report_lengths[selected_length]):
                    if result_type.endswith("Report") and result_type != "Final Report":
                        with st.expander(result_type, expanded=False):
                            st.write(content)
                        agent_outputs.append({"name": result_type.replace(" Report", ""), "content": content})
                    elif result_type == "Final Report":
                        st.subheader("Generated Report")
                        st.write(content)
                        final_report = content
                    elif result_type == "References":
                        st.subheader("All References")
                        for reference in content:
                            st.write(reference)
                        references = content

                # After the research is complete, save the report
                report_title = f"Research on: {user_request[:50]}..."  # Truncate long titles
                full_report = f"Final Report:\n\n{final_report}\n\nAll References:\n" + "\n".join(references)
                
                # Add agent outputs to the full report
                full_report += "\n\nSearch Results:\n\n"
                for agent_output in agent_outputs:
                    full_report += f"{agent_output['name']}:\n{agent_output['content']}\n\n"
                
                save_report(report_title, full_report)
                st.success("Research report saved successfully!")

        else:
            st.error("Please enter a research request.")

    # Add a section for viewing saved reports
    st.title("📗 Saved Reports")
    reports = get_all_reports()
    for report_id, title, date in reports:
        col1, col2, col3, col4, col5 = st.columns([12, 1, 1, 1, 1])
        with col1:
            st.write(f"🟩 {title} ({date})")
        with col2:
            if st.button("👀", key=f"view_{report_id}"):
                report_content = get_report_content(report_id)
                st.text_area("Report Content", report_content, height=300)
        with col3:
            if st.button("📕", key=f"pdf_{report_id}"):
                report_content = get_report_content(report_id)
                pdf_file = f"report_{report_id}.pdf"
                pdf_path = export_to_pdf(report_content, pdf_file)
                with open(pdf_path, "rb") as f:
                    st.download_button("PDF", f, file_name=pdf_file)
        with col4:
            if st.button("📄", key=f"txt_{report_id}"):
                report_content = get_report_content(report_id)
                txt_file = f"report_{report_id}.txt"
                txt_path = export_to_txt(report_content, txt_file)
                with open(txt_path, "rb") as f:
                    st.download_button("TXT", f, file_name=txt_file)
        with col5:
            if st.button("🗑️", key=f"delete_{report_id}"):
                delete_report(report_id)
                st.rerun()  # Refresh the page to show the updated list of reports

if __name__ == "__main__":
    research_interface()