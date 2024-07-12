# research.py
import streamlit as st
import ollama
from typing import List, Dict
import json
from langchain_community.embeddings import OllamaEmbeddings  # Updated import
from langchain_community.vectorstores import Chroma  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
import re
from agents import SearchManager, SearchAgent
from search_libraries import duckduckgo_search, google_search, serpapi_search, serper_search, bing_search
from ollama_utils import get_available_models
import sqlite3
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import pdfkit

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

# Export functions
def export_to_pdf(content: str, filename: str):
    pdf = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    flowables = []

    for line in content.split('\n'):
        p = Paragraph(line, styles['Normal'])
        flowables.append(p)
        flowables.append(Spacer(1, 12))

    pdf.build(flowables)

def export_to_txt(content: str, filename: str):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

def research_interface():
    st.title("🔬 Research")

    # Initialize database
    init_db()

    # Load API keys
    api_keys = load_api_keys()

    # API Key Settings
    st.sidebar.subheader("API Key Settings")
    api_keys["serpapi_api_key"] = st.sidebar.text_input("SerpApi API Key", value=api_keys.get("serpapi_api_key", ""))
    api_keys["serper_api_key"] = st.sidebar.text_input("Serper API Key", value=api_keys.get("serper_api_key", ""))
    api_keys["google_api_key"] = st.sidebar.text_input("Google Custom Search API Key", value=api_keys.get("google_api_key", ""))
    api_keys["google_cse_id"] = st.sidebar.text_input("Google Custom Search Engine ID", value=api_keys.get("google_cse_id", ""))
    api_keys["bing_api_key"] = st.sidebar.text_input("Bing Search API Key", value=api_keys.get("bing_api_key", ""))
    
    if st.sidebar.button("Save API Keys"):
        save_api_keys(api_keys)
        st.sidebar.success("API keys saved!")

    # User research request
    user_request = st.text_area("Enter your research request:")

    # Report length options
    report_lengths = ["short", "medium", "long"]
    selected_length = st.selectbox("Report Length", report_lengths)

    # Model selection for Search Manager and Agents
    available_models = get_available_models()
    col1, col2 = st.columns(2)
    with col1:
        manager_model = st.selectbox("Select Search Manager Model", available_models)
    with col2:
        agent_model = st.selectbox("Select Search Agent Model", available_models)

    if st.button("Start Research"):
        if user_request:
            with st.spinner("Initializing Search Manager..."):
                search_manager = SearchManager(
                    name="Search Manager",
                    model=manager_model,  # Use the selected manager model
                    temperature=0.7,
                    max_tokens=4000,
                    api_keys=api_keys
                )

            with st.spinner("Running Research..."):
                final_report = ""
                references = []
                agent_outputs = []
                for result_type, content in search_manager.run_research(user_request, selected_length, agent_model):  # Pass agent_model here
                    if result_type.startswith("Agent"):
                        with st.expander(result_type, expanded=False):  # Set expanded to False to collapse by default
                            st.write(content)
                        agent_outputs.append({"agent": result_type, "content": content})
                    elif result_type == "Final Report":
                        st.subheader("Generated Report")
                        st.write(content)
                        final_report = content
                    elif result_type == "References":
                        st.subheader("References")
                        for reference in content:
                            st.write(reference)
                        references = content

                # After the research is complete, save the report
                report_title = f"Research on: {user_request[:50]}..."  # Truncate long titles
                full_report = f"Final Report:\n\n{final_report}\n\nReferences:\n" + "\n".join(references)
                save_report(report_title, full_report)
                st.success("Research report saved successfully!")

        else:
            st.error("Please enter a research request.")

    # Add a section for viewing saved reports
    st.subheader("Saved Reports")
    reports = get_all_reports()
    for report_id, title, date in reports:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.write(f"{title} ({date})")
        with col2:
            if st.button("View", key=f"view_{report_id}"):
                report_content = get_report_content(report_id)
                st.text_area("Report Content", report_content, height=300)
        with col3:
            if st.button("Export PDF", key=f"pdf_{report_id}"):
                report_content = get_report_content(report_id)
                pdf_file = f"report_{report_id}.pdf"
                export_to_pdf(report_content, pdf_file)
                with open(pdf_file, "rb") as f:
                    st.download_button("Download PDF", f, file_name=pdf_file)
        with col4:
            if st.button("Export TXT", key=f"txt_{report_id}"):
                report_content = get_report_content(report_id)
                txt_file = f"report_{report_id}.txt"
                export_to_txt(report_content, txt_file)
                with open(txt_file, "rb") as f:
                    st.download_button("Download TXT", f, file_name=txt_file)

if __name__ == "__main__":
    research_interface()