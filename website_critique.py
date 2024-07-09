import streamlit as st
import pandas as pd
import sqlite3
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import ollama
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

if 'critique_model' not in st.session_state:
    st.session_state.critique_model = 'mistral:instruct'

if 'email_model' not in st.session_state:
    st.session_state.email_model = 'mistral:instruct'

if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7

@st.cache_data
def get_available_models():
    try:
        response = ollama.list()
        models = [model['name'] for model in response['models']]
        return models
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        st.error(f"Error fetching models: {e}")
        return ['mistral:instruct']  # Default to mistral if unable to fetch models

# Initialize database
def init_db():
    try:
        conn = sqlite3.connect('website_critique.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS critiques
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      Name TEXT,
                      Address TEXT,
                      Types TEXT,
                      Website TEXT,
                      Email TEXT,
                      "Phone Number" TEXT,
                      Rating REAL,
                      "Business Type" TEXT,
                      Keyword TEXT,
                      critique TEXT,
                      email_content TEXT,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        st.error(f"Error initializing database: {e}")

# Function to save critique to database
def save_critique(name, address, types, website, email, phone_number, rating, business_type, keyword, critique, email_content):
    try:
        conn = sqlite3.connect('website_critique.db')
        c = conn.cursor()
        c.execute('''INSERT INTO critiques (Name, Address, Types, Website, Email, "Phone Number", Rating, "Business Type", Keyword, critique, email_content)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (name, address, types, website, email, phone_number, rating, business_type, keyword, critique, email_content))
        conn.commit()
        conn.close()
        logger.info(f"Critique saved for {name}")
    except Exception as e:
        logger.error(f"Error saving critique: {e}")
        st.error(f"Error saving critique: {e}")

# Function to load critiques from database
def load_critiques():
    try:
        conn = sqlite3.connect('website_critique.db')
        df = pd.read_sql_query("SELECT * FROM critiques", conn)
        conn.close()
        logger.info(f"Loaded {len(df)} critiques from database")
        return df
    except Exception as e:
        logger.error(f"Error loading critiques: {e}")
        st.error(f"Error loading critiques: {e}")
        return pd.DataFrame()

# Function to update critique in database
def update_critique(id, name, address, types, website, email, phone_number, rating, business_type, keyword, critique, email_content):
    try:
        conn = sqlite3.connect('website_critique.db')
        c = conn.cursor()
        c.execute('''UPDATE critiques
                     SET Name=?, Address=?, Types=?, Website=?, Email=?, "Phone Number"=?, Rating=?, "Business Type"=?, Keyword=?, critique=?, email_content=?
                     WHERE id=?''',
                  (name, address, types, website, email, phone_number, rating, business_type, keyword, critique, email_content, id))
        conn.commit()
        conn.close()
        logger.info(f"Updated critique for {name}")
    except Exception as e:
        logger.error(f"Error updating critique: {e}")
        st.error(f"Error updating critique: {e}")

# Function to delete critique from database
def delete_critique(id):
    try:
        conn = sqlite3.connect('website_critique.db')
        c = conn.cursor()
        c.execute("DELETE FROM critiques WHERE id=?", (id,))
        conn.commit()
        conn.close()
        logger.info(f"Deleted critique with id {id}")
    except Exception as e:
        logger.error(f"Error deleting critique: {e}")
        st.error(f"Error deleting critique: {e}")

def reset_database():
    try:
        conn = sqlite3.connect('website_critique.db')
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS critiques")
        conn.commit()
        conn.close()
        logger.info("Database reset successfully")
        init_db()
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        st.error(f"Error resetting database: {e}")

# Function to scrape website content
def scrape_website(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract relevant information
        title = soup.title.string if soup.title else ""
        meta_description = soup.find("meta", attrs={"name": "description"})
        description = meta_description["content"] if meta_description else ""
        headings = [h.text.strip() for h in soup.find_all(['h1', 'h2', 'h3']) if h.text.strip()]
        paragraphs = [p.text.strip() for p in soup.find_all('p') if p.text.strip()]
        links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
        
        logger.info(f"Successfully scraped {url}")
        return {
            "title": title,
            "description": description,
            "headings": headings[:5],  # Limit to first 5 headings
            "paragraphs": paragraphs[:3],  # Limit to first 3 paragraphs
            "links": links[:10]  # Limit to first 10 links
        }
    except requests.RequestException as e:
        logger.error(f"Error scraping {url}: {e}")
        st.error(f"Error scraping {url}: {e}")
        return None

def generate_critique(website_data, model, temperature):
    try:
        prompt = f"""
        Analyze the following website data and provide a detailed critique:

        Title: {website_data['title']}
        Description: {website_data['description']}
        Headings: {website_data['headings'][:5]}
        Sample Paragraphs: {website_data['paragraphs'][:3]}
        Number of Links: {len(website_data['links'])}

        Provide a critique focusing on:
        1. Visual Appeal
        2. Usability and Navigation
        3. Content Quality
        4. Call-to-Actions (CTAs)
        5. Mobile Responsiveness
        6. Load Time and Performance

        Also, provide one free, actionable marketing tip.
        """

        response = ollama.generate(model=model, prompt=prompt, options={"temperature": temperature})
        logger.info("Generated critique successfully")
        return response['response']
    except Exception as e:
        logger.error(f"Error generating critique: {e}")
        st.error(f"Error generating critique: {e}")
        return None

def generate_email(critique, name, model, temperature):
    try:
        prompt = f"""
        Use the following website critique to craft a compelling, short cold-call email:

        Critique: {critique}

        The email should be addressed to someone from {name}.
        Focus on the single most important and interesting thing from the critique.
        Be concise, personalized, and highly engaging.
        Aim to sell websites, marketing services, consulting, coaching, or AI solutions.
        Get directly to the point, give them an out, and be gracious.
        Do not repeat content from the critique verbatim.

        Provide a novel, creative, attention-grabbing subject line and the email body.
        """

        response = ollama.generate(model=model, prompt=prompt, options={"temperature": temperature})
        logger.info(f"Generated email for {name}")
        return response['response']
    except Exception as e:
        logger.error(f"Error generating email: {e}")
        st.error(f"Error generating email: {e}")
        return None

def run_website_critique():
    st.title("💡 Website Critique")

    # Initialize session state variables
    if 'critique_model' not in st.session_state:
        st.session_state.critique_model = 'mistral:instruct'
    if 'email_model' not in st.session_state:
        st.session_state.email_model = 'mistral:instruct'
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()

    init_db()

    # Sidebar for model selection and settings
    st.sidebar.header("Model Settings")
    available_models = get_available_models()
    st.session_state.critique_model = st.sidebar.selectbox(
        "Select Critique Model", 
        available_models, 
        index=available_models.index(st.session_state.critique_model) if st.session_state.critique_model in available_models else 0 
    )
    st.session_state.email_model = st.sidebar.selectbox(
        "Select Email Model", 
        available_models, 
        index=available_models.index(st.session_state.email_model) if st.session_state.email_model in available_models else 0
    )
    st.session_state.temperature = st.sidebar.slider("Temperature", 0.1, 1.0, st.session_state.temperature)

    # File upload for CSV
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = pd.concat([st.session_state.df, df], ignore_index=True)
        st.success(f"CSV file uploaded successfully. {len(df)} entries added.")

    # Manual input
    st.subheader("Add Manual Entry")
    with st.form("manual_entry"):
        name = st.text_input("Name")
        address = st.text_input("Address")
        types = st.text_input("Types")
        website = st.text_input("Website")
        email = st.text_input("Email")
        phone_number = st.text_input("Phone Number")
        rating = st.number_input("Rating", min_value=0.0, max_value=5.0, step=0.1)
        business_type = st.text_input("Business Type")
        keyword = st.text_input("Keyword")

        if st.form_submit_button("Add to List"):
            new_entry = pd.DataFrame({
                "Name": [name],
                "Address": [address],
                "Types": [types],
                "Website": [website],
                "Email": [email],
                "Phone Number": [phone_number],
                "Rating": [rating],
                "Business Type": [business_type],
                "Keyword": [keyword]
            })
            st.session_state.df = pd.concat([st.session_state.df, new_entry], ignore_index=True)
            st.success("Entry added successfully.")
            st.experimental_rerun()

    # Display current entries
    st.subheader("Current Entries")
    if not st.session_state.df.empty:
        st.dataframe(st.session_state.df)

        st.subheader("Select Entries to Process")
        selected_indices = st.multiselect("Select rows to process:", options=st.session_state.df.index.tolist(), default=st.session_state.df.index.tolist())
        selected_df = st.session_state.df.loc[selected_indices]

        if st.button("Generate Critiques and Emails"):
            if not selected_df.empty:
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_rows = len(selected_df)

                for index, row in selected_df.iterrows():
                    status_text.text(f"Processing {row['Name']}...")
                    website_data = scrape_website(row["Website"])

                    if website_data:
                        critique = generate_critique(website_data, st.session_state.critique_model, st.session_state.temperature)
                        if critique:
                            email_content = generate_email(critique, row["Name"], st.session_state.email_model, st.session_state.temperature)
                            if email_content:
                                save_critique(row["Name"], row["Address"], row["Types"], row["Website"], row["Email"], row["Phone Number"], row["Rating"], row["Business Type"], row["Keyword"], critique, email_content)
                            else:
                                st.warning(f"Failed to generate email for {row['Name']}")
                        else:
                            st.warning(f"Failed to generate critique for {row['Name']}")
                    else:
                        st.warning(f"Failed to scrape website for {row['Name']}")

                    # Update progress
                    progress = min((index + 1) / total_rows, 1.0)
                    progress_bar.progress(progress)

                status_text.text("Processing complete!")
                st.success("Critiques and emails generated successfully!")

                # Update the DataFrame in the session state
                st.session_state.df = st.session_state.df.drop(selected_indices)
                
                # Offer to download the updated CSV
                csv = st.session_state.df.to_csv(index=False)
                st.download_button(
                    label="Download updated CSV",
                    data=csv,
                    file_name="updated_websites.csv",
                    mime="text/csv",
                )
                st.experimental_rerun()
            else:
                st.warning("No rows selected for processing.")
    else:
        st.info("No entries available. Please upload a CSV or add a manual entry.")

    # Display and edit saved critiques
    st.subheader("Saved Critiques and Emails")
    critiques_df = load_critiques()

    if not critiques_df.empty:
        # Add a 'Select' column for row selection
        critiques_df['Select'] = False  
        edited_df = st.data_editor(critiques_df, num_rows="dynamic")

        if st.button("Save Changes"):
            for index, row in edited_df.iterrows():
                update_critique(row['id'], row['Name'], row['Address'], row['Types'], row['Website'], row['Email'], row['Phone Number'], row['Rating'], row['Business Type'], row['Keyword'], row['critique'], row['email_content'])
            st.success("Changes saved successfully!")

        if st.button("Delete Selected"):
            selected_rows = edited_df[edited_df['Select'] == True]
            for index, row in selected_rows.iterrows():
                delete_critique(row['id'])
            st.success("Selected critiques deleted successfully!")
            st.experimental_rerun()
    else:
        st.info("No saved critiques found.")

if __name__ == "__main__":
    reset_database()
    run_website_critique()