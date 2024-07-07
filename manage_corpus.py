# manage_corpus.py
import streamlit as st
import os
import shutil
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

def manage_corpus():
    st.header("🗂 Manage Corpus")

    # Corpus folder
    corpus_folder = "corpus"
    if not os.path.exists(corpus_folder):
        os.makedirs(corpus_folder)

    # List existing corpus
    corpus_list = [f for f in os.listdir(corpus_folder) if os.path.isdir(os.path.join(corpus_folder, f))]
    st.subheader("⚫ Existing Corpus")
    if corpus_list:
        for corpus in corpus_list:
            col1, col2, col3 = st.columns([2, 1, 1])  # Add a column for renaming
            with col1:
                st.write(corpus)
            with col2:
                if st.button("✏️", key=f"rename_corpus_{corpus}"):
                    st.session_state.rename_corpus = corpus
                    st.experimental_rerun()
            with col3:
                if st.button("🗑️", key=f"delete_corpus_{corpus}"):
                    shutil.rmtree(os.path.join(corpus_folder, corpus))
                    st.success(f"Corpus '{corpus}' deleted.")
                    st.experimental_rerun()
    else:
        st.write("No existing corpus found.")

    # Handle renaming corpus
    if "rename_corpus" in st.session_state and st.session_state.rename_corpus:
        corpus_to_rename = st.session_state.rename_corpus
        new_corpus_name = st.text_input(f"Rename corpus '{corpus_to_rename}' to:", value=corpus_to_rename, key=f"rename_corpus_input_{corpus_to_rename}")
        if st.button("Confirm Rename", key=f"confirm_rename_{corpus_to_rename}"):
            if new_corpus_name:
                os.rename(os.path.join(corpus_folder, corpus_to_rename), os.path.join(corpus_folder, new_corpus_name))
                st.success(f"Corpus renamed to '{new_corpus_name}'")
                st.session_state.rename_corpus = None
                st.experimental_rerun()
            else:
                st.error("Please enter a new corpus name.")

    st.subheader("➕ Create New Corpus")
    # Create corpus from files
    st.write("**From Files:**")
    files_folder = "files"
    allowed_extensions = ['.json', '.txt']
    files = [f for f in os.listdir(files_folder) if os.path.isfile(os.path.join(files_folder, f)) and os.path.splitext(f)[1].lower() in allowed_extensions]
    selected_files = st.multiselect("Select files to create corpus:", files, key="create_corpus_files")
    corpus_name = st.text_input("Enter a name for the corpus:", key="create_corpus_name")
    if st.button("Create Corpus from Files", key="create_corpus_button"):
        if selected_files and corpus_name:
            create_corpus_from_files(corpus_folder, corpus_name, files_folder, selected_files)
            st.success(f"Corpus '{corpus_name}' created from selected files.")
            st.experimental_rerun()
        else:
            st.error("Please select files and enter a corpus name.")

def create_corpus_from_files(corpus_folder, corpus_name, files_folder, selected_files):
    corpus_path = os.path.join(corpus_folder, corpus_name)
    os.makedirs(corpus_path, exist_ok=True)
    
    # Combine all selected file content into one text
    all_text = ""
    for file in selected_files:
        file_path = os.path.join(files_folder, file)
        with open(file_path, "r", encoding='utf-8') as f:
            file_content = f.read()
        all_text += file_content + "\n\n"

    create_corpus_from_text(corpus_folder, corpus_name, all_text)

def create_corpus_from_text(corpus_folder, corpus_name, corpus_text):
    corpus_path = os.path.join(corpus_folder, corpus_name)
    os.makedirs(corpus_path, exist_ok=True)

    # Create Langchain documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(corpus_text)
    docs = [Document(page_content=t) for t in texts]

    # Create and load the vector database
    embeddings = OllamaEmbeddings()
    db = Chroma.from_documents(docs, embeddings, persist_directory=corpus_path)
    db.persist()