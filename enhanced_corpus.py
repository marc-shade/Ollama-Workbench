# enhanced_corpus.py

import streamlit as st
import os
import logging
import shutil
import json
import ollama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
RAGTEST_DIR = "ragtest"
FILES_DIR = "files"

# Ensure necessary directories exist
os.makedirs(RAGTEST_DIR, exist_ok=True)
os.makedirs(FILES_DIR, exist_ok=True)

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

    def load_text_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def load_pdf_file(self, file_path):
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text

    def process_file(self, file_path):
        if file_path.lower().endswith('.pdf'):
            text = self.load_pdf_file(file_path)
        else:
            text = self.load_text_file(file_path)
        return self.split_text(text)

    def process_url(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        return self.split_text(text)

    def process_text(self, text):
        return self.split_text(text)

class OllamaEmbedder:
    def __init__(self, model="llama2"):
        self.model = model

    def get_embedding(self, text):
        try:
            response = ollama.embeddings(model=self.model, prompt=text)
            return response['embedding']
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise

class GraphRAGCorpus:
    def __init__(self, corpus_name, embedder):
        self.corpus_name = corpus_name
        self.corpus_dir = os.path.join(RAGTEST_DIR, corpus_name)
        self.embedder = embedder
        self.documents = []
        self.embeddings = []

    def add_document(self, content, metadata=None):
        embedding = self.embedder.get_embedding(content)
        doc_id = len(self.documents)
        self.documents.append({
            "id": doc_id,
            "content": content,
            "metadata": metadata or {}
        })
        self.embeddings.append(embedding)

    def save(self):
        os.makedirs(self.corpus_dir, exist_ok=True)
        
        # Save documents
        for doc in self.documents:
            with open(os.path.join(self.corpus_dir, f"doc_{doc['id']}.txt"), "w", encoding='utf-8') as f:
                f.write(doc['content'])

        # Save metadata and embeddings
        metadata = {
            "documents": [{"id": doc["id"], "metadata": doc["metadata"]} for doc in self.documents]
        }
        with open(os.path.join(self.corpus_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        with open(os.path.join(self.corpus_dir, "embeddings.json"), "w") as f:
            json.dump(self.embeddings, f)

    @classmethod
    def load(cls, corpus_name, embedder):
        corpus = cls(corpus_name, embedder)
        corpus_dir = os.path.join(RAGTEST_DIR, corpus_name)

        # Load metadata
        with open(os.path.join(corpus_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        # Load embeddings
        with open(os.path.join(corpus_dir, "embeddings.json"), "r") as f:
            corpus.embeddings = json.load(f)

        # Load documents
        for doc_meta in metadata["documents"]:
            with open(os.path.join(corpus_dir, f"doc_{doc_meta['id']}.txt"), "r", encoding='utf-8') as f:
                content = f.read()
            corpus.documents.append({
                "id": doc_meta["id"],
                "content": content,
                "metadata": doc_meta["metadata"]
            })

        return corpus

    def query(self, query_text, n_results=3):
        query_embedding = self.embedder.get_embedding(query_text)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-n_results:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "content": self.documents[idx]["content"],
                "metadata": self.documents[idx]["metadata"],
                "similarity": similarities[idx]
            })
        
        return results

def create_graphrag_corpus(corpus_name, documents):
    embedder = OllamaEmbedder()
    corpus = GraphRAGCorpus(corpus_name, embedder)

    for i, doc in enumerate(documents):
        corpus.add_document(doc, metadata={"original_id": i})

    corpus.save()
    logger.info(f"Corpus '{corpus_name}' created successfully!")

def process_and_save_corpus(data, corpus_name, is_url=False, is_text=False):
    doc_processor = DocumentProcessor()
    if is_url:
        documents = doc_processor.process_url(data)
    elif is_text:
        documents = doc_processor.process_text(data)
    else:
        file_path = os.path.join(FILES_DIR, data.name)
        with open(file_path, "wb") as f:
            f.write(data.getbuffer())
        documents = doc_processor.process_file(file_path)

    try:
        create_graphrag_corpus(corpus_name, documents)
        st.success(f"Corpus '{corpus_name}' created successfully!")
    except Exception as e:
        logger.error(f"Error creating GraphRAG corpus: {str(e)}")
        st.error(f"Error creating GraphRAG corpus: {str(e)}")

def enhance_corpus_ui():
    st.title("GraphRAG Corpus Management")

    st.subheader("✚ Create New Corpus")
    tab1, tab2, tab3 = st.tabs(["From File", "From URL", "From Text"])

    with tab1:
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])
        corpus_name = st.text_input("Enter a name for the corpus:", key="create_corpus_name_file")
        if st.button("✚ Create Corpus", key="create_corpus_button_file"):
            if uploaded_file is not None and corpus_name:
                process_and_save_corpus(uploaded_file, corpus_name)
            else:
                st.error("Please upload a file and enter a corpus name.")

    with tab2:
        url = st.text_input("Enter a URL to process")
        corpus_name = st.text_input("Enter a name for the corpus:", key="create_corpus_name_url")
        if st.button("✚ Create Corpus", key="create_corpus_button_url"):
            if url and corpus_name:
                process_and_save_corpus(url, corpus_name, is_url=True)
            else:
                st.error("Please enter a URL and a corpus name.")

    with tab3:
        text_input = st.text_area("Enter text to process")
        corpus_name = st.text_input("Enter a name for the corpus:", key="create_corpus_name_text")
        if st.button("✚ Create Corpus", key="create_corpus_button_text"):
            if text_input and corpus_name:
                process_and_save_corpus(text_input, corpus_name, is_text=True)
            else:
                st.error("Please enter text and a corpus name.")

    st.subheader("📚 Existing Corpora")
    corpora = [d for d in os.listdir(RAGTEST_DIR) if os.path.isdir(os.path.join(RAGTEST_DIR, d))]
    if corpora:
        for corpus in corpora:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"- {corpus}")
            with col2:
                if st.button("🔍 Query", key=f"query_{corpus}"):
                    st.session_state[f"show_query_{corpus}"] = True
            with col3:
                if st.button("🗑️", key=f"delete_{corpus}"):
                    shutil.rmtree(os.path.join(RAGTEST_DIR, corpus))
                    st.success(f"Corpus '{corpus}' deleted.")
                    st.rerun()
            
            if st.session_state.get(f"show_query_{corpus}", False):
                query = st.text_input(f"Enter query for {corpus}:")
                if query:
                    embedder = OllamaEmbedder()
                    loaded_corpus = GraphRAGCorpus.load(corpus, embedder)
                    results = loaded_corpus.query(query)
                    st.write("Query Results:")
                    for result in results:
                        st.write(f"Similarity: {result['similarity']:.4f}")
                        st.write(result['content'])
                        st.write("---")
    else:
        st.write("No existing corpora found.")

if __name__ == "__main__":
    enhance_corpus_ui()