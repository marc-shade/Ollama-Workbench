import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader # Update imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import  OllamaEmbeddings # Update imports
from langchain_community.vectorstores import Chroma # Update imports
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama # Update imports
from langchain.prompts import PromptTemplate
from ollama_utils import get_available_models # Import the function
import os
import shutil

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def process_file(self, file_path):
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        return self.text_splitter.split_documents(loader.load())

    def process_url(self, url):
        loader = WebBaseLoader(url)
        return self.text_splitter.split_documents(loader.load())

    def process_text(self, text):
        return self.text_splitter.split_text(text)

class EmbeddingGenerator:
    def __init__(self, model_name="llama2:latest"): # Correct model name
        self.embeddings = OllamaEmbeddings(model=model_name) # Use OllamaEmbeddings

    def generate(self, texts):
        return self.embeddings.embed_documents(texts)

class VectorDatabase:
    def __init__(self, embedding_function, persist_directory: str):
        self.db = Chroma(embedding_function=embedding_function, persist_directory=persist_directory)

    def add_documents(self, documents):
        self.db.add_documents(documents)

    def similarity_search(self, query, k=4):
        return self.db.similarity_search(query, k=k)

class RetrievalEngine:
    def __init__(self, vector_db):
        self.vector_db = vector_db

    def hybrid_search(self, query, k=4):
        semantic_results = self.vector_db.similarity_search(query, k=k)
        # Implement additional keyword-based search here
        # Combine and re-rank results
        return semantic_results  # Placeholder for combined results

class QueryProcessor:
    def __init__(self, retrieval_engine):
        self.retrieval_engine = retrieval_engine

    def process_query(self, query):
        # Implement query classification (e.g., factoid, open-ended, etc.)
        # Adjust retrieval strategy based on query type
        return self.retrieval_engine.hybrid_search(query)

class RAGLLMIntegration:
    def __init__(self, query_processor, model="mistral:latest"): # Add model parameter
        self.query_processor = query_processor
        self.llm = Ollama(model=model) # Use the provided model
        self.prompt_template = PromptTemplate(
            template="Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
            input_variables=["context", "question"]
        )

    def generate_response(self, query):
        context_docs = self.query_processor.process_query(query)
        context = "\n".join([doc.page_content for doc in context_docs])
        prompt = self.prompt_template.format(context=context, question=query)
        return self.llm(prompt)

def enhance_corpus_ui():
    st.title("🗂️ Manage Corpus")

    # Corpus folder
    corpus_folder = "corpus"
    if not os.path.exists(corpus_folder):
        os.makedirs(corpus_folder)

    # List existing corpus
    corpus_list = [f for f in os.listdir(corpus_folder) if os.path.isdir(os.path.join(corpus_folder, f))]
    st.subheader("📋 Existing Corpus")
    if corpus_list:
        for corpus in corpus_list:
            col1, col2, col3 = st.columns([6, 1, 1])  # Add a column for renaming
            with col1:
                st.write(corpus)
            with col2:
                if st.button("✏️", key=f"rename_corpus_{corpus}"):
                    st.session_state.rename_corpus = corpus
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"delete_corpus_{corpus}"):
                    shutil.rmtree(os.path.join(corpus_folder, corpus))
                    st.success(f"Corpus '{corpus}' deleted.")
                    st.rerun()
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
                st.rerun()
            else:
                st.error("Please enter a new corpus name.")

    st.subheader("✚ Create New Corpus")

    # Model Selection
    available_models = get_available_models()
    default_index = available_models.index("llama2:latest") if "llama2:latest" in available_models else 0 # Fixed model name
    selected_model = st.selectbox("Select Model", available_models, index=default_index)

    # Tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["From File", "From URL", "From Text"])

    with tab1:
        # File upload
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])
        corpus_name = st.text_input("Enter a name for the corpus:", key="create_corpus_name_file")
        if st.button("Create Corpus", key="create_corpus_button_file"):
            if uploaded_file is not None and corpus_name:
                process_and_save_corpus(uploaded_file, corpus_name, selected_model)
            else:
                st.error("Please upload a file and enter a corpus name.")

    with tab2:
        # URL input
        url = st.text_input("Enter a URL to process")
        corpus_name = st.text_input("Enter a name for the corpus:", key="create_corpus_name_url")
        if st.button("Create Corpus", key="create_corpus_button_url"):
            if url and corpus_name:
                process_and_save_corpus(url, corpus_name, selected_model, is_url=True)
            else:
                st.error("Please enter a URL and a corpus name.")

    with tab3:
        # Text input
        text_input = st.text_area("Enter text to process")
        corpus_name = st.text_input("Enter a name for the corpus:", key="create_corpus_name_text")
        if st.button("Create Corpus", key="create_corpus_button_text"):
            if text_input and corpus_name:
                process_and_save_corpus(text_input, corpus_name, selected_model, is_text=True)
            else:
                st.error("Please enter text and a corpus name.")

def process_and_save_corpus(data, corpus_name, selected_model, is_url=False, is_text=False):
    """Processes the data and saves it as a corpus."""
    corpus_folder = "corpus"
    corpus_path = os.path.join(corpus_folder, corpus_name)

    doc_processor = DocumentProcessor()
    if is_url:
        documents = doc_processor.process_url(data)
    elif is_text:
        documents = doc_processor.process_text(data)
    else:
        # Save the uploaded file to disk first
        file_path = os.path.join("files", data.name)
        with open(file_path, "wb") as f:
            f.write(data.getbuffer())
        documents = doc_processor.process_file(file_path)

    embedding_generator = EmbeddingGenerator(model_name=selected_model) # Pass selected_model here
    embeddings = embedding_generator.generate([doc.page_content for doc in documents])

    vector_db = VectorDatabase(embedding_generator.embeddings, persist_directory=corpus_path)
    vector_db.add_documents(documents)

    st.success(f"Corpus '{corpus_name}' created successfully!")