# chroma_client.py

import os
import chromadb
from chromadb.config import Settings

def get_chroma_client(corpus_name):
    corpus_path = os.path.join("corpus", corpus_name)
    if not os.path.exists(corpus_path):
        os.makedirs(corpus_path)

    # Configure the Chroma client with the new API structure
    settings = Settings(
        chroma_api_impl="rest",  # Can also be "sqlite" or "duckdb+parquet" based on your setup
        chroma_server_host="localhost",  # Replace with your Chroma server host if remote
        chroma_server_port="8000",  # Replace with the appropriate port
        persist_directory=corpus_path  # Directory to store the persistent data
    )
    return chromadb.Client(settings=settings)

def sanitize_collection_name(name):
    sanitized_name = name.replace(" ", "_")
    sanitized_name = ''.join(c for c in sanitized_name if c.isalnum() or c in ['_', '-'])
    return sanitized_name
