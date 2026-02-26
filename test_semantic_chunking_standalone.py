"""
Standalone test script for the semantic chunking functionality.
This script tests the semantic chunking implementation without dependencies on other modules.
"""

import os
import logging
import re
import time
import uuid
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity
import ollama

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_semantic_chunking')

# Constants for the test
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Create test directory
TEST_DIR = "test_results"
os.makedirs(TEST_DIR, exist_ok=True)

class DocumentTypes:
    """Enumeration of supported document types."""
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"

class Document:
    """Class representing a document with its metadata and content."""
    
    def __init__(
        self, 
        content: str,
        doc_id: Optional[str] = None,
        title: Optional[str] = None,
        source: Optional[str] = None,
        doc_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
        embedding: Optional[np.ndarray] = None,
    ):
        self.doc_id = doc_id or str(uuid.uuid4())
        self.content = content
        self.title = title or f"Document-{self.doc_id[:8]}"
        self.source = source
        self.doc_type = doc_type or DocumentTypes.TEXT
        self.metadata = metadata or {}
        self.chunks = chunks or []
        self.embedding = embedding

class DocumentChunker:
    """Class for chunking documents using various strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the document chunker with the given configuration."""
        self.config = config or {
            "method": "semantic",
            "size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
            "unit": "characters",
            "respect_boundaries": True,
        }
        self.semantic_chunking_enabled = True
    
    def chunk_document(self, document: Document) -> List[Dict[str, Any]]:
        """Chunk a document based on the configured strategy."""
        method = self.config.get("method", "fixed")
        
        if method == "semantic" and self.semantic_chunking_enabled:
            try:
                chunks = self._semantic_chunking(document)
                logger.info(f"Semantic chunking created {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Semantic chunking failed: {str(e)}. Falling back to paragraph chunking.")
                chunks = self._paragraph_chunking(document)
        elif method == "sliding":
            chunks = self._sliding_window_chunking(document)
        elif method == "paragraph":
            chunks = self._paragraph_chunking(document)
        else:  # Default to fixed-size chunking
            chunks = self._fixed_size_chunking(document)
            
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk["doc_id"] = document.doc_id
            chunk["chunk_id"] = f"{document.doc_id}-{i}"
            chunk["document_title"] = document.title
            chunk["position"] = i
            
        return chunks
    
    def _fixed_size_chunking(self, document: Document) -> List[Dict[str, Any]]:
        """Split text into fixed-size chunks."""
        text = document.content
        size = self.config.get("size", 1000)
        overlap = self.config.get("overlap", 200)
        respect_boundaries = self.config.get("respect_boundaries", True)
        
        chunks = []
        
        if respect_boundaries:
            # Split by sentences first
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > size and current_chunk:
                    chunks.append({"content": current_chunk.strip()})
                    # Keep overlap if possible
                    words = current_chunk.split()
                    overlap_word_count = min(len(words), int(overlap / 5))  # Approximate words in overlap
                    if overlap_word_count > 0:
                        current_chunk = " ".join(words[-overlap_word_count:]) + " "
                    else:
                        current_chunk = ""
                
                current_chunk += sentence + " "
            
            # Add the last chunk if it's not empty
            if current_chunk.strip():
                chunks.append({"content": current_chunk.strip()})
        else:
            # Simple character-based chunking
            for i in range(0, len(text), size - overlap):
                chunk_text = text[i:i + size]
                chunks.append({"content": chunk_text.strip()})
                
        return chunks
    
    def _sliding_window_chunking(self, document: Document) -> List[Dict[str, Any]]:
        """Split text using a sliding window approach with more granular control."""
        text = document.content
        size = self.config.get("size", 1000)
        overlap = self.config.get("overlap", 200)
        step = size - overlap
        
        chunks = []
        
        for i in range(0, len(text), step):
            chunk_text = text[i:i + size]
            if len(chunk_text.strip()) > 0:
                chunks.append({"content": chunk_text.strip()})
                
        return chunks
    
    def _paragraph_chunking(self, document: Document) -> List[Dict[str, Any]]:
        """Split text by paragraphs and then combine into chunks of appropriate size."""
        text = document.content
        size = self.config.get("size", 1000)
        
        # Split by paragraphs (double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > size and current_chunk:
                chunks.append({"content": current_chunk.strip()})
                current_chunk = ""
            
            current_chunk += paragraph + "\n\n"
        
        # Add the last chunk if it's not empty
        if current_chunk.strip():
            chunks.append({"content": current_chunk.strip()})
            
        return chunks
    
    def _semantic_chunking(self, document: Document) -> List[Dict[str, Any]]:
        """
        Split text into semantically coherent chunks.
        
        This uses a more advanced approach that tries to keep related content together by:
        1. First splitting into paragraphs
        2. Then calculating semantic similarity between paragraphs
        3. Combining similar paragraphs into chunks up to the size limit
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Set the max size for chunks (in characters)
        max_chunk_size = self.config.get("size", 1000)
        text = document.content
        
        # Step 1: Split the text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # If very few paragraphs, just use paragraph chunking
        if len(paragraphs) <= 2:
            return self._paragraph_chunking(document)
        
        # Step 2: Calculate embeddings for each paragraph
        try:
            # Initialize embedder
            model = "nomic-embed-text"  # Default embedding model
            
            # Get embeddings for each paragraph
            paragraph_embeddings = []
            for paragraph in paragraphs:
                try:
                    response = ollama.embeddings(model=model, prompt=paragraph)
                    embedding = np.array(response['embedding'])
                    paragraph_embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Error getting embedding for paragraph: {str(e)}")
                    # If embedding fails, use an array of zeros as a placeholder
                    paragraph_embeddings.append(np.zeros(768))  # Typical embedding size
            
            # Step 3: Calculate similarity matrix between paragraphs
            similarity_matrix = cosine_similarity(paragraph_embeddings)
            
            # Step 4: Form chunks by combining similar paragraphs
            chunks = []
            current_chunk_paragraphs = [0]  # Start with the first paragraph
            current_chunk_text = paragraphs[0]
            visited = {0}  # Keep track of paragraphs already in chunks
            
            # Process each paragraph (starting from the second one)
            for i in range(1, len(paragraphs)):
                if i in visited:
                    continue
                    
                # Find the paragraph in the current chunk with highest similarity to paragraph i
                similarities = [similarity_matrix[j][i] for j in current_chunk_paragraphs]
                max_similarity = max(similarities) if similarities else 0
                
                # If the similarity is high enough and adding won't exceed size limit, add to current chunk
                new_chunk_text = current_chunk_text + "\n\n" + paragraphs[i]
                if max_similarity > 0.5 and len(new_chunk_text) <= max_chunk_size:
                    current_chunk_paragraphs.append(i)
                    current_chunk_text = new_chunk_text
                    visited.add(i)
                else:
                    # Save current chunk and start a new one
                    chunks.append({"content": current_chunk_text.strip()})
                    current_chunk_paragraphs = [i]
                    current_chunk_text = paragraphs[i]
                    visited.add(i)
            
            # Add the last chunk if it's not empty
            if current_chunk_text.strip():
                chunks.append({"content": current_chunk_text.strip()})
                
            # Process remaining paragraphs (if any weren't visited yet)
            remaining = [i for i in range(len(paragraphs)) if i not in visited]
            for i in remaining:
                # Try to find the most similar existing chunk
                max_similarity = 0
                best_chunk = -1
                
                for j, chunk in enumerate(chunks):
                    # Find paragraphs in this chunk
                    chunk_text = chunk["content"]
                    for k in range(len(paragraphs)):
                        if k in visited and paragraphs[k] in chunk_text:
                            # Calculate similarity between paragraph i and paragraph k
                            similarity = similarity_matrix[i][k]
                            if similarity > max_similarity:
                                max_similarity = similarity
                                best_chunk = j
                
                # If found a similar chunk and adding won't exceed size limit, add to that chunk
                if best_chunk >= 0 and max_similarity > 0.5:
                    new_content = chunks[best_chunk]["content"] + "\n\n" + paragraphs[i]
                    if len(new_content) <= max_chunk_size:
                        chunks[best_chunk]["content"] = new_content
                        visited.add(i)
                        continue
                
                # Otherwise, create a new chunk
                chunks.append({"content": paragraphs[i].strip()})
                visited.add(i)
                
            # Ensure all paragraphs are included
            if len(visited) < len(paragraphs):
                # Fall back to paragraph chunking if our algorithm missed any paragraphs
                logger.warning(f"Semantic chunking missed some paragraphs. Falling back to paragraph chunking.")
                return self._paragraph_chunking(document)
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {str(e)}")
            # Fall back to paragraph chunking if any errors
            return self._paragraph_chunking(document)

def create_test_document(text: str, doc_type: str = DocumentTypes.TEXT) -> Document:
    """Create a test document with the given text and type."""
    return Document(
        content=text,
        title=f"Test Document - {doc_type}",
        source="Test",
        doc_type=doc_type
    )

def test_chunking_methods(text: str, doc_type: str = DocumentTypes.TEXT) -> Dict[str, List[Dict[str, Any]]]:
    """Test different chunking methods on the same document."""
    document = create_test_document(text, doc_type)
    
    # Create document chunker with default config
    chunker = DocumentChunker()
    
    # Test different chunking methods
    results = {}
    
    # Fixed-size chunking
    chunker.config["method"] = "fixed"
    results["fixed"] = chunker.chunk_document(document)
    
    # Sliding window chunking
    chunker.config["method"] = "sliding"
    results["sliding"] = chunker.chunk_document(document)
    
    # Paragraph chunking
    chunker.config["method"] = "paragraph"
    results["paragraph"] = chunker.chunk_document(document)
    
    # Semantic chunking
    chunker.config["method"] = "semantic"
    results["semantic"] = chunker.chunk_document(document)
    
    return results

def compare_chunk_sizes(results: Dict[str, List[Dict[str, Any]]]) -> None:
    """Compare the sizes of chunks created by different methods."""
    chunk_sizes = {}
    
    for method, chunks in results.items():
        chunk_sizes[method] = [len(chunk["content"]) for chunk in chunks]
        avg_size = sum(chunk_sizes[method]) / len(chunk_sizes[method]) if chunk_sizes[method] else 0
        
        logger.info(f"{method.capitalize()} chunking: {len(chunks)} chunks, avg size: {avg_size:.1f} chars")
    
    # Plot chunk size distributions
    plt.figure(figsize=(12, 6))
    
    for method, sizes in chunk_sizes.items():
        plt.plot(sizes, label=f"{method.capitalize()} (n={len(sizes)})")
    
    plt.title("Chunk Size Distribution by Method")
    plt.xlabel("Chunk Index")
    plt.ylabel("Chunk Size (chars)")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = os.path.join(TEST_DIR, "chunk_size_comparison.png")
    plt.savefig(plot_path)
    logger.info(f"Plot saved to {plot_path}")

def test_embedded_similarity(results: Dict[str, List[Dict[str, Any]]]) -> None:
    """Test the semantic coherence of chunks by calculating similarity between consecutive chunks."""
    # Try to get embeddings from Ollama
    try:
        model = "nomic-embed-text"
        
        # Calculate semantic similarity within each chunking method
        similarities = {}
        
        for method, chunks in results.items():
            if len(chunks) <= 1:
                logger.warning(f"{method.capitalize()} chunking resulted in only {len(chunks)} chunks, skipping similarity analysis")
                continue
                
            # Get embeddings for each chunk
            embeddings = []
            for chunk in chunks:
                try:
                    response = ollama.embeddings(model=model, prompt=chunk["content"])
                    embedding = np.array(response['embedding'])
                    embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Error getting embedding: {str(e)}")
                    return  # Can't continue without embeddings
            
            # Calculate similarity between consecutive chunks
            chunk_similarities = []
            for i in range(len(embeddings) - 1):
                similarity = np.dot(embeddings[i], embeddings[i+1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
                chunk_similarities.append(similarity)
            
            similarities[method] = chunk_similarities
            avg_similarity = sum(chunk_similarities) / len(chunk_similarities) if chunk_similarities else 0
            logger.info(f"{method.capitalize()} chunking: avg similarity between consecutive chunks: {avg_similarity:.4f}")
        
        # Plot similarities
        plt.figure(figsize=(12, 6))
        
        for method, sims in similarities.items():
            plt.plot(sims, label=f"{method.capitalize()} (n={len(sims)})")
        
        plt.title("Semantic Coherence: Similarity Between Consecutive Chunks")
        plt.xlabel("Chunk Pair Index")
        plt.ylabel("Cosine Similarity")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(TEST_DIR, "chunk_similarity_comparison.png")
        plt.savefig(plot_path)
        logger.info(f"Plot saved to {plot_path}")
        
    except Exception as e:
        logger.error(f"Error in similarity analysis: {str(e)}")

def main():
    """Run tests with different document types."""
    # Test with a sample text document - coherent paragraphs on the same topic
    logger.info("Testing with coherent text document...")
    coherent_text = """
    # Introduction to Machine Learning
    
    Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.
    
    Machine learning is an important component of the growing field of data science. Through the use of statistical methods, algorithms are trained to make classifications or predictions, uncovering key insights within data mining projects.
    
    # Types of Machine Learning
    
    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
    
    ## Supervised Learning
    
    Supervised learning involves training a model on a labeled dataset, which means the input data is paired with the correct output. The goal is for the algorithm to learn the mapping function from the input to the output.
    
    Common supervised learning algorithms include:
    - Linear regression
    - Logistic regression
    - Support vector machines
    - Decision trees and random forests
    - Neural networks
    
    ## Unsupervised Learning
    
    Unsupervised learning involves training a model on an unlabeled dataset, without any guidance. The algorithm tries to find patterns or groupings within the data.
    
    Common unsupervised learning algorithms include:
    - Clustering algorithms (like K-means)
    - Association rules
    - Dimensionality reduction
    
    ## Reinforcement Learning
    
    Reinforcement learning is about training agents to make decisions by rewarding desired behaviors and punishing undesired ones. It's different from supervised learning because the correct input/output pairs are never presented, nor are sub-optimal actions explicitly corrected.
    
    # Applications of Machine Learning
    
    Machine learning has numerous applications across various industries:
    
    - Healthcare: Disease identification, patient monitoring
    - Finance: Fraud detection, risk assessment
    - Retail: Recommendation systems, inventory planning
    - Manufacturing: Predictive maintenance, quality control
    - Transportation: Autonomous vehicles, traffic prediction
    
    # Conclusion
    
    Machine learning continues to evolve rapidly, with new techniques and applications emerging regularly. As datasets grow larger and computing power increases, the potential applications of machine learning will continue to expand across industries.
    """
    coherent_results = test_chunking_methods(coherent_text, DocumentTypes.MARKDOWN)
    compare_chunk_sizes(coherent_results)
    test_embedded_similarity(coherent_results)
    
    # Test with disjointed text - paragraphs on different topics
    logger.info("\nTesting with disjointed text document...")
    disjointed_text = """
    # Bitcoin
    
    Bitcoin is a decentralized digital currency, without a central bank or single administrator, that can be sent from user to user on the peer-to-peer bitcoin network without the need for intermediaries. Transactions are verified by network nodes through cryptography and recorded in a public distributed ledger called a blockchain.
    
    # Gardening
    
    Gardening is the practice of growing and cultivating plants as part of horticulture. In gardens, ornamental plants are often grown for their flowers, foliage, or overall appearance; useful plants, such as root vegetables, leaf vegetables, fruits, and herbs, are grown for consumption.
    
    # Quantum Physics
    
    Quantum physics, also known as quantum mechanics, is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum information science.
    
    # Baking Bread
    
    Bread is a staple food prepared from a dough of flour and water, usually by baking. Throughout recorded history, it has been a prominent food in large parts of the world. It is one of the oldest human-made foods, having been of significant importance since the dawn of agriculture.
    
    # Marine Biology
    
    Marine biology is the scientific study of marine life, organisms in the sea. Given that in biology many phyla, families and genera have some species that live in the sea and others that live on land, marine biology classifies species based on the environment rather than on taxonomy.
    
    # Classical Music
    
    Classical music is art music produced or rooted in the traditions of Western culture, including both liturgical (religious) and secular music. While a more precise term is also used to refer to the period from 1750 to 1820 (the Classical period), this article is about the broad span of time from before the 6th century AD to the present day.
    """
    disjointed_results = test_chunking_methods(disjointed_text, DocumentTypes.MARKDOWN)
    compare_chunk_sizes(disjointed_results)
    test_embedded_similarity(disjointed_results)
    
    # Test with code document
    logger.info("\nTesting with code document...")
    code_text = """
    # Python implementation of various sorting algorithms
    
    def bubble_sort(arr):
        \"\"\"
    Bubble Sort: A simple sorting algorithm that repeatedly steps through the list,
    compares adjacent elements and swaps them if they are in the wrong order.
    Time Complexity: O(n^2)
        \"\"\"
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr
    
    def selection_sort(arr):
        \"\"\"
    Selection Sort: Sorts an array by repeatedly finding the minimum element
    from the unsorted part and putting it at the beginning.
    Time Complexity: O(n^2)
        \"\"\"
        n = len(arr)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr
    
    def insertion_sort(arr):
        \"\"\"
    Insertion Sort: Builds the sorted array one item at a time by
    comparing each new element to the already-sorted elements.
    Time Complexity: O(n^2)
        \"\"\"
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr
    
    def merge_sort(arr):
        \"\"\"
    Merge Sort: A divide and conquer algorithm that divides the input array into two halves,
    recursively sorts them, and then merges the sorted halves.
    Time Complexity: O(n log n)
        \"\"\"
        if len(arr) > 1:
            mid = len(arr) // 2
            L = arr[:mid]
            R = arr[mid:]
            
            merge_sort(L)
            merge_sort(R)
            
            i = j = k = 0
            
            while i < len(L) and j < len(R):
                if L[i] < R[j]:
                    arr[k] = L[i]
                    i += 1
                else:
                    arr[k] = R[j]
                    j += 1
                k += 1
            
            while i < len(L):
                arr[k] = L[i]
                i += 1
                k += 1
                
            while j < len(R):
                arr[k] = R[j]
                j += 1
                k += 1
        return arr
    
    def quick_sort(arr):
        \"\"\"
    Quick Sort: Another divide and conquer algorithm that picks an element as a pivot and
    partitions the array around the pivot.
    Time Complexity: Average O(n log n), Worst O(n^2)
        \"\"\"
        if len(arr) <= 1:
            return arr
        else:
            pivot = arr[0]
            less = [x for x in arr[1:] if x <= pivot]
            greater = [x for x in arr[1:] if x > pivot]
            return quick_sort(less) + [pivot] + quick_sort(greater)
    
    def heap_sort(arr):
        \"\"\"
    Heap Sort: Converts the array into a max heap, then repeatedly extracts the maximum element.
    Time Complexity: O(n log n)
        \"\"\"
        def heapify(arr, n, i):
            largest = i
            l = 2 * i + 1
            r = 2 * i + 2
            
            if l < n and arr[i] < arr[l]:
                largest = l
                
            if r < n and arr[largest] < arr[r]:
                largest = r
                
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(arr, n, largest)
        
        n = len(arr)
        
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)
            
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            heapify(arr, i, 0)
        
        return arr
    """
    code_results = test_chunking_methods(code_text, DocumentTypes.CODE)
    compare_chunk_sizes(code_results)
    test_embedded_similarity(code_results)
    
    logger.info("All tests completed.")

if __name__ == "__main__":
    main()