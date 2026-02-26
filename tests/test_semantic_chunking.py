"""
Test script for the semantic chunking functionality in the advanced RAG system.
"""

import os
import sys
import logging
import time
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from advanced_rag import DocumentChunker, Document, DocumentTypes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_semantic_chunking')

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
    plot_dir = "test_results"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "chunk_size_comparison.png"))
    logger.info(f"Plot saved to {os.path.join(plot_dir, 'chunk_size_comparison.png')}")

def test_embedded_similarity(results: Dict[str, List[Dict[str, Any]]]) -> None:
    """Test the semantic coherence of chunks by calculating similarity between consecutive chunks."""
    from ollama_workbench.providers.ollama_utils import get_ollama_client
    
    # Try to get embeddings from Ollama
    try:
        client = get_ollama_client()
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
                    response = client.embeddings(model=model, prompt=chunk["content"])
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
        plot_dir = "test_results"
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, "chunk_similarity_comparison.png"))
        logger.info(f"Plot saved to {os.path.join(plot_dir, 'chunk_similarity_comparison.png')}")
        
    except Exception as e:
        logger.error(f"Error in similarity analysis: {str(e)}")

def main():
    """Run tests with different document types."""
    # Create test directory
    test_dir = "test_results"
    os.makedirs(test_dir, exist_ok=True)
    
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