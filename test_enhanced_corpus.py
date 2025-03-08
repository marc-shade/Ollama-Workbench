#!/usr/bin/env python3
"""
Test script to verify PyPDF2 import in enhanced_corpus.py
"""

import sys
import importlib.util
import os

print("Python version:", sys.version)
print("\n=== Testing enhanced_corpus.py imports ===")

# Try to import PyPDF2 directly
try:
    import PyPDF2
    print("✅ Successfully imported PyPDF2 (uppercase)")
    print("PyPDF2 version:", getattr(PyPDF2, "__version__", "unknown"))
    print("PyPDF2 path:", PyPDF2.__file__)
except ImportError as e:
    print("❌ Failed to import PyPDF2 (uppercase):", e)

# Try to import enhanced_corpus
try:
    import enhanced_corpus
    print("✅ Successfully imported enhanced_corpus")
    
    # Test DocumentProcessor class
    try:
        processor = enhanced_corpus.DocumentProcessor()
        print("✅ Successfully created DocumentProcessor instance")
    except Exception as e:
        print("❌ Failed to create DocumentProcessor instance:", e)
    
    # Test OllamaEmbedder class
    try:
        embedder = enhanced_corpus.OllamaEmbedder()
        print("✅ Successfully created OllamaEmbedder instance")
    except Exception as e:
        print("❌ Failed to create OllamaEmbedder instance:", e)
    
    # Test GraphRAGCorpus class
    try:
        corpus = enhanced_corpus.GraphRAGCorpus("test_corpus", embedder)
        print("✅ Successfully created GraphRAGCorpus instance")
    except Exception as e:
        print("❌ Failed to create GraphRAGCorpus instance:", e)
        
except ImportError as e:
    print("❌ Failed to import enhanced_corpus:", e)
    
    # Try to trace the import error
    print("\n=== Tracing import error ===")
    try:
        spec = importlib.util.find_spec("enhanced_corpus")
        if spec is None:
            print("❌ enhanced_corpus.py not found in Python path")
        else:
            print("✅ enhanced_corpus.py found at:", spec.origin)
            
            # Try to load the module manually
            try:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print("✅ Successfully loaded enhanced_corpus module manually")
            except Exception as e:
                print("❌ Failed to load enhanced_corpus module manually:", e)
                
                # Try to trace the specific import that's failing
                with open(spec.origin, 'r') as f:
                    lines = f.readlines()
                
                print("\nImport statements in enhanced_corpus.py:")
                for i, line in enumerate(lines):
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        print(f"{i+1}: {line.strip()}")
                        
                        # Try to import each module
                        module_name = line.strip().split()[1].split('.')[0]
                        if module_name != 'enhanced_corpus':
                            try:
                                importlib.import_module(module_name)
                                print(f"  ✅ Successfully imported {module_name}")
                            except ImportError as e:
                                print(f"  ❌ Failed to import {module_name}: {e}")
    except Exception as e:
        print("❌ Error while tracing import:", e)

print("\n=== Test complete ===")