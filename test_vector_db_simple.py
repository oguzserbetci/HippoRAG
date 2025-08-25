#!/usr/bin/env python3
"""
Simple test for vector database functionality without full HippoRAG dependencies.
"""

import numpy as np
import tempfile
import os
import sys

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src', 'hipporag')
sys.path.insert(0, src_path)

# Direct imports
from vector_db.base import BaseVectorDB
from vector_db.memory_vector_db import MemoryVectorDB
from vector_db.faiss_vector_db import FAISSVectorDB


def test_memory_vector_db():
    """Test the memory vector database implementation."""
    print("Testing MemoryVectorDB...")
    
    # Create test data
    embedding_dim = 384  # Common embedding dimension
    vectors = np.random.randn(100, embedding_dim).astype(np.float32)
    ids = [f"doc_{i}" for i in range(100)]
    
    # Initialize database
    db = MemoryVectorDB(embedding_dim)
    
    # Add vectors
    db.add_vectors(vectors, ids)
    assert db.get_size() == 100
    
    # Test search
    query = np.random.randn(1, embedding_dim).astype(np.float32)
    similarities, indices = db.search(query, k=5)
    
    assert similarities.shape == (1, 5)
    assert indices.shape == (1, 5)
    
    # Test get vector
    retrieved = db.get_vector("doc_0")
    assert retrieved is not None
    assert retrieved.shape == (embedding_dim,)
    
    # Test save/load
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test_db")
        db.save(filepath)
        
        new_db = MemoryVectorDB(embedding_dim)
        new_db.load(filepath)
        assert new_db.get_size() == 100
    
    print("✓ MemoryVectorDB tests passed")


def test_faiss_vector_db():
    """Test the FAISS vector database implementation."""
    print("Testing FAISSVectorDB...")
    
    # Create test data
    embedding_dim = 384
    vectors = np.random.randn(100, embedding_dim).astype(np.float32)
    ids = [f"doc_{i}" for i in range(100)]
    
    # Test Flat index
    print("  Testing Flat index...")
    db = FAISSVectorDB(embedding_dim, index_type="Flat")
    db.add_vectors(vectors, ids)
    assert db.get_size() == 100
    
    # Test search
    query = np.random.randn(1, embedding_dim).astype(np.float32)
    similarities, indices = db.search(query, k=5)
    
    assert similarities.shape == (1, 5)
    assert indices.shape == (1, 5)
    
    # Test save/load
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test_faiss_db")
        db.save(filepath)
        
        new_db = FAISSVectorDB(embedding_dim, index_type="Flat")
        new_db.load(filepath)
        assert new_db.get_size() == 100
    
    # Test HNSW index
    print("  Testing HNSW index...")
    hnsw_db = FAISSVectorDB(embedding_dim, index_type="HNSW")
    hnsw_db.add_vectors(vectors, ids)
    assert hnsw_db.get_size() == 100
    
    similarities, indices = hnsw_db.search(query, k=5)
    assert similarities.shape == (1, 5)
    assert indices.shape == (1, 5)
    
    print("✓ FAISSVectorDB tests passed")


def test_vector_db_performance():
    """Test performance difference between memory and FAISS."""
    print("Testing performance comparison...")
    
    import time
    
    embedding_dim = 384
    n_vectors = 1000
    vectors = np.random.randn(n_vectors, embedding_dim).astype(np.float32)
    ids = [f"doc_{i}" for i in range(n_vectors)]
    query = np.random.randn(1, embedding_dim).astype(np.float32)
    
    # Test Memory DB
    memory_db = MemoryVectorDB(embedding_dim)
    start = time.time()
    memory_db.add_vectors(vectors, ids)
    memory_add_time = time.time() - start
    
    start = time.time()
    similarities, indices = memory_db.search(query, k=10)
    memory_search_time = time.time() - start
    
    # Test FAISS DB
    faiss_db = FAISSVectorDB(embedding_dim, index_type="Flat")
    start = time.time()
    faiss_db.add_vectors(vectors, ids)
    faiss_add_time = time.time() - start
    
    start = time.time()
    similarities, indices = faiss_db.search(query, k=10)
    faiss_search_time = time.time() - start
    
    print(f"  Memory DB - Add: {memory_add_time:.4f}s, Search: {memory_search_time:.4f}s")
    print(f"  FAISS DB  - Add: {faiss_add_time:.4f}s, Search: {faiss_search_time:.4f}s")
    
    print("✓ Performance comparison completed")


def main():
    """Run all tests."""
    print("Running vector database tests...\n")
    
    try:
        test_memory_vector_db()
        test_faiss_vector_db()
        test_vector_db_performance()
        
        print("\n✅ All vector database tests passed!")
        print("\nVector database integration is ready for HippoRAG!")
        print("To use FAISS backend, set vector_db_backend='faiss' in your configuration.")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())