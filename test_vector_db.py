#!/usr/bin/env python3
"""
Test vector database integration for HippoRAG embedding store.
"""

import numpy as np
import tempfile
import os
import sys
import shutil

# Add src to path for testing
sys.path.insert(0, 'src')

from hipporag.vector_db import MemoryVectorDB, FAISSVectorDB


def test_memory_vector_db():
    """Test the memory vector database implementation."""
    print("Testing MemoryVectorDB...")
    
    # Create test data
    embedding_dim = 10
    vectors = np.random.randn(100, embedding_dim).astype(np.float32)
    ids = [f"id_{i}" for i in range(100)]
    
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
    retrieved = db.get_vector("id_0")
    assert retrieved is not None
    assert retrieved.shape == (embedding_dim,)
    
    print("✓ MemoryVectorDB tests passed")


def test_faiss_vector_db():
    """Test the FAISS vector database implementation."""
    print("Testing FAISSVectorDB...")
    
    # Create test data
    embedding_dim = 10
    vectors = np.random.randn(100, embedding_dim).astype(np.float32)
    ids = [f"id_{i}" for i in range(100)]
    
    # Test Flat index
    db = FAISSVectorDB(embedding_dim, index_type="Flat")
    db.add_vectors(vectors, ids)
    assert db.get_size() == 100
    
    # Test search
    query = np.random.randn(1, embedding_dim).astype(np.float32)
    similarities, indices = db.search(query, k=5)
    
    assert similarities.shape == (1, 5)
    assert indices.shape == (1, 5)
    
    print("✓ FAISSVectorDB tests passed")


def test_embedding_store_integration():
    """Test EmbeddingStore with vector database backends."""
    print("Testing EmbeddingStore integration...")
    
    # Mock embedding model
    class MockEmbeddingModel:
        def batch_encode(self, texts):
            return [np.random.randn(10).astype(np.float32) for _ in texts]
    
    # Test with memory backend
    with tempfile.TemporaryDirectory() as temp_dir:
        from hipporag.embedding_store import EmbeddingStore
        
        config = {"backend": "memory"}
        store = EmbeddingStore(
            MockEmbeddingModel(), 
            temp_dir, 
            batch_size=16, 
            namespace="test",
            vector_db_config=config
        )
        
        # Insert some strings
        texts = ["hello world", "foo bar", "test document"]
        store.insert_strings(texts)
        
        # Test search
        query_embedding = np.random.randn(10).astype(np.float32)
        hash_ids, similarities = store.search_similar(query_embedding, k=2)
        
        assert len(hash_ids) <= 2
        assert len(similarities) <= 2
        
        print("✓ EmbeddingStore integration tests passed")


def main():
    """Run all tests."""
    print("Running vector database integration tests...\n")
    
    try:
        test_memory_vector_db()
        test_faiss_vector_db()
        test_embedding_store_integration()
        
        print("\n✅ All tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())