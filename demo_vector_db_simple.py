#!/usr/bin/env python3
"""
Simple vector database demo without full HippoRAG dependencies.
"""

import os
import sys
import tempfile
import time
import numpy as np

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src', 'hipporag')
sys.path.insert(0, src_path)

# Direct imports
from vector_db.memory_vector_db import MemoryVectorDB
from vector_db.faiss_vector_db import FAISSVectorDB


def demo_basic_usage():
    """Demo basic vector database usage."""
    print("=== Basic Vector Database Usage ===\n")
    
    # Create test data
    embedding_dim = 384
    n_docs = 1000
    docs = [f"Document {i} about topic {i % 10}" for i in range(n_docs)]
    
    # Generate embeddings (normally done by embedding model)
    np.random.seed(42)
    embeddings = np.random.randn(n_docs, embedding_dim).astype(np.float32)
    doc_ids = [f"doc_{i}" for i in range(n_docs)]
    
    print(f"Test dataset: {n_docs} documents with {embedding_dim}-dimensional embeddings")
    
    # Test different backends
    backends = [
        ("Memory", MemoryVectorDB(embedding_dim)),
        ("FAISS Flat", FAISSVectorDB(embedding_dim, index_type="Flat")),
        ("FAISS HNSW", FAISSVectorDB(embedding_dim, index_type="HNSW"))
    ]
    
    for name, db in backends:
        print(f"\n{name} Backend:")
        
        # Add vectors
        start = time.time()
        db.add_vectors(embeddings, doc_ids)
        add_time = time.time() - start
        print(f"  Added {n_docs} vectors in {add_time:.3f}s")
        
        # Search
        query = np.random.randn(1, embedding_dim).astype(np.float32)
        start = time.time()
        similarities, indices = db.search(query, k=10)
        search_time = time.time() - start
        
        print(f"  Search took {search_time:.4f}s")
        print(f"  Found {len(similarities[0])} results")
        print(f"  Top similarity: {similarities[0][0]:.3f}")


def demo_configuration():
    """Demo configuration options."""
    print("\n=== Configuration Guide ===\n")
    
    print("To use vector databases in HippoRAG, configure these parameters:")
    print()
    print("1. **Memory Backend (Default)**:")
    print("   vector_db_backend = 'memory'")
    print("   # Uses original in-memory storage")
    print()
    print("2. **FAISS Backend (Recommended for large datasets)**:")
    print("   vector_db_backend = 'faiss'")
    print("   vector_db_index_type = 'Flat'      # Exact search, slower but accurate")
    print("   vector_db_index_type = 'IVFFlat'   # Fast approximate search")
    print("   vector_db_index_type = 'HNSW'      # Balanced speed/accuracy")
    print("   vector_db_nlist = 100              # Clusters for IVF (tune based on dataset size)")
    print()
    print("3. **Integration Example**:")
    print("""
   from hipporag.utils.config_utils import BaseConfig
   
   config = BaseConfig()
   config.vector_db_backend = 'faiss'
   config.vector_db_index_type = 'IVFFlat'
   config.vector_db_nlist = 256
   
   hipporag = HippoRAG(global_config=config, ...)
   """)


def demo_performance():
    """Demo performance comparison."""
    print("\n=== Performance Comparison ===\n")
    
    # Test with different dataset sizes
    sizes = [100, 1000, 5000]
    embedding_dim = 384
    
    print(f"Performance test with {embedding_dim}D embeddings:\n")
    
    for n_docs in sizes:
        print(f"Dataset size: {n_docs} documents")
        
        # Generate data
        np.random.seed(42)
        embeddings = np.random.randn(n_docs, embedding_dim).astype(np.float32)
        doc_ids = [f"doc_{i}" for i in range(n_docs)]
        query = np.random.randn(1, embedding_dim).astype(np.float32)
        
        # Test backends
        results = {}
        
        for name, db_class, kwargs in [
            ("Memory", MemoryVectorDB, {}),
            ("FAISS Flat", FAISSVectorDB, {"index_type": "Flat"}),
            ("FAISS HNSW", FAISSVectorDB, {"index_type": "HNSW"})
        ]:
            db = db_class(embedding_dim, **kwargs)
            
            # Add time
            start = time.time()
            db.add_vectors(embeddings, doc_ids)
            add_time = time.time() - start
            
            # Search time (average of 10 searches)
            search_times = []
            for _ in range(10):
                start = time.time()
                similarities, indices = db.search(query, k=10)
                search_times.append(time.time() - start)
            
            avg_search_time = np.mean(search_times)
            results[name] = {"add": add_time, "search": avg_search_time}
            
            print(f"  {name}: Add {add_time:.3f}s, Search {avg_search_time:.4f}s")
        
        # Calculate speedup
        if "Memory" in results and "FAISS Flat" in results:
            speedup = results["Memory"]["search"] / results["FAISS Flat"]["search"]
            print(f"  → FAISS Flat is {speedup:.1f}x faster than Memory for search")
        
        print()


def demo_scaling_benefits():
    """Demo scaling benefits of vector databases."""
    print("=== Why Vector Databases Matter ===\n")
    
    print("Current HippoRAG limitation:")
    print("• All embeddings stored in memory")
    print("• KNN search uses full matrix multiplication: O(n×d)")
    print("• Memory usage: O(n×d) where n=documents, d=embedding_dim")
    print("• Search time grows linearly with dataset size")
    print()
    
    print("Vector database benefits:")
    print("• Efficient indexing structures (IVF, HNSW, etc.)")
    print("• Logarithmic or sub-linear search complexity")
    print("• Memory-mapped storage for large datasets")
    print("• Approximate search with tunable accuracy/speed tradeoff")
    print()
    
    # Calculate memory usage for different scenarios
    embedding_dim = 384  # Common for many models
    float_size = 4  # 32-bit float
    
    print("Memory usage comparison (384D embeddings):")
    for n_docs in [10_000, 100_000, 1_000_000, 10_000_000]:
        memory_mb = (n_docs * embedding_dim * float_size) / (1024 * 1024)
        
        if memory_mb < 1024:
            memory_str = f"{memory_mb:.1f} MB"
        else:
            memory_str = f"{memory_mb/1024:.1f} GB"
        
        print(f"  {n_docs:,} docs: {memory_str}")
        
        if memory_mb > 8000:  # > 8GB
            print(f"    ⚠️  Likely too large for in-memory approach")
        elif memory_mb > 2000:  # > 2GB  
            print(f"    ⚠️  Consider vector database for better performance")


def main():
    """Run all demos."""
    print("HippoRAG Vector Database Integration")
    print("=" * 50)
    
    try:
        demo_basic_usage()
        demo_performance()
        demo_scaling_benefits()
        demo_configuration()
        
        print("\n" + "=" * 50)
        print("✅ Vector database integration is ready!")
        print("\nKey takeaways:")
        print("• FAISS provides significant performance improvements")
        print("• Memory backend preserved for backward compatibility")
        print("• Configuration is simple and flexible")
        print("• Essential for scaling to large document collections")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())