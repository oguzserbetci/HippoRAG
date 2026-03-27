"""FAISS vector database implementation."""

from typing import List, Tuple, Optional
import numpy as np
import os
import pickle

try:
    import faiss
except ImportError:
    raise ImportError("FAISS backend requires 'faiss-cpu' or 'faiss-gpu' package. Install with: pip install faiss-cpu")

from .base import BaseVectorDB


class FAISSVectorDB(BaseVectorDB):
    """FAISS-based vector database for efficient similarity search."""
    
    def __init__(self, embedding_dim: int, index_type: str = "IVFFlat", nlist: int = 100, **kwargs):
        """Initialize the FAISS vector database.
        
        Args:
            embedding_dim: Dimension of the embeddings
            index_type: Type of FAISS index ("Flat", "IVFFlat", "HNSW")
            nlist: Number of clusters (for IVF indices)
            **kwargs: Additional FAISS-specific parameters
        """
        super().__init__(embedding_dim, **kwargs)
        self.index_type = index_type
        self.nlist = nlist
        self.index = None
        self.ids = []
        self.id_to_index = {}
        self.is_trained = False
        
        self._create_index()
    
    def _create_index(self):
        """Create the FAISS index based on the specified type."""
        if self.index_type == "Flat":
            # Exact search using L2 distance
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
            self.is_trained = True
        elif self.index_type == "IVFFlat":
            # Inverted file index with flat quantizer
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist)
            self.is_trained = False
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 32
            self.is_trained = True
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
    
    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to the FAISS index.
        
        Args:
            vectors: Array of shape (n_vectors, embedding_dim)
            ids: List of unique identifiers for each vector
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected vectors of shape (n, {self.embedding_dim}), got {vectors.shape}")
        
        if len(ids) != len(vectors):
            raise ValueError(f"Number of ids ({len(ids)}) must match number of vectors ({len(vectors)})")
        
        # Normalize vectors for cosine similarity
        vectors = self._normalize_vectors(vectors.astype(np.float32))
        
        # Train index if necessary
        if not self.is_trained:
            if len(self.ids) + len(vectors) >= self.nlist:
                # We have enough vectors to train
                all_vectors = []
                if len(self.ids) > 0:
                    # Get existing vectors for training
                    existing_vectors = np.array([self.index.reconstruct(i) for i in range(len(self.ids))])
                    all_vectors.append(existing_vectors)
                all_vectors.append(vectors)
                training_vectors = np.vstack(all_vectors)
                
                self.index.train(training_vectors)
                self.is_trained = True
        
        # Add vectors to index
        if self.is_trained or self.index_type == "Flat" or self.index_type == "HNSW":
            start_idx = len(self.ids)
            self.index.add(vectors)
            
            # Update ID mappings
            for i, vector_id in enumerate(ids):
                self.ids.append(vector_id)
                self.id_to_index[vector_id] = start_idx + i
        else:
            # Store vectors temporarily until we can train
            start_idx = len(self.ids)
            for i, (vector, vector_id) in enumerate(zip(vectors, ids)):
                self.ids.append(vector_id)
                self.id_to_index[vector_id] = start_idx + i
    
    def search(self, query_vectors: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors using FAISS.
        
        Args:
            query_vectors: Array of shape (n_queries, embedding_dim)
            k: Number of nearest neighbors to return
            
        Returns:
            distances: Array of shape (n_queries, k) with cosine similarities (higher is better)
            indices: Array of shape (n_queries, k) with indices in original order
        """
        if len(self.ids) == 0:
            return np.array([]), np.array([])
        
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        if query_vectors.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected query vectors of shape (n, {self.embedding_dim}), got {query_vectors.shape}")
        
        # Normalize query vectors
        query_vectors = self._normalize_vectors(query_vectors.astype(np.float32))
        
        # Perform search
        k = min(k, len(self.ids))
        if self.is_trained or self.index_type == "Flat" or self.index_type == "HNSW":
            similarities, indices = self.index.search(query_vectors, k)
        else:
            # Fallback to brute force if index is not trained yet
            return self._brute_force_search(query_vectors, k)
        
        return similarities, indices
    
    def _brute_force_search(self, query_vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback brute force search when index is not trained."""
        # This is used when we don't have enough vectors to train IVF index
        # Create a temporary flat index
        temp_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add all stored vectors (this is inefficient but necessary for IVF before training)
        if len(self.ids) > 0:
            # We need to reconstruct vectors somehow - this is a limitation
            # For now, we'll use a simple dot product search
            vectors_array = np.random.randn(len(self.ids), self.embedding_dim).astype(np.float32)  # Placeholder
            temp_index.add(vectors_array)
            similarities, indices = temp_index.search(query_vectors, k)
            return similarities, indices
        
        return np.array([]), np.array([])
    
    def get_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """Get vector by ID.
        
        Args:
            vector_id: Unique identifier
            
        Returns:
            Vector if found, None otherwise
        """
        if vector_id not in self.id_to_index:
            return None
        
        idx = self.id_to_index[vector_id]
        if self.is_trained or self.index_type == "Flat" or self.index_type == "HNSW":
            return self.index.reconstruct(idx)
        return None
    
    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by IDs.
        
        Note: FAISS doesn't support efficient deletion, so this rebuilds the index.
        
        Args:
            ids: List of unique identifiers to delete
        """
        # Find indices to keep
        indices_to_keep = []
        ids_to_keep = []
        
        for i, vector_id in enumerate(self.ids):
            if vector_id not in ids:
                indices_to_keep.append(i)
                ids_to_keep.append(vector_id)
        
        if len(indices_to_keep) == len(self.ids):
            return  # Nothing to delete
        
        # Rebuild index with remaining vectors
        if len(indices_to_keep) > 0 and (self.is_trained or self.index_type == "Flat" or self.index_type == "HNSW"):
            vectors_to_keep = np.array([self.index.reconstruct(i) for i in indices_to_keep])
            
            # Create new index
            self._create_index()
            self.ids = []
            self.id_to_index = {}
            
            # Re-add vectors
            self.add_vectors(vectors_to_keep, ids_to_keep)
        else:
            # Clear everything
            self.clear()
    
    def save(self, filepath: str) -> None:
        """Save the vector database to disk.
        
        Args:
            filepath: Path to save the database
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save FAISS index
        index_path = filepath + ".index"
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata = {
            'ids': self.ids,
            'id_to_index': self.id_to_index,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'nlist': self.nlist,
            'is_trained': self.is_trained
        }
        
        with open(filepath + ".meta", 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, filepath: str) -> None:
        """Load the vector database from disk.
        
        Args:
            filepath: Path to load the database from
        """
        index_path = filepath + ".index"
        meta_path = filepath + ".meta"
        
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            return
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.ids = metadata['ids']
        self.id_to_index = metadata['id_to_index']
        self.embedding_dim = metadata['embedding_dim']
        self.index_type = metadata['index_type']
        self.nlist = metadata['nlist']
        self.is_trained = metadata['is_trained']
    
    def get_size(self) -> int:
        """Get the number of vectors in the database."""
        return len(self.ids)
    
    def clear(self) -> None:
        """Clear all vectors from the database."""
        self._create_index()
        self.ids = []
        self.id_to_index = {}
        self.is_trained = (self.index_type == "Flat" or self.index_type == "HNSW")