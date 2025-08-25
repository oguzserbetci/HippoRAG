"""In-memory vector database implementation (original behavior)."""

from typing import List, Tuple, Optional
import numpy as np
import pickle
import os

from .base import BaseVectorDB


class MemoryVectorDB(BaseVectorDB):
    """In-memory vector database that replicates the original EmbeddingStore behavior."""
    
    def __init__(self, embedding_dim: int, **kwargs):
        """Initialize the in-memory vector database.
        
        Args:
            embedding_dim: Dimension of the embeddings
            **kwargs: Ignored for memory backend
        """
        super().__init__(embedding_dim, **kwargs)
        self.vectors = []
        self.ids = []
        self.id_to_index = {}
    
    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to the in-memory storage.
        
        Args:
            vectors: Array of shape (n_vectors, embedding_dim)
            ids: List of unique identifiers for each vector
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected vectors of shape (n, {self.embedding_dim}), got {vectors.shape}")
        
        if len(ids) != len(vectors):
            raise ValueError(f"Number of ids ({len(ids)}) must match number of vectors ({len(vectors)})")
        
        # Add vectors
        start_idx = len(self.vectors)
        for i, (vector, vector_id) in enumerate(zip(vectors, ids)):
            if vector_id in self.id_to_index:
                # Update existing vector
                old_idx = self.id_to_index[vector_id]
                self.vectors[old_idx] = vector
            else:
                # Add new vector
                self.vectors.append(vector)
                self.ids.append(vector_id)
                self.id_to_index[vector_id] = start_idx + i
    
    def search(self, query_vectors: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors using cosine similarity.
        
        Args:
            query_vectors: Array of shape (n_queries, embedding_dim)
            k: Number of nearest neighbors to return
            
        Returns:
            distances: Array of shape (n_queries, k) with cosine similarities (higher is better)
            indices: Array of shape (n_queries, k) with indices in original order
        """
        if len(self.vectors) == 0:
            return np.array([]), np.array([])
        
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        if query_vectors.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected query vectors of shape (n, {self.embedding_dim}), got {query_vectors.shape}")
        
        # Convert to numpy array
        vectors_array = np.array(self.vectors)
        
        # Normalize vectors for cosine similarity
        query_norm = query_vectors / (np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-10)
        vectors_norm = vectors_array / (np.linalg.norm(vectors_array, axis=1, keepdims=True) + 1e-10)
        
        # Compute cosine similarities
        similarities = np.dot(query_norm, vectors_norm.T)
        
        # Get top k for each query
        k = min(k, len(self.vectors))
        top_k_indices = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]  # Sort descending
        
        # Get corresponding similarities
        batch_indices = np.arange(similarities.shape[0])[:, np.newaxis]
        top_k_similarities = similarities[batch_indices, top_k_indices]
        
        return top_k_similarities, top_k_indices
    
    def get_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """Get vector by ID.
        
        Args:
            vector_id: Unique identifier
            
        Returns:
            Vector if found, None otherwise
        """
        if vector_id in self.id_to_index:
            idx = self.id_to_index[vector_id]
            return self.vectors[idx]
        return None
    
    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by IDs.
        
        Args:
            ids: List of unique identifiers to delete
        """
        # Find indices to delete
        indices_to_delete = []
        for vector_id in ids:
            if vector_id in self.id_to_index:
                indices_to_delete.append(self.id_to_index[vector_id])
        
        # Sort in descending order to delete from end to beginning
        indices_to_delete = sorted(indices_to_delete, reverse=True)
        
        # Delete vectors and update mappings
        for idx in indices_to_delete:
            del self.vectors[idx]
            del self.ids[idx]
        
        # Rebuild id_to_index mapping
        self.id_to_index = {vector_id: i for i, vector_id in enumerate(self.ids)}
    
    def save(self, filepath: str) -> None:
        """Save the vector database to disk.
        
        Args:
            filepath: Path to save the database
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'vectors': self.vectors,
            'ids': self.ids,
            'id_to_index': self.id_to_index,
            'embedding_dim': self.embedding_dim
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str) -> None:
        """Load the vector database from disk.
        
        Args:
            filepath: Path to load the database from
        """
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.vectors = data['vectors']
        self.ids = data['ids']
        self.id_to_index = data['id_to_index']
        self.embedding_dim = data['embedding_dim']
    
    def get_size(self) -> int:
        """Get the number of vectors in the database."""
        return len(self.vectors)
    
    def clear(self) -> None:
        """Clear all vectors from the database."""
        self.vectors = []
        self.ids = []
        self.id_to_index = {}