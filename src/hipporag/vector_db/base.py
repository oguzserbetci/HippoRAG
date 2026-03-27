"""Base vector database interface."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class BaseVectorDB(ABC):
    """Abstract base class for vector databases."""
    
    def __init__(self, embedding_dim: int, **kwargs):
        """Initialize vector database.
        
        Args:
            embedding_dim: Dimension of the embeddings
            **kwargs: Backend-specific configuration
        """
        self.embedding_dim = embedding_dim
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to the database.
        
        Args:
            vectors: Array of shape (n_vectors, embedding_dim)
            ids: List of unique identifiers for each vector
        """
        pass
    
    @abstractmethod
    def search(self, query_vectors: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors for each query vector.
        
        Args:
            query_vectors: Array of shape (n_queries, embedding_dim)
            k: Number of nearest neighbors to return
            
        Returns:
            distances: Array of shape (n_queries, k) with distances
            indices: Array of shape (n_queries, k) with indices in original order
        """
        pass
    
    @abstractmethod
    def get_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """Get vector by ID.
        
        Args:
            vector_id: Unique identifier
            
        Returns:
            Vector if found, None otherwise
        """
        pass
    
    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by IDs.
        
        Args:
            ids: List of unique identifiers to delete
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the vector database to disk.
        
        Args:
            filepath: Path to save the database
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load the vector database from disk.
        
        Args:
            filepath: Path to load the database from
        """
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """Get the number of vectors in the database."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors from the database."""
        pass