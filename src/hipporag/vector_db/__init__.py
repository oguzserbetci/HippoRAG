"""Vector database integration for HippoRAG."""

from .base import BaseVectorDB
from .memory_vector_db import MemoryVectorDB

__all__ = ["BaseVectorDB", "MemoryVectorDB"]

# Try to import optional vector DB backends
try:
    from .faiss_vector_db import FAISSVectorDB
    __all__.append("FAISSVectorDB")
except ImportError:
    FAISSVectorDB = None

def get_vector_db_class(backend_name: str):
    """Get vector database class by name."""
    if backend_name == "memory":
        return MemoryVectorDB
    elif backend_name == "faiss":
        if FAISSVectorDB is None:
            raise ImportError("FAISS backend requires 'faiss-cpu' or 'faiss-gpu' package. Install with: pip install faiss-cpu")
        return FAISSVectorDB
    else:
        raise ValueError(f"Unknown vector database backend: {backend_name}")