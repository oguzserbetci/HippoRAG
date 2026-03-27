import numpy as np
from tqdm import tqdm
import os
from typing import Union, Optional, List, Dict, Set, Any, Tuple, Literal
import logging
from copy import deepcopy
import pandas as pd

from .utils.misc_utils import compute_mdhash_id, NerRawOutput, TripleRawOutput
from .vector_db import get_vector_db_class

logger = logging.getLogger(__name__)

class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace, vector_db_config=None):
        """
        Initializes the class with necessary configurations and sets up the working directory.

        Parameters:
        embedding_model: The model used for embeddings.
        db_filename: The directory path where data will be stored or retrieved.
        batch_size: The batch size used for processing.
        namespace: A unique identifier for data segregation.
        vector_db_config: Configuration dict for vector database backend.
                         Expected keys: backend, index_type, nlist

        Functionality:
        - Assigns the provided parameters to instance variables.
        - Checks if the directory specified by `db_filename` exists.
          - If not, creates the directory and logs the operation.
        - Constructs the filename for storing data in a parquet file format.
        - Calls the method `_load_data()` to initialize the data loading process.
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(
            db_filename, f"vdb_{self.namespace}.parquet"
        )
        
        # Initialize vector database configuration
        if vector_db_config is None:
            vector_db_config = {"backend": "memory"}
        self.vector_db_config = vector_db_config
        self.vector_db_backend = vector_db_config.get("backend", "memory")
        
        # Vector database will be initialized after we know the embedding dimension
        self.vector_db = None
        
        self._load_data()

    def get_missing_string_hash_ids(self, texts: List[str]):
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  {}

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        return {h: {"hash_id": h, "content": t} for h, t in zip(missing_ids, texts_to_encode)}

    def insert_strings(self, texts: List[str]):
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  # Nothing to insert.

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        logger.info(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist.")

        if not missing_ids:
            return  {}# All records already exist.

        # Prepare the texts to encode from the "content" field.
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)

        self._upsert(missing_ids, texts_to_encode, missing_embeddings)

    def _load_data(self):
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids, self.texts, self.embeddings = df["hash_id"].values.tolist(), df["content"].values.tolist(), df["embedding"].values.tolist()
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t}
                for h, t in zip(self.hash_ids, self.texts)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h  for idx, h in enumerate(self.hash_ids)}
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
            
            # Initialize vector database with embeddings
            if len(self.embeddings) > 0:
                embedding_dim = len(self.embeddings[0])
                self._init_vector_db(embedding_dim)
                
                # Add embeddings to vector database
                embeddings_array = np.array(self.embeddings)
                self.vector_db.add_vectors(embeddings_array, self.hash_ids)
                logger.info(f"Initialized vector database with {len(self.hash_ids)} embeddings")
        else:
            self.hash_ids, self.texts, self.embeddings = [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}
            
    def _init_vector_db(self, embedding_dim):
        """Initialize the vector database with the specified embedding dimension."""
        if self.vector_db is not None:
            return  # Already initialized
            
        vector_db_class = get_vector_db_class(self.vector_db_backend)
        
        # Pass configuration to vector database
        kwargs = {}
        if self.vector_db_backend == "faiss":
            kwargs["index_type"] = self.vector_db_config.get("index_type", "Flat")
            kwargs["nlist"] = self.vector_db_config.get("nlist", 100)
        
        self.vector_db = vector_db_class(embedding_dim, **kwargs)
        
        # Load existing vector database if it exists
        vector_db_path = self.filename.replace(".parquet", f"_vectordb_{self.vector_db_backend}")
        if os.path.exists(vector_db_path) or os.path.exists(vector_db_path + ".index"):
            try:
                self.vector_db.load(vector_db_path)
                logger.info(f"Loaded existing vector database from {vector_db_path}")
            except Exception as e:
                logger.warning(f"Failed to load vector database: {e}. Will rebuild from embeddings.")

    def _save_data(self):
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "embedding": self.embeddings
        })
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {h: {"hash_id": h, "content": t} for h, t, e in zip(self.hash_ids, self.texts, self.embeddings)}
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        
        # Save vector database
        if self.vector_db is not None:
            vector_db_path = self.filename.replace(".parquet", f"_vectordb_{self.vector_db_backend}")
            try:
                self.vector_db.save(vector_db_path)
                logger.debug(f"Saved vector database to {vector_db_path}")
            except Exception as e:
                logger.warning(f"Failed to save vector database: {e}")
        
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings):
        # Initialize vector database if needed
        if self.vector_db is None and len(embeddings) > 0:
            embedding_dim = len(embeddings[0])
            self._init_vector_db(embedding_dim)
        
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)

        # Add to vector database
        if self.vector_db is not None and len(embeddings) > 0:
            embeddings_array = np.array(embeddings)
            self.vector_db.add_vectors(embeddings_array, hash_ids)

        logger.info(f"Saving new records.")
        self._save_data()

    def delete(self, hash_ids):
        indices = []

        for hash in hash_ids:
            indices.append(self.hash_id_to_idx[hash])

        sorted_indices = np.sort(indices)[::-1]

        for idx in sorted_indices:
            self.hash_ids.pop(idx)
            self.texts.pop(idx)
            self.embeddings.pop(idx)

        # Delete from vector database
        if self.vector_db is not None:
            self.vector_db.delete_vectors(hash_ids)

        logger.info(f"Saving record after deletion.")
        self._save_data()

    def get_row(self, hash_id):
        return self.hash_id_to_row[hash_id]

    def get_hash_id(self, text):
        return self.text_to_hash_id[text]

    def get_rows(self, hash_ids, dtype=np.float32):
        if not hash_ids:
            return {}

        results = {id : self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_all_ids(self):
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self):
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self):
        return set(row['content'] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)
    
    def get_embeddings(self, hash_ids, dtype=np.float32) -> list[np.ndarray]:
        if not hash_ids:
            return []

        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], np.ndarray]:
        """Search for k most similar embeddings using vector database.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of similar embeddings to return
            
        Returns:
            hash_ids: List of hash IDs of similar embeddings
            similarities: Array of similarity scores (higher is better)
        """
        if self.vector_db is None:
            # Fallback to original matrix multiplication approach
            if len(self.embeddings) == 0:
                return [], np.array([])
            
            embeddings_array = np.array(self.embeddings)
            
            # Normalize for cosine similarity
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
            embeddings_norm = embeddings_array / (np.linalg.norm(embeddings_array, axis=1, keepdims=True) + 1e-10)
            
            # Compute similarities
            similarities = np.dot(embeddings_norm, query_norm)
            
            # Get top k
            k = min(k, len(similarities))
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            top_k_similarities = similarities[top_k_indices]
            top_k_hash_ids = [self.hash_ids[i] for i in top_k_indices]
            
            return top_k_hash_ids, top_k_similarities
        
        # Use vector database
        similarities, indices = self.vector_db.search(query_embedding.reshape(1, -1), k)
        
        if len(similarities) == 0:
            return [], np.array([])
        
        # Convert indices to hash IDs
        top_k_hash_ids = [self.hash_ids[i] for i in indices[0]]
        top_k_similarities = similarities[0]
        
        return top_k_hash_ids, top_k_similarities