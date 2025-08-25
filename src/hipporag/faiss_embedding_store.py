import numpy as np
from tqdm import trange
import os
from typing import Union, Optional, List, Dict, Set, Any, Tuple, Literal
import logging
from copy import deepcopy
import pandas as pd
import faiss
import pickle
from .embedding_store import EmbeddingStore
from .utils.misc_utils import compute_mdhash_id, NerRawOutput, TripleRawOutput

logger = logging.getLogger(__name__)

class FaissEmbeddingStore(EmbeddingStore):
    def __init__(self, embedding_model, db_filename, batch_size, namespace, embedding_dim=None):
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace
        self.embedding_dim = embedding_dim

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(db_filename, f"vdb_{self.namespace}.parquet")
        self.faiss_filename = os.path.join(db_filename, f"faiss_{self.namespace}.index")
        self.id_mapping_filename = os.path.join(db_filename, f"id_mapping_{self.namespace}.pkl")
        
        self.faiss_index = None
        self.faiss_id_to_hash_id = {}  # Maps FAISS internal IDs to hash_ids
        self.hash_id_to_faiss_id = {}  # Maps hash_ids to FAISS internal IDs
        
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.id_mapping_filename):
            self.faiss_index = faiss.read_index(self.faiss_filename)
            self.embedding_dim = self.faiss_index.d
            
            with open(self.id_mapping_filename, 'rb') as f:
                mappings = pickle.load(f)
                self.faiss_id_to_hash_id = mappings.get('faiss_id_to_hash_id', {})
                self.hash_id_to_faiss_id = mappings.get('hash_id_to_faiss_id', {})
                self.texts = mappings.get('texts', [])
                self.hash_ids = list(self.hash_id_to_faiss_id.keys())

            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t}
                for h, t in zip(self.hash_ids, self.texts)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}

            logger.info(f"Loaded {len(self.hash_id_to_faiss_id)} records from {self.filename}")
        else:
            self.hash_ids, self.texts = [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}
            self.hash_id_to_text, self.text_to_hash_id = {}, {}

    def _initialize_faiss_index(self, embeddings=None):
        """Initialize FAISS index with existing embeddings"""
        if self.embedding_dim is None and embeddings is not None:
            self.embedding_dim = embeddings.shape[1]
        
        if self.embedding_dim is None:
            return  # Cannot initialize without knowing dimension
            
        # Create FAISS index (using IndexFlatIP for cosine similarity)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        if embeddings is not None and len(embeddings) > 0:
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings)
            
            # Update ID mappings
            for i, hash_id in enumerate(self.hash_ids):
                self.faiss_id_to_hash_id[i] = hash_id
                self.hash_id_to_faiss_id[hash_id] = i

    def _save_data(self):       
        # Save FAISS index
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, self.faiss_filename)
            
        # Save ID mappings
        with open(self.id_mapping_filename, 'wb') as f:
            pickle.dump({
                'texts': self.texts,
                'faiss_id_to_hash_id': self.faiss_id_to_hash_id,
                'hash_id_to_faiss_id': self.hash_id_to_faiss_id
            }, f)
        
        # Update other mappings
        self.hash_id_to_row = {h: {"hash_id": h, "content": t} for h, t in zip(self.hash_ids, self.texts)}
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings):
        # Update FAISS index
        if self.faiss_index is None:
            embedding_array = np.array(embeddings, dtype=np.float32)
            self.embedding_dim = embedding_array.shape[1]
            self.texts.extend(texts)
            self.hash_ids.extend(hash_ids)
            self._initialize_faiss_index(embedding_array)
        else:
            embedding_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embedding_array)
            
            start_id = self.faiss_index.ntotal
            self.faiss_index.add(embedding_array)
            self.hash_ids.extend(hash_ids)
            self.texts.extend(texts)
            
            # Update ID mappings
            for i, hash_id in enumerate(hash_ids):
                faiss_id = start_id + i
                self.faiss_id_to_hash_id[faiss_id] = hash_id
                self.hash_id_to_faiss_id[hash_id] = faiss_id

        logger.info(f"Saving new records.")
        self._save_data()

    def search_self(self, k: int = 5) -> Dict[str, Tuple[List[str], List[float]]]:
        """
        For each embedding in the database, find its top-k most similar embeddings (excluding itself).

        Returns:
            Dict mapping hash_id to (list of top-k hash_ids, list of scores)
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return {}

        # Get all embeddings from the index in chunks to avoid memory issues
        results = {}
        
        for idx in trange(self.faiss_index.ntotal, desc='Searching self'):
            query_embedding = np.zeros((1, self.embedding_dim), dtype=np.float32)
            self.faiss_index.reconstruct(idx, query_embedding[0])
            scores, indices = self.faiss_index.search(query_embedding, min(k + 1, self.faiss_index.ntotal))
            topk_ids = []
            topk_scores = []
            for score, neighbor_idx in zip(scores[0], indices[0]):
                if neighbor_idx == idx:
                    continue  # skip self
                if neighbor_idx != -1 and neighbor_idx in self.faiss_id_to_hash_id:
                    hash_id = self.faiss_id_to_hash_id[neighbor_idx]
                    topk_ids.append(hash_id)
                    topk_scores.append(float(score))
                if len(topk_ids) == k:
                    break
            query_hash_id = self.faiss_id_to_hash_id[idx]
            results[query_hash_id] = (topk_ids, topk_scores)
        return results

    def search(self, query_ids, query_embeddings: np.ndarray, k: int = 5) -> Dict[str, Tuple[List[str], List[float]]]:
        """
        Perform similarity search using FAISS
        
        Args:
            query_embedding: Query embedding vector
            k: Number of top results to return
            
        Returns:
            List of (hash_id, similarity_score) tuples
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return {}
            
        results = {}
        for query_ind, query_embedding in enumerate(query_embeddings):
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(query_embedding)

            scores, indices = self.faiss_index.search(query_embedding, min(k, self.faiss_index.ntotal))
            topk_ids = []
            topk_inds = []
            topk_scores = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx in self.faiss_id_to_hash_id:
                    hash_id = self.faiss_id_to_hash_id[idx]
                    query_id = query_ids[query_ind]
                    topk_inds.append(idx)
                    topk_ids.append(hash_id)
                    topk_scores.append(float(score))

            results[query_id] = (topk_ids, topk_inds, topk_scores)

        return results
