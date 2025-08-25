from typing import List, Optional, Union
import torch
from tqdm import tqdm
import numpy as np


def retrieve_knn(query_ids: List[str], key_ids: List[str], query_vecs, key_vecs, k=2047, query_batch_size=1000,
                 key_batch_size=10000, vector_db=None):
    """
    Retrieve the top-k nearest neighbors for each query id from the key ids.
    
    Args:
        query_ids: List of query identifiers
        key_ids: List of key identifiers  
        query_vecs: Query vectors (numpy array or torch tensor)
        key_vecs: Key vectors (numpy array or torch tensor)
        k: top-k neighbors to retrieve
        query_batch_size: Batch size for query processing
        key_batch_size: Batch size for key processing  
        vector_db: Optional vector database backend for efficient search
        
    Returns:
        dict: Mapping from query_id to (top_k_key_ids, similarities)
    """
    # If vector database is provided and contains the key vectors, use it
    if vector_db is not None and hasattr(vector_db, 'search') and len(key_ids) > 0:
        return _retrieve_knn_with_vector_db(query_ids, key_ids, query_vecs, key_vecs, k, vector_db)
    
    # Fallback to original matrix multiplication approach
    return _retrieve_knn_original(query_ids, key_ids, query_vecs, key_vecs, k, query_batch_size, key_batch_size)


def _retrieve_knn_with_vector_db(query_ids: List[str], key_ids: List[str], query_vecs, key_vecs, k: int, vector_db):
    """Retrieve KNN using vector database backend."""
    results = {}
    
    # Convert query vectors to numpy if needed
    if torch.is_tensor(query_vecs):
        query_vecs = query_vecs.cpu().numpy()
    
    if isinstance(query_vecs, list):
        query_vecs = np.array(query_vecs)
    
    # Normalize query vectors
    query_vecs = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-10)
    
    # Search for each query
    for i, query_id in enumerate(tqdm(query_ids, desc="Vector DB KNN Search")):
        query_vec = query_vecs[i:i+1]
        
        try:
            # Use vector database search
            similarities, indices = vector_db.search(query_vec.astype(np.float32), k=min(k, len(key_ids)))
            
            if len(similarities) > 0 and len(similarities[0]) > 0:
                # Convert indices to key IDs
                top_k_key_ids = [key_ids[idx] for idx in indices[0] if idx < len(key_ids)]
                top_k_similarities = similarities[0][:len(top_k_key_ids)]
                
                results[query_id] = (top_k_key_ids, top_k_similarities.tolist())
            else:
                results[query_id] = ([], [])
                
        except Exception as e:
            # Fallback to original method for this query
            print(f"Vector DB search failed for query {query_id}: {e}")
            single_query_result = _retrieve_knn_original([query_id], key_ids, query_vecs[i:i+1], key_vecs, k, 1, 10000)
            results.update(single_query_result)
    
    return results


def _retrieve_knn_original(query_ids: List[str], key_ids: List[str], query_vecs, key_vecs, k=2047, query_batch_size=1000,
                          key_batch_size=10000):
    """
    Original retrieve_knn implementation using matrix multiplication.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(key_vecs) == 0: return {}

    query_vecs = torch.tensor(query_vecs, dtype=torch.float32)
    query_vecs = torch.nn.functional.normalize(query_vecs, dim=1)

    key_vecs = torch.tensor(key_vecs, dtype=torch.float32)
    key_vecs = torch.nn.functional.normalize(key_vecs, dim=1)

    results = {}

    def get_batches(vecs, batch_size):
        for i in range(0, len(vecs), batch_size):
            yield vecs[i:i + batch_size], i

    for query_batch, query_batch_start_idx in tqdm(
            get_batches(vecs=query_vecs, batch_size=query_batch_size),
            total=(len(query_vecs) + query_batch_size - 1) // query_batch_size,  # Calculate total batches
            desc="KNN for Queries"
    ):
        query_batch = query_batch.clone().detach()
        query_batch = query_batch.to(device)

        batch_topk_sim_scores = []
        batch_topk_indices = []

        offset_keys = 0

        for key_batch, key_batch_start_idx in get_batches(vecs=key_vecs, batch_size=key_batch_size):
            key_batch = key_batch.to(device)
            actual_key_batch_size = key_batch.size(0)

            similarity = torch.mm(query_batch, key_batch.T)

            topk_sim_scores, topk_indices = torch.topk(similarity, min(k, actual_key_batch_size), dim=1, largest=True,
                                                       sorted=True)

            topk_indices += offset_keys

            batch_topk_sim_scores.append(topk_sim_scores)
            batch_topk_indices.append(topk_indices)

            del similarity
            key_batch = key_batch.cpu()
            torch.cuda.empty_cache()

            offset_keys += actual_key_batch_size
        # end for each kb batch

        batch_topk_sim_scores = torch.cat(batch_topk_sim_scores, dim=1)
        batch_topk_indices = torch.cat(batch_topk_indices, dim=1)

        final_topk_sim_scores, final_topk_indices = torch.topk(batch_topk_sim_scores,
                                                               min(k, batch_topk_sim_scores.size(1)), dim=1,
                                                               largest=True, sorted=True)
        final_topk_indices = final_topk_indices.cpu()
        final_topk_sim_scores = final_topk_sim_scores.cpu()

        for i in range(final_topk_indices.size(0)):
            query_relative_idx = query_batch_start_idx + i
            query_idx = query_ids[query_relative_idx]

            final_topk_indices_i = final_topk_indices[i]
            final_topk_sim_scores_i = final_topk_sim_scores[i]

            query_to_topk_key_relative_ids = batch_topk_indices[i][final_topk_indices_i]
            query_to_topk_key_ids = [key_ids[idx] for idx in query_to_topk_key_relative_ids.cpu().numpy()]
            results[query_idx] = (query_to_topk_key_ids, final_topk_sim_scores_i.numpy().tolist())

        query_batch = query_batch.cpu()
        torch.cuda.empty_cache()
    # end for each query batch

    return results