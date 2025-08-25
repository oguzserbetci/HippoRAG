from typing import List, Tuple, Dict, Any, Optional
import numpy as np


from .base import BaseMetric
from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig




logger = get_logger(__name__)



class RetrievalRecall(BaseMetric):
    
    metric_name: str = "retrieval_recall"
    
    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)
        
    
    def calculate_metric_scores(self, gold_docs: List[List[str]], retrieved_docs: List[List[str]], k_list: List[int] = [1, 5, 10, 20]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates Recall@k for each example and pools results for all queries.

        Args:
            gold_docs (List[List[str]]): List of lists containing the ground truth (relevant documents) for each query.
            retrieved_docs (List[List[str]]): List of lists containing the retrieved documents for each query.
            k_list (List[int]): List of k values to calculate Recall@k for.

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - A pooled dictionary with the averaged Recall@k across all examples.
                - A list of dictionaries with Recall@k for each example.
        """
        k_list = sorted(set(k_list))
        
        example_eval_results = []
        pooled_eval_results = {f"Recall@{k}": 0.0 for k in k_list}
        for example_gold_docs, example_retrieved_docs in zip(gold_docs, retrieved_docs):
            if len(example_retrieved_docs) < k_list[-1]:
                logger.warning(f"Length of retrieved docs ({len(example_retrieved_docs)}) is smaller than largest topk for recall score ({k_list[-1]})")
            
            example_eval_result = {f"Recall@{k}": 0.0 for k in k_list}
  
            # Compute Recall@k for each k
            for k in k_list:
                # Get top-k retrieved documents
                top_k_docs = example_retrieved_docs[:k]
                # Calculate intersection with gold documents
                relevant_retrieved = set(top_k_docs) & set(example_gold_docs)
                # Compute recall
                if example_gold_docs:  # Avoid division by zero
                    example_eval_result[f"Recall@{k}"] = len(relevant_retrieved) / len(set(example_gold_docs))
                else:
                    example_eval_result[f"Recall@{k}"] = 0.0
            
            # Append example results
            example_eval_results.append(example_eval_result)
            
            # Accumulate pooled results
            for k in k_list:
                pooled_eval_results[f"Recall@{k}"] += example_eval_result[f"Recall@{k}"]

        # Average pooled results over all examples
        num_examples = len(gold_docs)
        for k in k_list:
            pooled_eval_results[f"Recall@{k}"] /= num_examples

        # round off to 4 decimal places for pooled results
        pooled_eval_results = {k: round(v, 4) for k, v in pooled_eval_results.items()}
        return pooled_eval_results, example_eval_results


class RetrievalPrecision(BaseMetric):
    
    metric_name: str = "retrieval_precision"
    
    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)
        
    def calculate_metric_scores(self, gold_docs: List[List[str]], retrieved_docs: List[List[str]], k_list: List[int] = [1, 5, 10, 20]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates Precision@k for each example and pools results for all queries.
        """
        k_list = sorted(set(k_list))
        
        example_eval_results = []
        pooled_eval_results = {f"Precision@{k}": 0.0 for k in k_list}
        
        for example_gold_docs, example_retrieved_docs in zip(gold_docs, retrieved_docs):
            if len(example_retrieved_docs) < k_list[-1]:
                logger.warning(f"Length of retrieved docs ({len(example_retrieved_docs)}) is smaller than largest topk for precision score ({k_list[-1]})")
            
            example_eval_result = {f"Precision@{k}": 0.0 for k in k_list}
            
            for k in k_list:
                top_k_docs = example_retrieved_docs[:k]
                relevant_retrieved = set(top_k_docs) & set(example_gold_docs)
                
                if top_k_docs:
                    example_eval_result[f"Precision@{k}"] = len(relevant_retrieved) / len(top_k_docs)
                else:
                    example_eval_result[f"Precision@{k}"] = 0.0
            
            example_eval_results.append(example_eval_result)
            
            for k in k_list:
                pooled_eval_results[f"Precision@{k}"] += example_eval_result[f"Precision@{k}"]

        num_examples = len(gold_docs)
        for k in k_list:
            pooled_eval_results[f"Precision@{k}"] /= num_examples

        pooled_eval_results = {k: round(v, 4) for k, v in pooled_eval_results.items()}
        return pooled_eval_results, example_eval_results


class RetrievalNDCG(BaseMetric):
    
    metric_name: str = "retrieval_ndcg"
    
    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)
        
    def calculate_metric_scores(self, gold_docs: List[List[str]], retrieved_docs: List[List[str]], k_list: List[int] = [1, 5, 10, 20]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates NDCG@k for each example and pools results for all queries.
        """
        k_list = sorted(set(k_list))
        
        example_eval_results = []
        pooled_eval_results = {f"NDCG@{k}": 0.0 for k in k_list}
        
        for example_gold_docs, example_retrieved_docs in zip(gold_docs, retrieved_docs):
            if len(example_retrieved_docs) < k_list[-1]:
                logger.warning(f"Length of retrieved docs ({len(example_retrieved_docs)}) is smaller than largest topk for NDCG score ({k_list[-1]})")
            
            example_eval_result = {f"NDCG@{k}": 0.0 for k in k_list}
            
            for k in k_list:
                top_k_docs = example_retrieved_docs[:k]
                
                # Create relevance scores (1 for relevant, 0 for non-relevant)
                relevance_scores = [1 if doc in example_gold_docs else 0 for doc in top_k_docs]
                
                # Calculate DCG
                dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
                
                # Calculate IDCG (ideal DCG)
                ideal_relevance = sorted([1] * min(len(example_gold_docs), k), reverse=True)
                idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
                
                if idcg > 0:
                    example_eval_result[f"NDCG@{k}"] = dcg / idcg
                else:
                    example_eval_result[f"NDCG@{k}"] = 0.0
            
            example_eval_results.append(example_eval_result)
            
            for k in k_list:
                pooled_eval_results[f"NDCG@{k}"] += example_eval_result[f"NDCG@{k}"]

        num_examples = len(gold_docs)
        for k in k_list:
            pooled_eval_results[f"NDCG@{k}"] /= num_examples

        pooled_eval_results = {k: round(v, 4) for k, v in pooled_eval_results.items()}
        return pooled_eval_results, example_eval_results


class RetrievalMRR(BaseMetric):
    
    metric_name: str = "retrieval_mrr"
    
    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)
        
    def calculate_metric_scores(self, gold_docs: List[List[str]], retrieved_docs: List[List[str]], k_list: List[int] = [1, 5, 10, 20]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates MRR for each example and pools results for all queries.
        MRR is calculated over the entire retrieved document list, not just top-k.
        """
        example_eval_results = []
        pooled_eval_results = {"MRR": 0.0}
        
        for example_gold_docs, example_retrieved_docs in zip(gold_docs, retrieved_docs):
            example_eval_result = {"MRR": 0.0}
            
            # Find the rank of the first relevant document in the entire list
            reciprocal_rank = 0.0
            for i, doc in enumerate(example_retrieved_docs):
                if doc in example_gold_docs:
                    reciprocal_rank = 1.0 / (i + 1)
                    break
            
            example_eval_result["MRR"] = reciprocal_rank
            example_eval_results.append(example_eval_result)
            
            pooled_eval_results["MRR"] += reciprocal_rank

        # Average pooled results over all examples
        num_examples = len(gold_docs)
        pooled_eval_results["MRR"] /= num_examples
        pooled_eval_results["MRR"] = round(pooled_eval_results["MRR"], 4)
        
        return pooled_eval_results, example_eval_results
