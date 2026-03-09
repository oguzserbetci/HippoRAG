import os
from typing import List
import json
import argparse
import logging

from hipporag.utils.config_utils import BaseConfig
from src.hipporag import HippoRAG

def main():

    # Prepare datasets and evaluation
    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Marina is bom in Minsk.",
        "Montebello is a part of Rockland County."
    ]

    save_dir = 'outputs/demo_llama'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = "llm3"  # Any OpenAI model name
    llm_base_url = "https://llm3-compute.cms.hu-berlin.de/v1"
    embedding_model_name = "llm5"
    embedding_base_url = "https://llm5-compute.cms.hu-berlin.de/v1/"

    config = BaseConfig(
        dataset='demo',
        embedding_batch_size=128,
        max_new_tokens=4096,
        # vector_db_backend='faiss',
        # vector_db_index_type='Flat',  # 'IVFFlat', 'Flat', 'HNSW'
    )

    # Startup a HippoRAG instance
    hipporag = HippoRAG(
        global_config=config,
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        llm_base_url=llm_base_url,
        embedding_base_url=embedding_base_url,
        embedding_model_name=embedding_model_name,
    )

    # Run indexing
    hipporag.index(docs=docs)

    # Separate Retrieval & QA
    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?"
    ]

    # For Evaluation
    answers = [
        ["Politician"],
        ["By going to the ball."],
        ["Rockland County"]
    ]

    gold_docs = [
        ["George Rankin is a politician."],
        ["Cinderella attended the royal ball.",
         "The prince used the lost glass slipper to search the kingdom.",
         "When the slipper fit perfectly, Cinderella was reunited with the prince."],
        ["Erik Hort's birthplace is Montebello.",
         "Montebello is a part of Rockland County."]
    ]

    print(hipporag.rag_qa(queries=queries,
                                  gold_docs=gold_docs,
                                  gold_answers=answers))

if __name__ == "__main__":
    main()
