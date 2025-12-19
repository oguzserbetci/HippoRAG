import os
from hipporag import HippoRAG
from data_loaders import load_medhop_corpus_all, load_medhop_queries
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

hipporag = HippoRAG(
    save_dir="outputs/medhop_bio",
    llm_model_name="gpt-4o-mini",
    embedding_model_name="nvidia/NV-Embed-v2"
)

corpus = load_medhop_corpus_all() 
queries, gold_answers = load_medhop_queries(split="validation")

print("indexing data into HippoRAG")
hipporag.index(docs=corpus) 

#pred_answers = hipporag.rag_qa(queries=queries[:5])