from itertools import islice
import os
from typing import List
import json
import argparse
import logging
import srsly
from hipporag import HippoRAG
import polars as pl

from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from hipporag.utils.config_utils import BaseConfig


def main():
    queries = srsly.read_jsonl(
        "/vol/wbi/wbi_stud/wbi-datasets/PMC-Patients/ReCDS_benchmark/queries/dev_queries.jsonl"
    )
    queries = {s["_id"]: s["text"] for s in queries}
    corpus = srsly.read_jsonl(
        "/vol/wbi/wbi_stud/wbi-datasets/PMC-Patients/ReCDS_benchmark/PPR/corpus.jsonl"
    )
    corpus = {s["_id"]: s for s in corpus}

    qrels = pl.read_csv(
        "/vol/wbi/wbi_stud/wbi-datasets/PMC-Patients/ReCDS_benchmark/PPR/qrels_dev.tsv",
        separator="\t",
    )
    qrels = {
        row["query-id"]: row["corpus-id"]
        for row in qrels.group_by("query-id", maintain_order=True)
        .agg(pl.col("corpus-id"))
        .rows(named=True)
    }

    hipporag_queries = []
    hipporag_gold_docs = []
    hipporag_docs = []
    hipporag_doc_ids = set()
    for query_id in qrels:
        hipporag_queries.append(queries[query_id])
        corpus_ids = qrels.get(query_id, [])
        gold_docs = [corpus[corpus_id] for corpus_id in corpus_ids]
        hipporag_doc_ids.update([d_id for d_id in corpus_ids])

        gold_docs = [
            f"# {d['title']}\n{d['text']}" if d["title"] else d["text"]
            for d in gold_docs
        ]
        hipporag_gold_docs.append(gold_docs)
        hipporag_docs.extend(gold_docs)
        hipporag_doc_ids.update(corpus_ids)

    # Uncomment for full corpus
    # hipporag_docs = [
    #     f"# {d['title']}\n{d['text']}" if d["title"] else d["text"]
    #     for d in corpus.values()
    # ]

    print("Data is ready:", 'Corpus:',len(hipporag_docs), 'Queries:', len(hipporag_queries), 'Qrels:', len(hipporag_gold_docs))

    save_dir = "outputs/pmc_subcorpus"  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = "google/medgemma-4b-it"  # Any OpenAI model name
    # embedding_model_name = "Qwen/Qwen3-Embedding-8B"  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"  # Embedding model name (NV-Embed, GritLM or Contriever for now)

    # Startup a HippoRAG instance
    hipporag = HippoRAG(
        # global_config=BaseConfig(openie_mode='offline', llm_name=llm_model_name, embedding_model_name=embedding_model_name),
        global_config=BaseConfig(embedding_batch_size=128),
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        llm_base_url="http://localhost:8000/v1",
    )

    # Run indexing
    hipporag.index(docs=hipporag_docs)

    print(hipporag.retrieve(queries=hipporag_queries, gold_docs=hipporag_gold_docs))


if __name__ == "__main__":
    main()
