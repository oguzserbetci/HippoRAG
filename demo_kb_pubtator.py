import argparse
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import srsly

from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import QuerySolution
from src.hipporag import HippoRAG
from demo_local_bioasq import make_json_serializable, write_json


def main():
    save_dir = "outputs/bioasq_subsample_pubtator"  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = "llm3"  # Any OpenAI model name
    llm_base_url = "https://llm3-compute.cms.hu-berlin.de/v1"
    embedding_model_name = "llm5"
    embedding_base_url = "https://llm5-compute.cms.hu-berlin.de/v1/"

    data = json.load(open("data/bioasq_subsample_202602.json"))
    corpus = data["corpus"]
    queries = data["queries"]
    answers = data["answers"]
    qrels = data["qrels"]

    queries = dict(list(queries.items())[:200])
    qrels = {k: v for k, v in qrels.items() if k in queries}
    subcorpus = {}
    for k, v in qrels.items():
        print(f'Relevant documents {v.keys()}')
        for docid in v.keys():
            if docid in corpus:
                subcorpus[docid] = corpus[docid]
    corpus = subcorpus
    print(f"Loaded {len(corpus)} documents, {len(queries)} queries, and {len(qrels)} qrels. Now starting HippoRAG...")

    config = BaseConfig(
        dataset="bio",
        embedding_batch_size=128,
        embedding_base_url=embedding_base_url,
        embedding_model_name=embedding_model_name,
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
        embedding_model_name=embedding_model_name,
        embedding_base_url=embedding_base_url,
    )
    print("Corpus and queries loaded. Now loading KB and running RAG-QA...")

    df_concepts = pl.read_parquet("pubtator3_concepts.parquet")
    df_concepts = df_concepts.with_columns(
        pl.col("labels").list.unique().list.drop_nulls().list.sort().list.join("|")
    )
    df_concepts = df_concepts.drop_nulls("labels")
    df_triples = pl.read_parquet("pubtator3_relations.parquet")
    df_triples = df_triples.filter(pl.col('pmid').is_in(list(corpus.keys())))
    df_concepts = df_concepts.filter(pl.col('pmid').is_in(list(corpus.keys())))

    print(
        f'Loaded {len(df_concepts)} concepts ({df_concepts["id"].n_unique()} unique) and {len(df_triples)} triples from PubTator3. Now preparing KB for HippoRAG...'
    )

    df_concepts = df_concepts.select(["pmid", "id", "labels"])
    triples = (
        df_triples.join(
            df_concepts,
            right_on=["id", "pmid"],
            left_on=["subject", "pmid"],
            how="left",
        )
        .join(
            df_concepts,
            right_on=["id", "pmid"],
            left_on=["object", "pmid"],
            how="left",
            suffix="_object",
        )
        .select(["labels", "rel", "labels_object"])
    )

    doc_ids = df_triples["pmid"].to_list()

    breakpoint()

    print(
        f"Loading KB with {len(df_concepts)} entities, {len(triples)} triples and {len(doc_ids)} documents..."
    )
    
    hipporag.load_kb(
        entities=df_concepts["labels"].unique().to_list(),
        triples=triples.rows(),
        doc_ids=doc_ids,
        corpus=corpus,
    )

    print(
        f"KB loaded. Starting RAG-QA with {len(queries)} queries, {len(corpus)} corpus documents, and {len(qrels)} qrels."
    )
    queries = list(queries.values())
    gold_docs = [[corpus[k] for k in qrels[qid].keys() if k in corpus] for qid in qrels]
    gold_answers = [a for a in answers.values()]

    name = config.dataset + f"_pubtator_corpus{len(corpus)}_queries{len(queries)}"
    if not Path(f"{save_dir}/{name}_queries_solutions.json").exists():
        query_solutions, overall_retrieval_result = hipporag.retrieve(
            queries=queries, gold_docs=gold_docs
        )
        write_json(
            f"{save_dir}/{name}_queries_solutions.json",
            make_json_serializable([asdict(s) for s in query_solutions]),
        )
        write_json(
            f"{save_dir}/{name}_overall_retrieval_result.json",
            make_json_serializable(overall_retrieval_result),
        )
    else:
        query_solutions = json.load(open(f"{save_dir}/{name}_queries_solutions.json"))
        query_solutions = [QuerySolution(**qs) for qs in query_solutions]
        overall_retrieval_result = json.load(
            open(f"{save_dir}/{name}_overall_retrieval_result.json")
        )

    (
        queries_solutions,
        all_response_message,
        all_metadata,
        overall_retrieval_result,
        overall_qa_results,
    ) = hipporag.rag_qa(
        queries=query_solutions, gold_docs=gold_docs, gold_answers=gold_answers
    )
    write_json(
        f"{save_dir}/{name}_qa_results.json", make_json_serializable(overall_qa_results)
    )
    write_json(
        f"{save_dir}/{name}_all_response_message.json",
        make_json_serializable(all_response_message),
    )
    write_json(
        f"{save_dir}/{name}_all_metadata.json", make_json_serializable(all_metadata)
    )


if __name__ == "__main__":
    main()
