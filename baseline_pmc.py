from itertools import islice
import os
import pathlib
import time
import srsly
import polars as pl

from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

def main():
    queries = srsly.read_jsonl(
        "/vol/wbi/wbi_stud/wbi-datasets/PMC-Patients/Cancer_Patients/queries/cancer_dev_queries.jsonl"
    )
    queries = {s["_id"]: s["text"] for s in queries}
    corpus = srsly.read_jsonl(
            "/vol/wbi/wbi_stud/wbi-datasets/PMC-Patients/Cancer_Patients/PPR_cancer/cancer_corpus.jsonl"
        )
    corpus = {s["_id"]: s for s in corpus}

    qrels = pl.read_csv(
        "/vol/wbi/wbi_stud/wbi-datasets/PMC-Patients/Cancer_Patients/PPR_cancer/cancer_qrels_dev.tsv",
        separator="\t",
    )
    qrels = {
        row["query_id"]: {c_id: 1 for c_id in row["corpus_id"]}
        for row in qrels.group_by("query_id").agg(pl.col("corpus_id")).rows(named=True)
    }
    queries = {k: v for k, v in queries.items() if k in qrels}

    subcorpus = {}
    for query_id, docs in qrels.items():
        for doc_id in docs:
            if doc_id in corpus:
                subcorpus[doc_id] = corpus[doc_id]

    # Uncomment for subcorpus
    # corpus = subcorpus
    print("Data is ready:", 'Corpus:',len(corpus), 'Queries:', len(queries), 'Qrels:', len(qrels))

    # model_name_or_path = "Qwen/Qwen3-Embedding-8B"
    # dense_model = models.HuggingFace(
    #     model_path=model_name_or_path,
    #     max_length=2048,
    #     pooling='eos',
    #     normalize=True,
    #     prompts={"query": 'Given a bioclinical text, retrieve relevant passages that are most similar in clinical sense.',
    #              "passage": ''},
    # )
    
    model_name_or_path = "sentence-transformers/all-mpnet-base-v2"
    dense_model = models.SentenceBERT(
        model_name_or_path,
        max_length=512,
        trust_remote_code=True,
    )

    model = DRES(dense_model, batch_size=512, corpus_chunk_size=10000)
    retriever = EvaluateRetrieval(
        model, score_function="cos_sim", k_values=[1, 5, 10, 1000]
    )

    start_time = time.time()
    results = retriever.encode_and_retrieve(
        corpus=corpus,
        queries=queries,
        overwrite=True
    )
    end_time = time.time()

    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    mrr = retriever.evaluate_custom(qrels, results, k_values=retriever.k_values + [len(corpus)], metric='mrr')
    print('Indexed and retrieved in', (end_time - start_time) / 60, 'minutes')
    
    print("Corpus", len(corpus), "Queries", len(queries), "Qrels", len(qrels))
    print("RECALL", recall)
    print("PRECISION", precision)
    print("NDCG", ndcg)
    print("MAP", _map)
    print("MRR", mrr)
    print(f"{mrr['MRR']*100}, {precision['P@10']*100}, {ndcg['NDCG@10']*100}, {recall['Recall@10']*100}")

if __name__ == "__main__":
    main()
