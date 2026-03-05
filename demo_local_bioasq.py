import json
from pathlib import Path

import datasets
import numpy as np
import srsly
import torch
from tqdm import tqdm
from dataclasses import asdict

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import QuerySolution


def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def main():
    if Path('data/bioasq_subsample_202602.json').exists():
        data = json.load(open('data/bioasq_subsample_202602.json'))
        corpus = data['corpus']
        queries = data['queries']
        answers = data['answers']
        qrels = data['qrels']
    else:
        with open('data/BioASQ-training14b/trainining14b.json')  as f:
            data = json.load(f)
        
        corpus = {}
        queries = {}
        answers = {}
        qrels = {}
        rel_doc_ids = set()
        for q in tqdm(data['questions']):
            queries[q['id']] = q['body']
            answers[q['id']] = q.get('ideal_answer', [])
            rel_docs = {d.rsplit('/')[-1]: 1 for d in q['documents']}
            qrels[q['id']] = rel_docs
            rel_doc_ids.update(rel_docs.keys())
            corpus.update({doc.rsplit('/')[-1]: doc for doc in q['documents']})

        ds = datasets.load_from_disk('/vol/wbi/wbi/wbi_nlp/datasets/pubmed.dataset/')

        ds = ds.filter(lambda x: x['pmid'] in rel_doc_ids, num_proc=16)
        corpus = dict(zip(ds['pmid'], [f"# {t}\n{a}" for t, a in zip(ds['title'], ds['abstract'])]))
        corpus_ids = set(ds['pmid'])

        print(len(corpus_ids), " corpus documents found for ", len(rel_doc_ids), " relevant documents.")

        json.dump({'corpus': corpus, 'queries': queries, 'answers': answers, 'qrels': qrels}, open('data/bioasq_subsample_202602.json', 'w'), indent=4)

    print("Data is ready:", 'Corpus:',len(corpus), 'Queries:', len(queries), ' with average ', np.mean([len(qrels[qid]) for qid in qrels]), ' relevant documents per query.')
    save_dir = "outputs/bioasq_subsample"  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = "llm3"  # Any OpenAI model name
    llm_base_url = "https://llm3-compute.cms.hu-berlin.de/v1"
    # llm_model_name = "google/medgemma-4b-it"  # Any OpenAI model name
    # embedding_model_name = "Qwen/Qwen3-Embedding-8B"  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    # embedding_model_name = "sentence-transformers/all-mpnet-base-v2"  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    embedding_model_name = "llm5"
    embedding_base_url = 'https://llm5-compute.cms.hu-berlin.de/v1/'

    config = BaseConfig(
        dataset='bio',
        embedding_batch_size=128,
        embedding_base_url=embedding_base_url,
        embedding_model_name=embedding_model_name,
        # vector_db_backend='faiss',
        # vector_db_index_type='Flat',  # 'IVFFlat', 'Flat', 'HNSW'
    )
    
    # Startup a HippoRAG instance
    hipporag = HippoRAG(
        # global_config=BaseConfig(openie_mode='offline', llm_name=llm_model_name, embedding_model_name=embedding_model_name),
        global_config=config,
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        llm_base_url=llm_base_url,
        embedding_model_name=embedding_model_name,
        embedding_base_url=embedding_base_url,
    )
    
    index_corpus = list(corpus.values())
    # Run indexing
    hipporag.index(docs=index_corpus)

    queries = list(queries.values())
    gold_docs = [[corpus[k] for k in qrels[qid].keys() if k in corpus] for qid in qrels]
    gold_answers = [a for a in answers.values()]
    
    name = config.dataset + f'_corpus{len(corpus)}_queries{len(queries)}'
    if not Path(f'{save_dir}/{name}_queries_solutions.json').exists():
        query_solutions, overall_retrieval_result = hipporag.retrieve(queries=queries, gold_docs=gold_docs)
        write_json(f'{save_dir}/{name}_queries_solutions.json', make_json_serializable([asdict(s) for s in query_solutions]))
        write_json(f'{save_dir}/{name}_overall_retrieval_result.json', make_json_serializable(overall_retrieval_result))
    else:
        query_solutions = json.load(open(f'{save_dir}/{name}_queries_solutions.json'))
        query_solutions = [QuerySolution(**qs) for qs in query_solutions]
        overall_retrieval_result = json.load(open(f'{save_dir}/{name}_overall_retrieval_result.json'))

    queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results = hipporag.rag_qa(queries=query_solutions, gold_docs=gold_docs, gold_answers=gold_answers)
    write_json(f'{save_dir}/{name}_qa_results.json', make_json_serializable(overall_qa_results))
    write_json(f'{save_dir}/{name}_all_response_message.json', make_json_serializable(all_response_message))
    write_json(f'{save_dir}/{name}_all_metadata.json', make_json_serializable(all_metadata))

if __name__ == "__main__":
    main()
