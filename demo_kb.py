import os
from typing import List
import json
import argparse
import logging
import polars as pl
from hipporag.utils.config_utils import BaseConfig

from src.hipporag import HippoRAG


def main():
    save_dir = "outputs/kb"  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    # llm_model_name = "google/medgemma-4b-it"  # Any OpenAI model name
    llm_model_name = 'gpt-4o-mini'  # Any OpenAI model name
    # llm_model_name = 'gpt-5-mini'  # Any OpenAI model name
    embedding_model_name = "Transformers/sentence-transformers/all-mpnet-base-v2"  # Embedding model name (NV-Embed, GritLM or Contriever for now)

    # Startup a HippoRAG instance
    hipporag = HippoRAG(
        global_config=BaseConfig(embedding_batch_size=128, dataset='bio'),
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
    )

    df_concepts = pl.scan_parquet("snomed_ct_concepts.parquet").collect()
    df_concepts = df_concepts.with_columns(pl.col("labels").list.get(0))
    df_triples = pl.scan_parquet("snomed_ct_triples.parquet").collect()
    triples = (
        df_triples.join(df_concepts, right_on="id", left_on="subject", how="left")
        .join(
            df_concepts, right_on="id", left_on="object", how="left", suffix="_object"
        )
        .select(["labels", "rel", "labels_object"])
    )
    hipporag.load_kb(
        entities=df_concepts['labels'].to_list(), triples=triples.rows()
    )
    
    # Prepare datasets and evaluation
    docs = [
        "BCR-ABL1 Fusion has a targeted therapy Dasatinib for B-Lymphoblastic Leukemia/Lymphoma.",
        "BCR-ABL1 Fusion is well-studied and results in constitutive downstream JAK/STAT and PI3K signaling.",
        "Small molecule inhibitors of ABL1, including FDA-approved imatinib, dasatinib, and nilotinib, have had high levels of clinical activity in patients with the BCR-ABL1 fusion.",
        "PIK3CA also known as PI3K.",
        "PIK3CA, the catalytic subunit of PI3-kinase, is frequently mutated in a diverse range of cancers including breast, endometrial and cervical cancers.",
        "The PI3K pathway is an intracellular signal transduction pathway that regulates key cellular processes like growth, proliferation, survival, and metabolism. It starts with PI3K (phosphatidylinositol 3-kinase), which is activated by extracellular signals, leading to the activation of downstream proteins like AKT (also known as Protein Kinase B). The pathway is crucial for many cellular functions, and its dysregulation is frequently linked to diseases like cancer.",
        "The PI3K/AKT/mTOR pathway is an intracellular signaling pathway important in regulating the cell cycle. Therefore, it is directly related to cellular quiescence, proliferation, cancer, and longevity."
    ]

    hipporag.index(docs=docs)

    # Separate Retrieval & QA
    queries = [
        "What molecular alteration results in PI3K signaling?",
        "What therapies are available for PI3K signaling?",
    ]

    # For Evaluation
    answers = [
        ["BCR-ABL1 Fusion"],
        ["Dasatinib"],
    ]

    gold_docs = [
        ["BCR-ABL1 Fusion is well-studied and results in constitutive downstream JAK/STAT and PI3K signaling."],
        ["BCR-ABL1 Fusion is well-studied and results in constitutive downstream JAK/STAT and PI3K signaling.",
         "BCR-ABL1 Fusion has a targeted therapy Dasatinib for B-Lymphoblastic Leukemia/Lymphoma."],
    ]

    print(hipporag.rag_qa(queries=queries,
                        gold_docs=gold_docs,
                        gold_answers=answers))



if __name__ == "__main__":
    main()
