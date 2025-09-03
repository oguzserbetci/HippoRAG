import os
from typing import List
import json
import argparse
import logging

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig


def main():
    # Prepare datasets and evaluation
    docs = [
        # Lung ADC (EGFR/ALK/KRAS/PD-L1 mix)
        "P001 | Lung adenocarcinoma, stage IV, never-smoker; EGFR exon19del; 1L gefitinib → PR 10m; acquired EGFR T790M; 2L osimertinib → PR 14m; PD after 26m.",
        "P002 | Lung adenocarcinoma, stage IIIB; EGFR L858R; lobectomy; adjuvant osimertinib; disease-free at 18m.",
        "P003 | Lung adenocarcinoma, stage IV; ALK rearranged (EML4::ALK); 1L alectinib → durable CR 24m.",
        "P004 | Lung adenocarcinoma, stage IV; KRAS G12C; 1L platinum/pemetrexed → SD 4m; 2L sotorasib → PR 8m.",
        "P005 | Lung adenocarcinoma, stage IV; PD-L1 TPS 80%, TTF-1 positive, EGFR/KRAS/ALK wildtype; 1L pembrolizumab monotherapy → PR 12m.",
        "P006 | Lung adenocarcinoma, stage IV; EGFR exon19del; 1L erlotinib → PD 5m; no T790M; 2L chemo; deceased at 11m.",
        # Lung SCC / TTF-1 negative
        "P007 | Lung squamous cell carcinoma, stage IV; PD-L1 TPS 10%; 1L platinum + pembrolizumab → SD 6m.",
        # CRC liver met (for negative controls / multi-entity hops)
        "P008 | Colorectal adenocarcinoma, liver metastasis; KRAS wildtype; 1L FOLFOX + bevacizumab → PR 9m.",
        # Breast (ER/PR/HER2)
        "P009 | Breast invasive ductal carcinoma, ER+/PR+, HER2-; metastatic; 1L letrozole + CDK4/6 inhibitor → PR 14m.",
        # Lung ADC EGFR, resistance without T790M (rare)
        "P010 | Lung adenocarcinoma, stage IV; EGFR exon19del; 1L gefitinib → PD 8m; C797S detected; switched to chemo; SD 4m.",
        # Lung ADC ALK, sequence lines
        "P011 | Lung adenocarcinoma, stage IV; ALK rearranged; 1L crizotinib → PD 8m; 2L alectinib → PR 12m.",
        # Lung ADC PD-L1 high but smoker (edge)
        "P012 | Lung adenocarcinoma, stage IV, heavy smoker; PD-L1 TPS 75%; 1L pembrolizumab → SD 5m; early immune‑related hepatitis.",
    ]

    save_dir = "outputs/demo"  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = "google/medgemma-4b-it"  # Any OpenAI model name
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"  # Embedding model name (NV-Embed, GritLM or Contriever for now)

    # Startup a HippoRAG instance
    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        llm_base_url="http://localhost:8000/v1"
    )

    # Run indexing
    hipporag.index(docs=docs)

    # Separate Retrieval & QA
    queries = [
        # A) Single-hop feature match (Feature similarity)
        "Find patients most similar to: stage IV lung adenocarcinoma with ALK rearrangement treated with an ALK inhibitor as first line.",
        # B) Multi-hop (Feature → resistance biomarker → next-line therapy)
        "Retrieve patients who started on an EGFR TKI for EGFR exon19del and, after developing T790M, received osimertinib.",
        # C) Exposure + Outcome constraint (therapy response)
        "Return lung adenocarcinoma cases with PD-L1 ≥ 50% treated with pembrolizumab monotherapy achieving ≥ PR for at least 10 months.",
        # D) Hard negative control (mismatch by tumor type)
        "Exclude colorectal or breast cancer; return only lung cases with KRAS G12C who received a KRAS G12C inhibitor.",
        # E) EGFR progression without T790M (contrast case)
        "Find EGFR exon19del lung adenocarcinoma progressing on 1L gefitinib/erlotinib without T790M and NOT receiving osimertinib.",
    ]

    # For Evaluation
    answers = [
        # A
        ["P003", "P011"],  # alectinib/crizotinib→alectinib paths; both ALK+
        # B
        ["P001"],  # explicit T790M → osimertinib sequence
        # C
        ["P005"],  # PD-L1 80% on pembro with PR 12m
        # D
        ["P004"],  # KRAS G12C → sotorasib; lung only
        # E
        [
            "P006",
            "P010",
        ],  # exon19del, progressed; P006: no T790M; P010: C797S; neither got osimertinib as effective 2L
    ]

    gold_docs = [
        # A
        [
            "P003 | Lung adenocarcinoma, stage IV; ALK rearranged (EML4::ALK); 1L alectinib → durable CR 24m.",
            "P011 | Lung adenocarcinoma, stage IV; ALK rearranged; 1L crizotinib → PD 8m; 2L alectinib → PR 12m.",
        ],
        # B
        [
            "P001 | Lung adenocarcinoma, stage IV, never-smoker; EGFR exon19del; 1L gefitinib → PR 10m; acquired EGFR T790M; 2L osimertinib → PR 14m; PD after 26m."
        ],
        # C
        [
            "P005 | Lung adenocarcinoma, stage IV; PD-L1 TPS 80%, TTF-1 positive, EGFR/KRAS/ALK wildtype; 1L pembrolizumab monotherapy → PR 12m."
        ],
        # D
        [
            "P004 | Lung adenocarcinoma, stage IV; KRAS G12C; 1L platinum/pemetrexed → SD 4m; 2L sotorasib → PR 8m."
        ],
        # E
        [
            "P006 | Lung adenocarcinoma, stage IV; EGFR exon19del; 1L erlotinib → PD 5m; no T790M; 2L chemo; deceased at 11m.",
            "P010 | Lung adenocarcinoma, stage IV; EGFR exon19del; 1L gefitinib → PD 8m; C797S detected; switched to chemo; SD 4m.",
        ],
    ]

    print(hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=answers))

    print(hipporag.retrieve(queries=queries, gold_docs=gold_docs))


if __name__ == "__main__":
    main()
