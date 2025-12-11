import os
from typing import List
import json
import argparse
import logging

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig


def main():

    docs = [
        # Patients
        "Breast carcinoma shows HER2 negativity and strong ER positivity. PR status was not assessed.",
        "Breast carcinoma is ER positive but PR negative. HER2 testing was not documented.",
        "Endometrial carcinoma demonstrates MSI-high by MLH1 loss. PTEN status was not yet available.",
        "Endometrial carcinoma shows a PTEN mutation with ER positivity. MSI testing was not performed.",
        "Cutaneous melanoma harbors a BRAF V600E mutation. S100 immunostaining is positive.",
        "Cutaneous melanoma expresses SOX10 and also carries a BRAF V600E mutation. S100 was not tested.",
        "Colorectal adenocarcinoma is CK20 and CDX2 positive. RAS testing was not performed.",
        "Colorectal carcinoma harbors a KRAS G12D mutation. Immunohistochemistry was not available.",
        "Glioblastoma shows GFAP positivity and IDH1 wildtype. MGMT status was not reported.",
        "Glioblastoma demonstrates MGMT promoter methylation. IDH status was not documented.",
        "Gastric adenocarcinoma shows HER2 overexpression (3+). TP53 status was not available.",
        "Gastric adenocarcinoma carries a TP53 mutation. HER2 status was not provided.",
        # Articles
        "Elacestrant versus standard endocrine therapy in ER-positive, HER2-negative advanced breast cancer (EMERALD trial). Patients with pretreated ER+/HER2- breast cancer often have poor prognosis, with PR status influencing therapeutic response.",
        "Immune checkpoint blockade in MSI-high endometrial carcinoma with PTEN pathway alterations. MSI-high and PTEN mutations frequently co-occur, and ER positivity may stratify therapeutic response.",
        "BRAF/MEK inhibition in melanoma with melanocytic markers. S100 and SOX10 expression often coincide with BRAF V600E mutations, guiding targeted therapy.",
        "KRAS mutations in colorectal carcinoma. CK20/CDX2 positive tumors often harbor KRAS mutations such as G12D, impacting response to anti-EGFR therapy.",
        "HER2-positive gastric cancers and TP53 mutations. HER2 overexpression frequently coexists with TP53 alterations, with therapeutic implications for trastuzumab-based regimens.",
    ]

    save_dir = "outputs/new_onco_bio"  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = "gpt-5"  # Any OpenAI model name
    embedding_model_name = "text-embedding-3-small"  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    # llm_model_name = "TechxGenus/Mistral-Large-Instruct-2411-AWQ"  # Any OpenAI model name
    # embedding_model_name = 'GritLM/GritLM-7B'  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    # llm_base_url = "https://llm1-compute.cms.hu-berlin.de/v1"

    config = BaseConfig(temperature=1.)
    # Startup a HippoRAG instance
    hipporag = HippoRAG(
        global_config=config,
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        # llm_base_url=llm_base_url,
    )

    # Run indexing
    hipporag.index(docs=docs)

    # Separate Retrieval & QA
    queries = [
        # A
        "Breast carcinoma is HER2 negative and ER positive with PR negativity. The combined biomarker profile suggests aggressive endocrine-resistant disease.",
        # B
        "Endometrial carcinoma is MSI-high with a PTEN mutation and ER positivity. These features align with potential benefit from immunotherapy.",
        # C
        "Melanoma shows BRAF V600E with both S100 and SOX10 positivity. The combined features suggest suitability for BRAF/MEK inhibition.",
        # D
        "Colorectal adenocarcinoma is CK20/CDX2 positive with a KRAS G12D mutation. These biomarkers indicate resistance to anti-EGFR therapy.",
        # E
        "Gastric adenocarcinoma is HER2 positive with a TP53 mutation. This combination has implications for trastuzumab response.",
    ]

    gold_docs = [
        # A
        [
            "Breast carcinoma shows HER2 negativity and strong ER positivity. PR status was not assessed.",
            "Elacestrant versus standard endocrine therapy in ER-positive, HER2-negative advanced breast cancer (EMERALD trial). Patients with pretreated ER+/HER2- breast cancer often have poor prognosis, with PR status influencing therapeutic response.",
        ],
        # B
        [
            "Endometrial carcinoma demonstrates MSI-high by MLH1 loss. PTEN status was not yet available.",
            "Immune checkpoint blockade in MSI-high endometrial carcinoma with PTEN pathway alterations. MSI-high and PTEN mutations frequently co-occur, and ER positivity may stratify therapeutic response.",
        ],
        # C
        [
            "Cutaneous melanoma harbors a BRAF V600E mutation. S100 immunostaining is positive.",
            "BRAF/MEK inhibition in melanoma with melanocytic markers. S100 and SOX10 expression often coincide with BRAF V600E mutations, guiding targeted therapy.",
        ],
        # D
        [
            "Colorectal adenocarcinoma is CK20 and CDX2 positive. RAS testing was not performed.",
            "KRAS mutations in colorectal carcinoma. CK20/CDX2 positive tumors often harbor KRAS mutations such as G12D, impacting response to anti-EGFR therapy.",
        ],
        # E
        [
            "Gastric adenocarcinoma shows HER2 overexpression (3+). TP53 status was not available.",
            "HER2-positive gastric cancers and TP53 mutations. HER2 overexpression frequently coexists with TP53 alterations, with therapeutic implications for trastuzumab-based regimens.",
        ],
    ]

    print(hipporag.retrieve(queries=queries, num_to_retrieve=2, gold_docs=gold_docs))


if __name__ == "__main__":
    main()
