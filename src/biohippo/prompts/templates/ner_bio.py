ner_system = """Your task is to extract biomedical named entities from the given paragraph.
You should focus on entities such as genes, immunohistochemistry markers, molecular alterations, TNM staging, histological tumor types, treatments, operations, medications, and progression status.
Respond with a JSON object containing a list of entities.
"""

one_shot_ner_paragraph = """Right upper lobectomy with mediastinal lymph node dissection was performed for a right upper lobe mass. Final pathology: invasive lung adenocarcinoma, acinar predominant. Immunohistochemistry shows TTF-1 positive, p40 negative, and PD-L1 TPS 60%. Next-generation sequencing identifies EGFR exon 19 deletion (E746_A750del). Pathologic stage: pT2aN1M0, AJCC 8th (Stage IIB). Adjuvant targeted therapy with osimertinib 80 mg orally once daily was initiated. At 9 months, surveillance MRI revealed a solitary brain metastasis consistent with disease progression; stereotactic radiosurgery was delivered and osimertinib was continued."""


one_shot_ner_output = """\
{
    "named_entities": [
        "Right upper lobectomy with mediastinal lymph node dissection",
        "invasive lung adenocarcinoma",
        "acinar growth pattern",
        "TTF-1",
        "p40",
        "PD-L1 TPS",
        "EGFR exon 19 deletion (E746_A750del)",
        "pT2a",
        "pN1",
        "pM0",
        "Stage IIB",
        "adjuvant targeted therapy",
        "osimertinib",
        "brain metastasis",
        "stereotactic radiosurgery"
    ]
}
"""


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": "${passage}"},
]
