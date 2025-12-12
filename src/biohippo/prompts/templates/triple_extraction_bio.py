from .ner import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.
- Make sure to include attributes of entities as relationships in the triples such as immunohistochemistry results, molecular alterations, staging information, treatments, and progression status.

"""


ner_conditioned_re_frame = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
"""


ner_conditioned_re_input = ner_conditioned_re_frame.format(passage=one_shot_ner_paragraph, named_entity_json=one_shot_ner_output)


ner_conditioned_re_output = """
{
    "triples": [
        ["TTF-1", "has status", "TTF-1 positive"],
        ["p40", "has status", "p40 negative"],
        ["PD-L1 TPS", "has status", "PD-L1 TPS positive"],
        ["EGFR", "has alteration", "EGFR exon 19 deletion (E746_A750del)"],
        ["EGFR exon 19 deletion (E746_A750del)", "drives", "osimertinib"],
        ["invasive lung adenocarcinoma", "is staged as", "pT2a"],
        ["invasive lung adenocarcinoma", "is staged as", "pN1"],
        ["invasive lung adenocarcinoma", "is staged as", "pM0"],
        ["invasive lung adenocarcinoma", "is staged as", "Stage IIB"],
        ["invasive lung adenocarcinoma", "treated with", "adjuvant targeted therapy"],
        ["invasive lung adenocarcinoma", "treated with", "osimertinib"],
        ["brain metastasis", "treated with", "stereotactic radiosurgery"],
        ["invasive lung adenocarcinoma", "progressed to", "brain metastasis"]
    ]
}
"""


prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    {"role": "user", "content": convert_format_to_template(original_string=ner_conditioned_re_frame, placeholder_mapping=None, static_values=None)}
]