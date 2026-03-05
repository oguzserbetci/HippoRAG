from .ner_bio import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

"""


ner_conditioned_re_frame = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
"""


ner_conditioned_re_input = ner_conditioned_re_frame.format(passage=one_shot_ner_paragraph, named_entity_json=one_shot_ner_output)


ner_conditioned_re_output = """{"triples": [
        ["Aplysia neurons", "has_receptor_for", "Acetylcholine"],
        ["Aplysia neurons", "has_receptor_for", "Dopamine"],
        ["Aplysia neurons", "has_receptor_for", "Octopamine"],
        ["Aplysia neurons", "has_receptor_for", "Phenylethanolamine"],
        ["Aplysia neurons", "has_receptor_for", "Histamine"],
        ["Aplysia neurons", "has_receptor_for", "Gamma-aminobutyric acid"],
        ["Aplysia neurons", "has_receptor_for", "Aspartic acid"],
        ["Aplysia neurons", "has_receptor_for", "Glutamic acid"],

        ["Curare", "tested_on", "Aplysia neurons"],
        ["Curare", "blocks", "Sodium conductance"],
        ["Curare", "blocks", "Chloride conductance"],
        ["Curare", "does_not_block", "Potassium conductance"],

        ["Gamma-aminobutyric acid", "elicits_response_via", "Sodium conductance"],
        ["Gamma-aminobutyric acid", "elicits_response_via", "Chloride conductance"],
        ["Gamma-aminobutyric acid", "elicits_response_via", "Potassium conductance"],

        ["Acetylcholine", "elicits_response_via", "Sodium conductance"],
        ["Acetylcholine", "elicits_response_via", "Chloride conductance"],
        ["Acetylcholine", "elicits_response_via", "Potassium conductance"],

        ["Dopamine", "elicits_response_via", "Sodium conductance"],
        ["Dopamine", "elicits_response_via", "Chloride conductance"],
        ["Dopamine", "elicits_response_via", "Potassium conductance"],

        ["Octopamine", "elicits_response_via", "Sodium conductance"],
        ["Octopamine", "elicits_response_via", "Chloride conductance"],
        ["Octopamine", "elicits_response_via", "Potassium conductance"],

        ["Phenylethanolamine", "elicits_response_via", "Sodium conductance"],
        ["Phenylethanolamine", "elicits_response_via", "Chloride conductance"],
        ["Phenylethanolamine", "elicits_response_via", "Potassium conductance"],

        ["Histamine", "elicits_response_via", "Sodium conductance"],
        ["Histamine", "elicits_response_via", "Chloride conductance"],
        ["Histamine", "elicits_response_via", "Potassium conductance"],

        ["Aspartic acid", "elicits_response_via", "Sodium conductance"],
        ["Aspartic acid", "elicits_response_via", "Chloride conductance"],
        ["Aspartic acid", "elicits_response_via", "Potassium conductance"],

        ["Glutamic acid", "elicits_response_via", "Sodium conductance"],
        ["Glutamic acid", "elicits_response_via", "Chloride conductance"],
        ["Glutamic acid", "elicits_response_via", "Potassium conductance"],

        ["Sodium conductance", "mediated_by", "Sodium ion"],
        ["Chloride conductance", "mediated_by", "Chloride ion"],
        ["Potassium conductance", "mediated_by", "Potassium ion"],

        ["Neurotransmitter receptor", "associated_with", "Ionophore"],
        ["Ionophore", "mediates_conductance_increase", "Sodium conductance"],
        ["Ionophore", "mediates_conductance_increase", "Chloride conductance"],
        ["Ionophore", "mediates_conductance_increase", "Potassium conductance"],

        ["Curare", "is_not_specific_antagonist_of", "Nicotinic acetylcholine receptor"],
        ["Curare", "acts_as_blocking_agent_for", "Sodium conductance"],
        ["Curare", "acts_as_blocking_agent_for", "Chloride conductance"],
        ["Curare", "observed_in", "Aplysia nervous tissue"]
    ]
}
"""


prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    {"role": "user", "content": convert_format_to_template(original_string=ner_conditioned_re_frame, placeholder_mapping=None, static_values=None)}
]