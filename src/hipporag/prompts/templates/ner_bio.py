ner_system = """Your task is to extract biomedical named entities from a pubmed article's title and abstract. Consider all entities that the pubmed article might be able to answer. 
Provide a normalized, non-redundant list. Include nested entities when they refer to distinct biological entities such as distinct levels (e.g., organism, tissue, cell type, receptor, molecule).
Respond with a JSON list of entities.
"""

one_shot_ner_paragraph = """# Effect of curare on responses to different putative neurotransmitters in Aplysia neurons.
We have studied the effects of curare on responses resulting from iontophoretic application of several putative neurotransmitters onto Aplysia neurons. These neurons have specific receptors for acetylcholine (ACh), dopamine, octopamine, phenylethanolamine, histamine, gamma-aminobutyric acid (GABA), aspartic acid, and glutamic acid. Each of these substances may on different specific neurons elicit at least three types of response, caused by a fast depolarizing Na+, a fast hyperpolarizing Cl-, or a slow hyperpolarizing K+ conductance increase. All responses resulting from either Na+ or Cl- conductance increases, irrespective of which putative transmitter activated the response, were sensitive to curare. Most were totally blocked by less than or equal to 10-4 M curare. GABA responses were less sensitive and were often only depressed by 10-3 M curare. K+ conductance responses, irrespective of the transmitter, were not curare sensitive. These results are consistent with a model of receptor organization in which one neurotransmitter receptor may be associated with any of at least three ionophores, mediating conductance increase responses to Na+, Cl-, and K+, respectively. In Aplysia nervous tissue, curare appears not to be a specific antagonist for the nicotinic ACh receptor, but rather to be a specific blocking agent for a class of receptor-activated Na+ and Cl- responses."""


one_shot_ner_output = """{"named_entities":
    [ "Curare", "Neurotransmitters", "Aplysia", "Aplysia neurons", "Aplysia nervous tissue", "Acetylcholine", "Nicotinic acetylcholine receptor", "Dopamine", "Octopamine", "Phenylethanolamine", "Histamine", "Gamma-aminobutyric acid", "Aspartic acid", "Glutamic acid", "Neurotransmitter receptor", "Ionophore", "Sodium ion", "Chloride ion", "Potassium ion", "Sodium conductance", "Chloride conductance", "Potassium conductance" ]
}
"""


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": "${passage}"}
]