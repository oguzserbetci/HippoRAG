import polars as pl
from owlready2 import *
from owlready2.pymedtermino2 import *
from owlready2.pymedtermino2.umls import *

default_world.set_backend(filename = "pym.sqlite3")
# import_umls("/vol/wbi/wbi_stud/wbi-datasets/UMLS/umls.zip", terminologies = ["SNOMEDCT_US"])
# default_world.save()

PYM = get_ontology("http://PYM/").load()
SNOMEDCT_US = PYM["SNOMEDCT_US"]

concepts = []
triples = []

for concept in SNOMEDCT_US['138875005'].descendant_concepts(include_self = False):
    concepts.append({'id': concept.name, 'labels': concept.label.en, 'synonyms': concept.synonyms.en})
    if hasattr(concept, 'parent'):
        triples.extend([(concept.name, 'has parent', target.name) for target in concept.parents])
    if hasattr(concept, 'children'):
        triples.extend([(concept.name, 'has child', target.name) for target in concept.children])
    if hasattr(concept, 'part_of'):
        triples.extend([(concept.name, 'part of', target.name) for target in concept.part_of])
    if hasattr(concept, 'has_part'):
        triples.extend([(concept.name, 'has part', target.name) for target in concept.has_part])
    

df_concepts = pl.DataFrame(concepts)
df_concepts.write_parquet('snomed_ct_concepts.parquet')
df_triples = pl.DataFrame(triples, schema=['subject', 'rel', 'object'])
df_triples.write_parquet('snomed_ct_triples.parquet')
