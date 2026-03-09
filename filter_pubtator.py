# %%
import polars as pl

# %%
df_concepts = pl.scan_parquet("snomed_ct_concepts.parquet").collect()
df_triples = pl.scan_parquet("snomed_ct_triples.parquet").collect()


#%% 
import json

data = json.load(open('data/bioasq_subsample_202602.json'))
corpus = data['corpus']

# %%
concept_df = pl.read_csv('data/bioconcepts2pubtator3.tsv', separator='\t', new_columns=['pmid', 'type', 'id', 'mentions', 'resources'], schema_overrides=[pl.String, pl.String, pl.String, pl.String,pl.String], quote_char=None)
concept_df = concept_df.filter(pl.col('pmid').cast(pl.String).is_in(list(corpus.keys()))).with_columns(
    pl.col('id'), pl.col('mentions').str.split('|').alias('labels'), pl.lit([]).alias('synonyms')
)
concept_df = concept_df.write_parquet('pubtator3_concepts.parquet')

# %%
relation_df = pl.read_csv('data/relation2pubtator3.tsv', separator='\t', new_columns=['pmid', 'rel', 'subject', 'object'],schema_overrides=[pl.String, pl.String, pl.String, pl.String], quote_char=None)
relation_df = relation_df.filter(pl.col('pmid').cast(pl.String).is_in(list(corpus.keys())))
relation_df = relation_df.with_columns(pl.col('subject').str.split('|').list.get(-1), pl.col('object').str.split('|').list.get(-1))

relation_df = relation_df.write_parquet('pubtator3_relations.parquet')
