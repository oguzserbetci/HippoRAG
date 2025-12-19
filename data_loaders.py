from datasets import load_dataset
from datasets import concatenate_datasets

def load_medhop_corpus():

    print("loading MedHop corpus")
    
    dataset = load_dataset("bigbio/medhop", name="medhop_source", split="train", trust_remote_code=True)
    
    unique_docs = set()
    
    for entry in dataset:
        supports_list = entry.get('supports', [])
        
        for doc in supports_list:
            if isinstance(doc, str) and len(doc) > 50:
                unique_docs.add(doc)
            
    corpus = list(unique_docs)
    
    if len(corpus) == 0:
        raise ValueError("corpuss is empty")
        
    print(f"laoded {len(corpus)} unique documents")
    return corpus



def load_medhop_corpus_all():
    print("loading train + validation corpus")
    
    ds_train = load_dataset("bigbio/medhop", name="medhop_source", split="train", trust_remote_code=True)
    ds_val = load_dataset("bigbio/medhop", name="medhop_source", split="validation", trust_remote_code=True)
    
    combined_dataset = concatenate_datasets([ds_train, ds_val])
    
    unique_docs = set()
    
    for entry in combined_dataset:
        supports_list = entry.get('supports', [])
        for doc in supports_list:
            if isinstance(doc, str) and len(doc) > 50:
                unique_docs.add(doc)
            
    corpus = list(unique_docs)
    print(f"loaded {len(corpus)} docs")
    return corpus

def load_medhop_queries(split="validation"):
    dataset = load_dataset("bigbio/medhop", name="medhop_bigbio_qa", split=split, trust_remote_code=True)
    
    queries = []
    gold_answers = []
    
    for entry in dataset:
        queries.append(entry['question'])
        ans = entry['answer']
        # answers to a list
        if isinstance(ans, str):
            gold_answers.append([ans])
        else:
            gold_answers.append(ans)
            
    return queries, gold_answers