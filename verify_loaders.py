import sys
from data_loaders import load_medhop_corpus, load_medhop_queries, load_medhop_corpus_all

def test_corpus_loader():
    try:
        
        corpus = load_medhop_corpus_all()
        
        
        assert isinstance(corpus, list), f"error  {type(corpus)}"
        assert len(corpus) > 0, "corpus is empty"
        assert isinstance(corpus[0], str), f"error {type(corpus[0])}"
        
        print(f" loaded {len(corpus)}  docs")
        print(f" {corpus[0][:100]}...\n")
        return True
    
    except Exception as e:
        print(f" failed {e}")
        return False

def test_query_loader():
    try:
        queries, gold_answers = load_medhop_queries(split="validation")
        
        assert len(queries) == len(gold_answers), "answers and queieries are not equal"
        assert len(queries) > 0, "empty quiereis"
        
        assert isinstance(queries[0], str), "Q is not a valid string type"
        assert isinstance(gold_answers[0], list), f" errror {type(gold_answers[0])}"
        
        print(f" loaded {len(queries)} quereis")
        print(f"   query: {queries[0]}")
        print(f"   answer:  {gold_answers[0]}\n")
        return True
    
    except Exception as e:
        print(f"errorx: {e}")
        return False

if __name__ == "__main__":

    corpus_ok = test_corpus_loader()
    queries_ok = test_query_loader()
    
    if corpus_ok and queries_ok:
        print("all mworks fine")
        sys.exit(0)
    else:
        print("failed-")
        sys.exit(1)