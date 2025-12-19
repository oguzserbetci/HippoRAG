from datasets import load_dataset

# Load just the first example
ds = load_dataset("bigbio/medhop", name="medhop_source", split="train", streaming=True)
entry = next(iter(ds))

for key in ['supports', 'candidates', 'documents', 'context']:
    if key in entry:
        print(f"\found '{key}': {type(entry[key])}")
        if isinstance(entry[key], list) and len(entry[key]) > 0:
            print(f"sample item inside '{key}': {entry[key][0]}")