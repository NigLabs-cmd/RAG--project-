from src.ingestion.loader import load_document
import os

# Test with sample.txt
docs = load_document('data/sample.txt')
print(f'Loaded {len(docs)} document(s)')
if docs:
    print(f'First doc has {len(docs[0]["content"])} characters')
    print(f'Metadata: {docs[0]["metadata"]}')
    print(f'\nFirst 200 characters:\n{docs[0]["content"][:200]}...')
