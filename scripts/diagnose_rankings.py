import json
from app.retrieval.coverage_engine import ContrastiveCoverageEngine

with open('data_storage/training_full_corpus/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

refex_chunks = [c for c in chunks if 'Refex' in c.get('document', '')]
q = "Which rating agency rates Refex Industries bank facilities, and what long-term rating did it reaffirm?"
requirements = {'query': q, 'required_facets': [q, 'rating agency bank facilities', 'long-term rating reaffirm']}
selected = ContrastiveCoverageEngine.select_evidence(refex_chunks, requirements, max_chunks=8)
print('Old Refex selection:')
for c in selected:
    print(c.get('page'), repr(c.get('text', '')[:60]))


