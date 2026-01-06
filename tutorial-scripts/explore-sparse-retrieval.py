from pyserini.index.lucene import LuceneIndexReader
from pyserini.analysis import Analyzer, get_lucene_analyzer
import json
import numpy as np


DOC_ID = 'MED-4555'
QUERY = 'How to Help Prevent Abdominal Aortic Aneurysms'

# Generate BM25 vector representation for a document
index_reader = LuceneIndexReader('indexes/lucene.nfcorpus')
tf = index_reader.get_document_vector(DOC_ID)
bm25_weights = {term: index_reader.compute_bm25_term_weight(DOC_ID, term, analyzer=None) for term in tf.keys()}
print(json.dumps(bm25_weights, indent=4, sort_keys=True))

# Generate query vector representation
analyzer = Analyzer(get_lucene_analyzer())
query_tokens = analyzer.analyze(QUERY)
multihot_query_weights = {k: 1 for k in query_tokens}

# Compute inner product
terms = set(bm25_weights.keys()).union(set(multihot_query_weights.keys()))
bm25_vector = np.array([bm25_weights.get(term, 0) for term in terms])
query_vector = np.array([multihot_query_weights.get(term, 0) for term
    in terms])
dot_product = np.dot(bm25_vector, query_vector)

print(f'Score: {dot_product}')