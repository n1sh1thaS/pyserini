from pyserini.index.lucene import LuceneIndexReader
from pyserini.analysis import Analyzer, get_lucene_analyzer
from pyserini.search.lucene import LuceneSearcher
import numpy as np
import json

DOC_ID = '7186459'
QUERY = 'what is operating system misconfiguration'

# Generate BM25 vector representation for a document
index_reader = LuceneIndexReader('pyserini/indexes/lucene-index-msmarco-passage')
tf = index_reader.get_document_vector(DOC_ID)
bm25_weights = {term: index_reader.compute_bm25_term_weight(DOC_ID, term, analyzer=None) for term in tf.keys()}

print(json.dumps(bm25_weights, indent=4, sort_keys=True))


# Generate query vector representation
analyzer = Analyzer(get_lucene_analyzer())
query_tokens = analyzer.analyze(QUERY)
multihot_query = {k: 1 for k in query_tokens}


# Compute inner product
terms = set(bm25_weights.keys()).union(set(multihot_query.keys()))

bm25_vector = np.array([bm25_weights.get(term, 0) for term in terms])
query_vector = np.array([multihot_query.get(term, 0) for term in terms])

dot_product = np.dot(bm25_vector, query_vector)

print(f'Score for Document {DOC_ID} and Query "{QUERY}": {dot_product}')


# Use LuceneSearcher to verify
searcher = LuceneSearcher('pyserini/indexes/lucene-index-msmarco-passage')
hits = searcher.search(QUERY)

for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')



