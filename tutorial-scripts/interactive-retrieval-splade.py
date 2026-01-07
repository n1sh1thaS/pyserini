from pyserini.search.lucene import LuceneSearcher
from pyserini.encode import SpladeQueryEncoder

QUERY = 'How to Help Prevent Abdominal Aortic Aneurysms'

encoder = SpladeQueryEncoder('naver/splade-v3', device='cpu')
searcher = LuceneSearcher('indexes/nfcorpus.splade-v3')
hits = searcher.search(QUERY, k=10)

for i in range(10):
    print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.6f}')