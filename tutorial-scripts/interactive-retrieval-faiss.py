from pyserini.search.faiss import FaissSearcher
from pyserini.encode import AutoQueryEncoder

ENCODER_DIR = 'BAAI/bge-base-en-v1.5'
INDEX_DIR = 'indexes/nfcorpus.bge-base-en-v1.5'
QUERY = 'How to Help Prevent Abdominal Aortic Aneurysms'

encoder = AutoQueryEncoder(ENCODER_DIR, device='cpu', pooling='mean', l2_norm=True)
searcher = FaissSearcher(INDEX_DIR, encoder)
hits = searcher.search(QUERY, k=10)

for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.6f}')
