import faiss
from pyserini.encode import AutoDocumentEncoder, AutoQueryEncoder
import numpy as np

DOC_ID = 'MED-4555'
QUERY = 'How to Help Prevent Abdominal Aortic Aneurysms'
DOC_TEXT = 'Analysis of risk factors for abdominal aortic aneurysm in a cohort of more than 3 million individuals. BACKGROUND: Abdominal aortic aneurysm (AAA) disease is an insidious condition with an 85% chance of death after rupture. Ultrasound screening can reduce mortality, but its use is advocated only for a limited subset of the population at risk. METHODS: We used data from a retrospective cohort of 3.1 million patients who completed a medical and lifestyle questionnaire and were evaluated by ultrasound imaging for the presence of AAA by Life Line Screening in 2003 to 2008. Risk factors associated with AAA were identified using multivariable logistic regression analysis. RESULTS: We observed a positive association with increasing years of smoking and cigarettes smoked and a negative association with smoking cessation. Excess weight was associated with increased risk, whereas exercise and consumption of nuts, vegetables, and fruits were associated with reduced risk. Blacks, Hispanics, and Asians had lower risk of AAA than whites and Native Americans. Well-known risk factors were reaffirmed, including male gender, age, family history, and cardiovascular disease. A predictive scoring system was created that identifies aneurysms more efficiently than current criteria and includes women, nonsmokers, and individuals aged <65 years. Using this model on national statistics of risk factors prevalence, we estimated 1.1 million AAAs in the United States, of which 569,000 are among women, nonsmokers, and individuals aged <65 years. CONCLUSIONS: Smoking cessation and a healthy lifestyle are associated with lower risk of AAA. We estimated that about half of the patients with AAA disease are not eligible for screening under current guidelines. We have created a high-yield screening algorithm that expands the target population for screening by including at-risk individuals not identified with existing screening criteria.'


# Read FAISS index
index = faiss.read_index('indexes/nfcorpus.bge-base-en-v1.5/index')
num_vectors = index.ntotal
for i in range(10):
    vector = index.reconstruct(i)
    print(vector)

# Read docids
docids = []
with open('indexes/nfcorpus.bge-base-en-v1.5/docid', 'r') as f:
    docids = [line.strip() for line in f.readlines()]

# Reconstruct vector for document
v1 = index.reconstruct(docids.index(DOC_ID)) # embedding of document with docid 'MED-4555'

# Confirm that the vectors are identical
doc_encoder = AutoDocumentEncoder('BAAI/bge-base-en-v1.5', device='cpu', pooling='mean', l2_norm=True)
v2 = doc_encoder.encode(DOC_TEXT)[0]
print('Vectors are identical: ', np.linalg.norm(v2 - v1))

# Get query vector
query_encoder = AutoQueryEncoder('BAAI/bge-base-en-v1.5', device='cpu', pooling='mean', l2_norm=True)
query_vector = query_encoder.encode(QUERY)

# Compute query-document score
score = np.dot(query_vector, v2)
print(f'Query-document score: {score}')

