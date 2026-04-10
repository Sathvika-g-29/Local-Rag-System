from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# 1️⃣ Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2️⃣ Knowledge base
documents = [
    "Python is a programming language.",
    "Artificial Intelligence is the future of technology.",
    "The capital of France is Paris.",
    "Machine learning is a subset of AI."
]

# 3️⃣ Convert documents to embeddings
doc_embeddings = model.encode(documents).astype('float32')  # FAISS requires float32

# 4️⃣ Create FAISS index
dimension = doc_embeddings.shape[1]  # 384
index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance
index.add(doc_embeddings)  # add all document vectors

# 5️⃣ Query
query = "Tell me about AI"
query_embedding = model.encode([query]).astype('float32')

# 6️⃣ Search top 1
k = 2  # top result
distances, indices = index.search(query_embedding, k)

# 7️⃣ Output
print("Query:", query)
for i in range(k):
    idx = indices[0][i]
    score = distances[0][i]
    print(f"{i+1}. {documents[idx]} (distance: {score:.4f})")