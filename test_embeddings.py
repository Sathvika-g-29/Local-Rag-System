from sentence_transformers import SentenceTransformer
import numpy as np

# 1️⃣ Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2️⃣ Create small knowledge base
documents = [
    "Python is a programming language.",
    "Artificial Intelligence is the future of technology.",
    "The capital of France is Paris.",
    "Machine learning is a subset of AI."
]

# 3️⃣ Convert documents into embeddings
doc_embeddings = model.encode(documents)

# 4️⃣ User query
query = "What is AI?"

# 5️⃣ Convert query into embedding
query_embedding = model.encode(query)

# 6️⃣ Compute cosine similarity manually
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 7️⃣ Compare query with all documents
scores = []

for emb in doc_embeddings:
    score = cosine_similarity(query_embedding, emb)
    scores.append(score)

# 8️⃣ Find best match
best_match_index = np.argmax(scores)

print("Query:", query)
print("Most relevant document:", documents[best_match_index])