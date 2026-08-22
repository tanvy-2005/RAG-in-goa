from sentence_transformers import SentenceTransformer
import numpy as np
import json

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print("Loading model...")

model = SentenceTransformer(MODEL_NAME)

query = "कॉर्पोरेशन क्या है?"

documents = []

# Read first 10,000 sentence chunks
print("Loading chunks...")

with open(
    "chunks_sentence.jsonl",
    "r",
    encoding="utf-8"
) as f:

    for i, line in enumerate(f):

        if i >= 10000:
            break

        documents.append(
            json.loads(line)
        )

texts = [
    doc["text"]
    for doc in documents
]

print(f"Loaded {len(texts):,} chunks.")

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Encoding query...")

query_embedding = model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True
)

# Cosine similarity because vectors are normalized
scores = np.dot(
    embeddings,
    query_embedding[0]
)

# Top 5
top_indices = np.argsort(scores)[::-1][:5]

print("\n================================")
print("TOP 5 RESULTS")
print("================================")

for rank, index in enumerate(top_indices, start=1):

    doc = documents[index]

    print(f"\n--- Rank {rank} ---")

    print(f"Score: {scores[index]:.4f}")

    print(f"Language: {doc['language']}")

    print(f"Query ID: {doc['query_id']}")

    print(f"Relevant: {doc['relevant']}")

    print(f"\nText:")
    print(doc["text"][:1000])