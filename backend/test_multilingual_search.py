import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================
# CONFIGURATION
# ============================================

INDEX_FILE = "multilingual.index"
METADATA_FILE = "multilingual_metadata.jsonl"

MODEL_NAME = "intfloat/multilingual-e5-small"

TOP_K = 5


# ============================================
# LOAD MODEL
# ============================================

print("=" * 60)
print("LOADING MULTILINGUAL MODEL")
print("=" * 60)

model = SentenceTransformer(MODEL_NAME)

print("Model loaded!")


# ============================================
# LOAD FAISS INDEX
# ============================================

print("\nLoading FAISS index...")

index = faiss.read_index(INDEX_FILE)

print(f"Index vectors: {index.ntotal:,}")


# ============================================
# LOAD METADATA
# ============================================

print("\nLoading metadata...")

metadata = []

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            metadata.append(json.loads(line))

print(f"Metadata records: {len(metadata):,}")


# ============================================
# SEARCH FUNCTION
# ============================================

def search(query, top_k=5):

    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    # E5 models expect "query:" for user queries
    query_text = "query: " + query

    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    query_embedding = query_embedding.astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx < 0 or idx >= len(metadata):
            continue

        doc = metadata[idx]

        print(f"\n--- Rank {rank} ---")

        print(f"Score: {score:.4f}")

        print(
            f"Language: {doc.get('language', 'unknown')}"
        )

        print(
            f"Query ID: {doc.get('query_id')}"
        )

        print(
            f"Dataset Relevant: {doc.get('relevant', False)}"
        )

        print(
            f"Original Query: {doc.get('query', '')}"
        )

        print("\nText:")

        print(
            doc.get("text", "")
        )


# ============================================
# TEST QUERIES
# ============================================

queries = [

    # Hindi query
    "कॉर्पोरेशन क्या है?",

    # English query
    "what is a corporation?",

    # Hindi query for cross-lingual retrieval
    "निगम क्या होता है?",

    # English query for cross-lingual retrieval
    "what does corporation mean?"
]


# ============================================
# RUN TESTS
# ============================================

for query in queries:

    search(
        query,
        TOP_K
    )


print("\n" + "=" * 60)
print("ALL MULTILINGUAL SEARCH TESTS COMPLETE")
print("=" * 60)