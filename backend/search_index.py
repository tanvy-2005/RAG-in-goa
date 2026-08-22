import faiss
import json
from sentence_transformers import SentenceTransformer

INDEX_FILE = "sentence.index"
METADATA_FILE = "sentence_metadata.json"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ============================================
# LOAD MODEL
# ============================================

print("Loading model...")

model = SentenceTransformer(MODEL_NAME)

# ============================================
# LOAD FAISS
# ============================================

print("Loading FAISS index...")

index = faiss.read_index(INDEX_FILE)

# ============================================
# LOAD METADATA
# ============================================

print("Loading metadata...")

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)

print(f"Index contains {index.ntotal:,} vectors.")

# ============================================
# QUERY
# ============================================

query = input("\nEnter your question: ")

query_embedding = model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True
)

# Search top 5
scores, indices = index.search(
    query_embedding.astype("float32"),
    5
)

# ============================================
# RESULTS
# ============================================

print("\n" + "=" * 60)
print("TOP 5 RESULTS")
print("=" * 60)

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    document = metadata[idx]

    print(f"\n--- Rank {rank} ---")

    print(f"Score: {score:.4f}")

    print(f"Query ID: {document.get('query_id')}")

    print(f"Language: {document.get('language')}")

    print(f"Relevant: {document.get('relevant')}")

    print("\nText:")

    print(document.get("text", "")[:1000])