import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
INPUT_FILE = "chunks_sentence.jsonl"

INDEX_FILE = "sentence.index"
METADATA_FILE = "sentence_metadata.json"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

BATCH_SIZE = 64

print("=" * 50)
print("LOADING EMBEDDING MODEL")
print("=" * 50)

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully!")

print("\n" + "=" * 50)
print("READING CHUNKS")
print("=" * 50)

texts = []
metadata = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in tqdm(f, desc="Reading chunks"):

        document = json.loads(line)

        text = document.get("text", "").strip()

        if not text:
            continue

        texts.append(text)
        meta = {
            key: value
            for key, value in document.items()
            if key != "text"
        }
        meta["text"] = text

        metadata.append(meta)

print(f"\nTotal chunks: {len(texts):,}")

print("\n" + "=" * 50)
print("CREATING EMBEDDINGS")
print("=" * 50)

all_embeddings = []

for start in tqdm(
    range(0, len(texts), BATCH_SIZE),
    desc="Embedding"
):

    batch = texts[start:start + BATCH_SIZE]

    embeddings = model.encode(
        batch,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    all_embeddings.append(
        embeddings.astype("float32")
    )

embeddings = np.vstack(all_embeddings)

print("\nEmbedding matrix shape:")
print(embeddings.shape)


print("\n" + "=" * 50)
print("BUILDING FAISS INDEX")
print("=" * 50)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"FAISS vectors: {index.ntotal:,}")
print(f"Vector dimension: {dimension}")


print("\n" + "=" * 50)
print("SAVING INDEX")
print("=" * 50)

faiss.write_index(index, INDEX_FILE)

print(f"Saved: {INDEX_FILE}")



with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        ensure_ascii=False
    )

print(f"Saved: {METADATA_FILE}")

print("\n" + "=" * 50)
print("INDEX BUILD COMPLETE")
print("=" * 50)