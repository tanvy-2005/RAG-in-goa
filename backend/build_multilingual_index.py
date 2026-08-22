import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


INPUT_DOCS = "bilingual_documents.jsonl"
INDEX_OUTPUT = "multilingual.index"
METADATA_OUTPUT = "multilingual_metadata.jsonl"

MODEL_NAME = "intfloat/multilingual-e5-small"

SAMPLE_LIMIT = 60000



print("=" * 50)
print("LOADING MULTILINGUAL EMBEDDING MODEL")
print("=" * 50)

model = SentenceTransformer(MODEL_NAME)

print("Model loaded!")


print("\nReading bilingual documents...")

texts_to_embed = []
metadata_list = []

with open(INPUT_DOCS, "r", encoding="utf-8") as f:

    for line_idx, line in enumerate(f):

        if line_idx >= SAMPLE_LIMIT:
            break

        if not line.strip():
            continue

        doc = json.loads(line)

        raw_text = doc.get("text", "").strip()

        if not raw_text:
            continue

        passage_text = "passage: " + raw_text

        texts_to_embed.append(passage_text)

        metadata_list.append({
            "doc_id": f"{doc.get('language', 'unk')}_{doc.get('query_id', line_idx)}_{line_idx}",
            "query_id": doc.get("query_id"),
            "language": doc.get("language", "unknown"),
            "query": doc.get("query", ""),
            "text": raw_text,
            "relevant": bool(doc.get("relevant", False))
        })


print(f"Documents selected: {len(texts_to_embed):,}")




print("\nGenerating embeddings...")
print("This may take a few minutes.")

embeddings = model.encode(
    texts_to_embed,
    batch_size=128,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)

print("Embeddings generated!")

print("Embedding shape:", embeddings.shape)




print("\nBuilding FAISS index...")

dimension = embeddings.shape[1]

print("Embedding dimension:", dimension)
index = faiss.IndexFlatIP(dimension)

index.add(embeddings.astype("float32"))

print("FAISS index created!")
print("Number of vectors:", index.ntotal)

print("\nSaving FAISS index...")

faiss.write_index(index, INDEX_OUTPUT)

print(f"Index saved: {INDEX_OUTPUT}")

print("\nSaving metadata...")

with open(METADATA_OUTPUT, "w", encoding="utf-8") as f:

    for metadata in metadata_list:
        f.write(
            json.dumps(
                metadata,
                ensure_ascii=False
            ) + "\n"
        )

print(f"Metadata saved: {METADATA_OUTPUT}")

print("\n")
print("=" * 50)
print("MULTILINGUAL INDEX COMPLETE")
print("=" * 50)

print(f"Documents indexed : {len(texts_to_embed):,}")
print(f"FAISS vectors     : {index.ntotal:,}")
print(f"Embedding size    : {dimension}")
print(f"Index file        : {INDEX_OUTPUT}")
print(f"Metadata file     : {METADATA_OUTPUT}")

print("\nNext step:")
print("Test Hindi, English and cross-lingual retrieval.")