import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EXISTING_BILINGUAL = "bilingual_documents.jsonl"  
NEW_INDIC = "all_indic_docs.jsonl"              

INDEX_OUTPUT = "multilingual.index"
METADATA_OUTPUT = "multilingual_metadata.jsonl"
MODEL_NAME = "intfloat/multilingual-e5-small"
MAX_PER_LANG = 3000

print("=" * 60)
print("BUILDING UNIFIED 14-LANGUAGE FAISS INDEX")
print("=" * 60)

model = SentenceTransformer(MODEL_NAME)

texts_to_embed = []
metadata_list = []
lang_counts = {}

def add_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            lang = doc.get("language", "unknown")
            raw_text = doc.get("text", "").strip()

            if not raw_text:
                continue

            if lang_counts.get(lang, 0) >= MAX_PER_LANG:
                continue

            texts_to_embed.append(f"passage: {raw_text}")
            metadata_list.append(doc)
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

print("Collecting passages across all 14 Indic + English languages...")
add_from_file(EXISTING_BILINGUAL)
add_from_file(NEW_INDIC)

print("\nCorpus Distribution:")
for l, c in sorted(lang_counts.items()):
    print(f"  - {l.upper()}: {c:,} passages")

print(f"\nTotal Corpus Chunks: {len(texts_to_embed):,}")
print("Generating Multilingual E5 Embeddings...")

embeddings = model.encode(
    texts_to_embed,
    batch_size=256,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings, dtype=np.float32)
dimension = embeddings.shape[1]

print("\nBuilding FAISS IndexFlatIP...")
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, INDEX_OUTPUT)
with open(METADATA_OUTPUT, "w", encoding="utf-8") as f:
    for meta in metadata_list:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

print(f"\nSUCCESS: Unified index saved to {INDEX_OUTPUT} ({index.ntotal:,} vectors) & {METADATA_OUTPUT}!")