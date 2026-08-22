from sentence_transformers import SentenceTransformer
import json
import numpy as np

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print("Loading multilingual embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully!")

# Read first 5 chunks
texts = []

with open(
    "chunks_sentence.jsonl",
    "r",
    encoding="utf-8"
) as f:

    for i, line in enumerate(f):

        if i >= 5:
            break

        document = json.loads(line)

        texts.append(document["text"])


print(f"\nEncoding {len(texts)} sample chunks...")

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embedding successful!")

print("\nEmbedding shape:")
print(embeddings.shape)

print("\nFirst embedding:")
print(embeddings[0][:10])

print("\nEmbedding dimension:")
print(embeddings.shape[1])