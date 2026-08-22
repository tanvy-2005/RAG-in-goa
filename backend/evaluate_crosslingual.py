import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_FILE = "multilingual.index"
METADATA_FILE = "multilingual_metadata.jsonl"
MODEL_NAME = "intfloat/multilingual-e5-small"
EVAL_SAMPLE_SIZE = 100

print("Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_FILE)
metadata = []
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        metadata.append(json.loads(line))

print(f"Loading embedding model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
seen_qids = set()
eval_set = []
for item in metadata:
    qid = item["query_id"]
    if qid not in seen_qids and item.get("relevant"):
        seen_qids.add(qid)
        eval_set.append(item)
        if len(eval_set) >= EVAL_SAMPLE_SIZE:
            break

def evaluate_mode(queries, mode_label, target_lang=None, top_k=5):
    latencies = []
    hits = 0
    
    for item in queries:
        q_text = item["query"]
        target_qid = item["query_id"]
        
        t0 = time.perf_counter()
        q_emb = model.encode([f"query: {q_text}"], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)
        
        distances, indices = index.search(q_emb, top_k * 2)
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000.0)
        
        retrieved_docs = [metadata[i] for i in indices[0] if i < len(metadata)]
        
        if target_lang:
            retrieved_docs = [d for d in retrieved_docs if d["language"] == target_lang][:top_k]
        else:
            retrieved_docs = retrieved_docs[:top_k]
            
        retrieved_qids = [d["query_id"] for d in retrieved_docs]
        if target_qid in retrieved_qids:
            hits += 1
            
    total = len(queries)
    recall = (hits / total) * 100.0 if total > 0 else 0.0
    p50 = np.percentile(latencies, 50)
    p70 = np.percentile(latencies, 70)
    p100 = np.percentile(latencies, 100)
    
    print(f"\n==========================================")
    print(f"Mode: {mode_label} ({total} queries)")
    print(f"==========================================")
    print(f"Recall@{top_k}: {recall:.2f}%")
    print(f"P50 Latency:  {p50:.2f} ms")
    print(f"P70 Latency:  {p70:.2f} ms")
    print(f"P100 Latency: {p100:.2f} ms")

hi_eval = [q for q in eval_set if q["language"] == "hi"]
en_eval = [q for q in eval_set if q["language"] == "en"]

if hi_eval:
    evaluate_mode(hi_eval, "Hindi -> Hindi (Mono-lingual)", target_lang="hi")
    evaluate_mode(hi_eval, "Hindi -> English (Cross-lingual)", target_lang="en")

if en_eval:
    evaluate_mode(en_eval, "English -> English (Mono-lingual)", target_lang="en")
    evaluate_mode(en_eval, "English -> Hindi (Cross-lingual)", target_lang="hi")

evaluate_mode(eval_set, "Multilingual Combined Benchmark")