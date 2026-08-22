import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_FILE = "multilingual.index"
METADATA_FILE = "multilingual_metadata.jsonl"
MODEL_NAME = "intfloat/multilingual-e5-small"
EVAL_QUERY_COUNT = 100

print("=" * 60)
print("LOADING INDEX AND METADATA")
print("=" * 60)

index = faiss.read_index(INDEX_FILE)
print(f"Index loaded: {index.ntotal:,} vectors")

metadata = []
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            metadata.append(json.loads(line))

print(f"Metadata records loaded: {len(metadata):,}")

print("\nLoading multilingual embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded successfully!")
_ = model.encode(["query: warmup test"], normalize_embeddings=True)
seen_hi_qids = set()
seen_en_qids = set()
hi_queries = []
en_queries = []

for doc in metadata:
    qid = doc.get("query_id")
    q_text = str(doc.get("query", "")).strip()
    lang = doc.get("language")
    is_rel = doc.get("relevant") is True

    if not q_text or not is_rel:
        continue
    if (lang == "hi" or any("\u0900" <= c <= "\u097f" for c in q_text)) and qid not in seen_hi_qids:
        seen_hi_qids.add(qid)
        hi_queries.append({"query_id": qid, "query_text": q_text, "language": "hi"})
    elif lang == "en" and qid not in seen_en_qids:
        seen_en_qids.add(qid)
        en_queries.append({"query_id": qid, "query_text": q_text, "language": "en"})

hi_queries = hi_queries[:EVAL_QUERY_COUNT]
en_queries = en_queries[:EVAL_QUERY_COUNT]

print(f"\nEvaluation Set: {len(hi_queries)} Hindi Queries | {len(en_queries)} English Queries")
def evaluate_retrieval(query_list, mode_name, filter_corpus_lang=None):
    if not query_list:
        print(f"\nSkipping {mode_name}: No queries available.")
        return

    latencies = []
    hits_r5 = 0
    hits_r10 = 0
    reciprocal_ranks = []
    search_k = 50 if filter_corpus_lang else 10

    for item in query_list:
        q_text = item["query_text"]
        target_qid = item["query_id"]

        t0 = time.perf_counter()
        q_emb = model.encode([f"query: {q_text}"], normalize_embeddings=True, convert_to_numpy=True)
        q_emb = q_emb.astype("float32")
        scores, indices = index.search(q_emb, search_k)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)

        retrieved_docs = []
        for idx in indices[0]:
            if 0 <= idx < len(metadata):
                doc = metadata[idx]
                if filter_corpus_lang is None or doc.get("language") == filter_corpus_lang:
                    retrieved_docs.append(doc)

        rr = 0.0
        r5_hit = False
        r10_hit = False

        for rank, doc in enumerate(retrieved_docs[:10], start=1):
            if doc.get("query_id") == target_qid and doc.get("relevant") is True:
                if rr == 0.0:
                    rr = 1.0 / rank
                if rank <= 5:
                    r5_hit = True
                if rank <= 10:
                    r10_hit = True

        reciprocal_ranks.append(rr)
        if r5_hit:
            hits_r5 += 1
        if r10_hit:
            hits_r10 += 1

    total = len(query_list)
    recall5 = (hits_r5 / total) * 100.0 if total > 0 else 0.0
    recall10 = (hits_r10 / total) * 100.0 if total > 0 else 0.0
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    p50 = np.percentile(latencies, 50)
    p70 = np.percentile(latencies, 70)
    p100 = np.percentile(latencies, 100)

    print("\n" + "=" * 60)
    print(f"BENCHMARK: {mode_name}")
    print("=" * 60)
    print(f"Queries Evaluated : {total}")
    print(f"Recall@5           : {recall5:.2f}%")
    print(f"Recall@10          : {recall10:.2f}%")
    print(f"MRR                : {mrr:.4f}")
    print(f"P50 Latency        : {p50:.2f} ms")
    print(f"P70 Latency        : {p70:.2f} ms")
    print(f"P100 Latency       : {p100:.2f} ms")
    print(f"Latency < 200ms    : {'PASSED' if p100 < 200 else 'FAILED'}")
evaluate_retrieval(hi_queries, "Hindi -> Hindi (Monolingual)", filter_corpus_lang="hi")
evaluate_retrieval(en_queries, "English -> English (Monolingual)", filter_corpus_lang="en")
evaluate_retrieval(hi_queries, "Hindi -> English (Cross-Lingual)", filter_corpus_lang="en")
evaluate_retrieval(en_queries, "English -> Hindi (Cross-Lingual)", filter_corpus_lang="hi")
evaluate_retrieval(hi_queries + en_queries, "Overall Multilingual (All Chunks Combined)", filter_corpus_lang=None)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)