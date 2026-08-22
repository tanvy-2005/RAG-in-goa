import json
import time
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
PARQUET_FILE = "hinval.parquet"

INDEX_FILE = "sentence.index"
METADATA_FILE = "sentence_metadata.json"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

NUM_QUERIES = 100
TOP_K = 5

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

model = SentenceTransformer(MODEL_NAME)

print("Model loaded!")

print("\nLoading FAISS index...")

index = faiss.read_index(INDEX_FILE)

print(f"Index vectors: {index.ntotal:,}")

print("\nLoading metadata...")

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)

print(
    f"Metadata records: {len(metadata):,}"
)

print("\nLoading validation dataset...")

df = pd.read_parquet(
    PARQUET_FILE,
    columns=[
        "query_id",
        "query",
        "passages"
    ]
)

print(
    f"Dataset rows: {len(df):,}"
)
print("\nPreparing evaluation queries...")

evaluation_queries = []

for _, row in df.iterrows():

    query = row["query"]
    query_id = row["query_id"]
    passages = row["passages"]

    if not query:
        continue

    selected = passages.get("is_selected", [])

    relevant_count = sum(
        int(x) for x in selected
    )

    if relevant_count == 0:
        continue

    evaluation_queries.append(
        {
            "query_id": query_id,
            "query": query,
            "relevant_count": relevant_count
        }
    )

    if len(evaluation_queries) >= NUM_QUERIES:
        break


print(
    f"Evaluation queries: "
    f"{len(evaluation_queries)}"
)

recall_values = []
reciprocal_ranks = []
latencies = []

print("\n" + "=" * 60)
print("EVALUATING RETRIEVAL")
print("=" * 60)


for count, item in enumerate(
    evaluation_queries,
    start=1
):

    query_id = item["query_id"]
    query = item["query"]

    start_time = time.perf_counter()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    scores, indices = index.search(
        query_embedding.astype("float32"),
        TOP_K
    )

    elapsed = (
        time.perf_counter() - start_time
    ) * 1000

    latencies.append(elapsed)

    first_relevant_rank = None

    for rank, idx in enumerate(
        indices[0],
        start=1
    ):

        document = metadata[idx]

        document_query_id = str(
            document.get("query_id")
        )

        current_query_id = str(
            query_id
        )

        is_relevant = (
            document_query_id == current_query_id
            and document.get("relevant") is True
        )

        if is_relevant:

            first_relevant_rank = rank
            break

    if first_relevant_rank is not None:

        recall_values.append(1)

        reciprocal_ranks.append(
            1 / first_relevant_rank
        )

    else:

        recall_values.append(0)

        reciprocal_ranks.append(0)

    if count <= 10:

        print(
            f"\n{count}. {query}"
        )

        print(
            f"Query ID: {query_id}"
        )

        print(
            f"First relevant rank: "
            f"{first_relevant_rank}"
        )

        print(
            f"Latency: "
            f"{elapsed:.2f} ms"
        )

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)


if recall_values:

    recall_at_5 = np.mean(
        recall_values
    )

    mrr = np.mean(
        reciprocal_ranks
    )

    avg_latency = np.mean(
        latencies
    )

    print(
        f"\nRecall@5: "
        f"{recall_at_5:.4f}"
    )

    print(
        f"MRR: "
        f"{mrr:.4f}"
    )

    print(
        f"Average query latency: "
        f"{avg_latency:.2f} ms"
    )

    print(
        f"\nQueries evaluated: "
        f"{len(recall_values)}"
    )

else:

    print("\nERROR:")
    print("No queries were successfully evaluated.")


print("\nEvaluation complete!")