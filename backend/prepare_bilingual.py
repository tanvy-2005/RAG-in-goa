import pandas as pd
import json
import re

INPUT_FILE = "hinval.parquet"
OUTPUT_FILE = "bilingual_documents.jsonl"

print("Loading dataset in batches...")

# Read only the columns we actually need
df = pd.read_parquet(
    INPUT_FILE,
    columns=["query_id", "query", "Eng_Query", "passages"]
)

print(f"Loaded {len(df):,} records.")

count = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for _, row in df.iterrows():

        passages = row["passages"]

        english = passages["English_passages"]
        hindi = passages["Translated_passages"]
        relevance = passages["is_selected"]

        for i in range(len(english)):

            # Clean text
            en_text = re.sub(r"\s+", " ", str(english[i])).strip()
            hi_text = re.sub(r"\s+", " ", str(hindi[i])).strip()

            # English document
            en_doc = {
                "query_id": int(row["query_id"]),
                "language": "en",
                "query": str(row["Eng_Query"]).strip(),
                "text": en_text,
                "relevant": bool(relevance[i])
            }

            # Hindi document
            hi_doc = {
                "query_id": int(row["query_id"]),
                "language": "hi",
                "query": str(row["query"]).strip(),
                "text": hi_text,
                "relevant": bool(relevance[i])
            }

            f.write(json.dumps(en_doc, ensure_ascii=False) + "\n")
            f.write(json.dumps(hi_doc, ensure_ascii=False) + "\n")

            count += 2

            if count % 10000 == 0:
                print(f"Created {count:,} documents...")

print("\n================================")
print("BILINGUAL DATASET COMPLETE")
print("================================")
print(f"Documents created: {count:,}")
print(f"Output: {OUTPUT_FILE}")