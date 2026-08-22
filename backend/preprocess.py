import pyarrow.parquet as pq
import json
import re
import os

INPUT_FILE = "hinval.parquet"
OUTPUT_FILE = "documents.jsonl"

BATCH_SIZE = 1000


def clean_text(text):
    if not text:
        return ""

    text = str(text)

    # Only normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


print("Opening MSMARCO-XI...")

parquet_file = pq.ParquetFile(INPUT_FILE)

print(f"Total rows: {parquet_file.metadata.num_rows:,}")
print(f"Processing in batches of {BATCH_SIZE:,}...")

seen = set()
total_documents = 0
total_rows = 0

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    errors="replace",
    newline="\n"
) as output:

    for batch_number, batch in enumerate(
        parquet_file.iter_batches(batch_size=BATCH_SIZE),
        start=1
    ):

        rows = batch.to_pylist()

        for row in rows:

            passages = row["passages"]

            english_passages = passages["English_passages"]
            hindi_passages = passages["Translated_passages"]
            selected = passages["is_selected"]

            for i in range(len(english_passages)):

                # -------------------------
                # Hindi passage
                # -------------------------

                hindi_text = clean_text(hindi_passages[i])

                if hindi_text:

                    key = ("hi", hindi_text)

                    if key not in seen:

                        seen.add(key)

                        document = {
                            "text": hindi_text,
                            "language": "hi",
                            "query_id": str(row["query_id"]),
                            "query_type": str(row["query_type"]),
                            "source": "Translated_passages",
                            "passage_index": i,
                            "relevant": bool(selected[i])
                        }

                        output.write(
                            json.dumps(
                                document,
                                ensure_ascii=False
                            ) + "\n"
                        )

                        total_documents += 1

                # -------------------------
                # English passage
                # -------------------------

                english_text = clean_text(english_passages[i])

                if english_text:

                    key = ("en", english_text)

                    if key not in seen:

                        seen.add(key)

                        document = {
                            "text": english_text,
                            "language": "en",
                            "query_id": str(row["query_id"]),
                            "query_type": str(row["query_type"]),
                            "source": "English_passages",
                            "passage_index": i,
                            "relevant": bool(selected[i])
                        }

                        output.write(
                            json.dumps(
                                document,
                                ensure_ascii=False
                            ) + "\n"
                        )

                        total_documents += 1

        total_rows += len(rows)

        print(
            f"Batch {batch_number}: "
            f"{total_rows:,}/{parquet_file.metadata.num_rows:,} rows | "
            f"{total_documents:,} documents"
        )


print("\n===================================")
print("PREPROCESSING COMPLETE")
print("===================================")

print(f"Rows processed: {total_rows:,}")
print(f"Unique documents: {total_documents:,}")

if os.path.exists(OUTPUT_FILE):

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)

    print(f"Output file: {OUTPUT_FILE}")
    print(f"Output size: {size_mb:.2f} MB")