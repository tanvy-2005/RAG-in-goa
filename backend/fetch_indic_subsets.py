import os
import gc
import json
import re
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
LANG_FILES = {
    "bn": "validation/benval.parquet",
    "ta": "validation/tamval.parquet",
    "te": "validation/telval.parquet",
    "mr": "validation/marval.parquet",
    "gu": "validation/gujval.parquet",
    "as": "validation/asmval.parquet",
    "kn": "validation/kanval.parquet",
    "ml": "validation/malval.parquet",
    "pa": "validation/panval.parquet",
    "or": "validation/orival.parquet",
    "ur": "validation/urdval.parquet",
    "ne": "validation/nepval.parquet",
    "sa": "validation/sanval.parquet"
}

OUTPUT_FILE = "all_indic_docs.jsonl"
QUERIES_PER_LANG = 300  # Samples ~2,500 to 3,000 passages per language (lightweight)

def clean(text):
    return re.sub(r"\s+", " ", str(text)).strip()

print("=" * 60)
print("ZERO-DISK MULTILINGUAL STREAMER (AUTO-CLEANUP)")
print("=" * 60)

total_docs = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for lang, repo_path in LANG_FILES.items():
        print(f"\nProcessing [{lang.upper()}] from '{repo_path}'...")
        local_parquet = None
        try:
            # 1. Download single file
            local_parquet = hf_hub_download(
                repo_id="ai4bharat/MSMARCO-XI",
                filename=repo_path,
                repo_type="dataset"
            )
            
            # 2. Extract only first batch of rows
            parquet_file = pq.ParquetFile(local_parquet)
            count = 0
            
            for batch in parquet_file.iter_batches(batch_size=300, columns=["query_id", "query", "passages"]):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    if count >= QUERIES_PER_LANG:
                        break
                    
                    q_id = row.get("query_id")
                    indic_query = clean(row.get("query", ""))
                    passages = row.get("passages", {})
                    
                    if not isinstance(passages, dict):
                        continue
                    
                    trans_passages = passages.get("Translated_passages", [])
                    flags = passages.get("is_selected", [])
                    
                    for p_idx, (p_text, flag) in enumerate(zip(trans_passages, flags)):
                        clean_p = clean(p_text)
                        if clean_p:
                            doc = {
                                "doc_id": f"{lang}_{q_id}_{p_idx}",
                                "query_id": q_id,
                                "language": lang,
                                "query": indic_query,
                                "text": clean_p,
                                "relevant": bool(flag == 1)
                            }
                            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                            total_docs += 1
                    
                    count += 1
                
                break  # Stop after the first batch to save memory and time
                
            print(f"Extracted {count} queries for {lang.upper()} ({total_docs} total docs).")
            
        except Exception as e:
            print(f"Error processing [{lang.upper()}]: {e}")
            
        finally:
            # 3. Immediately delete cached file to keep disk usage near zero
            if local_parquet and os.path.exists(local_parquet):
                try:
                    os.remove(local_parquet)
                except Exception:
                    pass
            gc.collect()

print("\n" + "=" * 60)
print(f"SUCCESS: Saved {total_docs:,} clean passages to {OUTPUT_FILE} with minimal disk usage!")
print("=" * 60)