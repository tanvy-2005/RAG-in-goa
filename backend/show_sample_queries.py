import json

metadata = []
with open("multilingual_metadata.jsonl", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.strip():
            try:
                metadata.append(json.loads(line))
            except Exception:
                pass

seen_langs = {}

print("=" * 80)
print("SAMPLE QUERIES PER LANGUAGE FROM YOUR DATASET")
print("=" * 80)

for doc in metadata:
    lang = doc.get("language")
    q = str(doc.get("query", "")).strip()
    text = str(doc.get("text", "")).strip()
    
    if lang not in seen_langs and doc.get("relevant") and len(q) > 3:
        seen_langs[lang] = True
        print(f"[{lang.upper()}]")
        print(f"  Query : {q}")
        print(f"  Text  : {text[:120]}...\n")

print("=" * 80)