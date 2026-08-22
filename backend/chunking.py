import json
import re
import os

INPUT_FILE = "documents.jsonl"
MAX_DOCUMENTS = 100_000

def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def fixed_chunks(text, size=500, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += size - overlap

    return chunks

def sentence_chunks(text, max_chars=500):

    sentences = re.split(
        r"(?<=[.!?।])\s+",
        text
    )

    chunks = []
    current = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        if len(current) + len(sentence) <= max_chars:

            if current:
                current += " "

            current += sentence

        else:

            if current:
                chunks.append(current)

            current = sentence

    if current:
        chunks.append(current)

    return chunks

def recursive_chunks(text, max_chars=500):

    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"\n\s*\n", text)

    if len(parts) == 1:
        parts = re.split(
            r"(?<=[.!?।])\s+",
            text
        )

    chunks = []
    current = ""

    for part in parts:

        part = part.strip()

        if not part:
            continue

        if len(current) + len(part) <= max_chars:

            if current:
                current += " "

            current += part

        else:

            if current:
                chunks.append(current)
            if len(part) > max_chars:

                chunks.extend(
                    fixed_chunks(
                        part,
                        size=max_chars,
                        overlap=50
                    )
                )

                current = ""

            else:

                current = part

    if current:
        chunks.append(current)

    return chunks

strategies = {
    "fixed": fixed_chunks,
    "sentence": sentence_chunks,
    "recursive": recursive_chunks
}


files = {
    name: open(
        f"chunks_{name}.jsonl",
        "w",
        encoding="utf-8"
    )
    for name in strategies
}


print("Starting chunking...")
print(f"Maximum documents: {MAX_DOCUMENTS:,}")


count = 0
chunk_counts = {
    name: 0
    for name in strategies
}


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as input_file:

    for line in input_file:

        if count >= MAX_DOCUMENTS:
            break

        document = json.loads(line)

        text = clean_text(
            document["text"]
        )

        if not text:
            continue

        for name, strategy in strategies.items():

            chunks = strategy(text)

            for index, chunk in enumerate(chunks):

                record = {
                    "text": chunk,
                    "language": document["language"],
                    "query_id": document["query_id"],
                    "query_type": document["query_type"],
                    "source": document["source"],
                    "passage_index": document["passage_index"],
                    "relevant": document["relevant"],
                    "chunk_index": index,
                    "strategy": name
                }

                files[name].write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

                chunk_counts[name] += 1

        count += 1

        if count % 10_000 == 0:

            print(
                f"Processed {count:,} documents..."
            )


for file in files.values():
    file.close()


print("\n================================")
print("CHUNKING COMPLETE")
print("================================")

print(
    f"Documents processed: {count:,}"
)

for name, number in chunk_counts.items():

    filename = f"chunks_{name}.jsonl"

    size = (
        os.path.getsize(filename)
        / (1024 * 1024)
    )

    print(
        f"{name}: "
        f"{number:,} chunks | "
        f"{size:.2f} MB"
    )