import pandas as pd

print("Loading dataset...")

df = pd.read_parquet("hinval.parquet")

print("Dataset loaded!")

# Get first record
row = df.iloc[0]

print("\n==============================")
print("QUERY")
print("==============================")
print(row["query"])

print("\n==============================")
print("ENGLISH QUERY")
print("==============================")
print(row["Eng_Query"])

print("\n==============================")
print("HINDI ANSWER")
print("==============================")
print(row["Answer"])

# Get passages
passages = row["passages"]

print("\n==============================")
print("PASSAGE KEYS")
print("==============================")
print(passages.keys())

print("\n==============================")
print("NUMBER OF PASSAGES")
print("==============================")
print(len(passages["English_passages"]))

print("\n==============================")
print("RELEVANCE")
print("==============================")
print(passages["is_selected"])

print("\n==============================")
print("PASSAGES")
print("==============================")

for i in range(len(passages["English_passages"])):

    print(f"\n--- Passage {i + 1} ---")

    print("Selected:",
          passages["is_selected"][i])

    print("\nEnglish:")
    print(passages["English_passages"][i])

    print("\nHindi:")
    print(passages["Translated_passages"][i])