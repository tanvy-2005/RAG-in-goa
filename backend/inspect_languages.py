import pandas as pd

FILE = "hinval.parquet"

print("Loading dataset...")

df = pd.read_parquet(
    FILE,
    columns=[
        "source_lang",
        "target_lang",
        "query",
        "Eng_Query",
        "Answer",
        "Eng_Answer",
        "passages"
    ]
)

print("\n================================")
print("LANGUAGE INFORMATION")
print("================================")

print("Source languages:")
print(df["source_lang"].value_counts())

print("\nTarget languages:")
print(df["target_lang"].value_counts())


print("\n================================")
print("EXAMPLE")
print("================================")

row = df.iloc[0]

print("\nHindi query:")
print(row["query"])

print("\nEnglish query:")
print(row["Eng_Query"])

print("\nHindi answer:")
print(row["Answer"])

print("\nEnglish answer:")
print(row["Eng_Answer"])

print("\nPassage keys:")
print(row["passages"].keys())

print("\nNumber of passages:")
print(len(row["passages"]["English_passages"]))

print("\nRelevance:")
print(row["passages"]["is_selected"])