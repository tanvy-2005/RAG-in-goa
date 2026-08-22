import pandas as pd
import re

df = pd.read_parquet("hinval.parquet")

text = df.iloc[0]["passages"]["Translated_passages"][5]

print("BEFORE CLEANING:")
print(text)


def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


cleaned = clean_text(text)

print("\n\nAFTER CLEANING:")
print(cleaned)