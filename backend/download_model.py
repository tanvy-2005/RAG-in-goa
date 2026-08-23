import os
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "intfloat/multilingual-e5-small"
SAVE_PATH = os.path.join(os.path.dirname(__file__), "model_cache")

print(f"Pre-caching model weights to {SAVE_PATH} during build...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)
print("Model caching complete.")