import os
import time
import json
import re
import numpy as np
import faiss
import requests
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel

# ============================================================
# 1. OPTIMIZED RUNTIME CONFIGURATION
# ============================================================
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "sk_8x94884j_MrW1uKlHhyOVd3Qf4tAhxopU")
INDEX_FILE = os.path.join(os.path.dirname(__file__), "multilingual.index")
METADATA_FILE = os.path.join(os.path.dirname(__file__), "multilingual_metadata.jsonl")
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
FALLBACK_MODEL_NAME = "intfloat/multilingual-e5-small"

app = FastAPI(
    title="Voice-Enabled Multilingual Indic RAG Harness",
    description="True Neural Indic Vector Search with Pre-cached E5 Embeddings",
    version="27.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 2. LOAD FAISS INDEX, SEEK TABLE & PRE-CACHED E5
# ============================================================
print("=" * 60)
print("INITIALIZING MULTILINGUAL INDIC NEURAL RAG ENGINE")
print("=" * 60)

if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
    INDEX_FILE = "multilingual.index"
    METADATA_FILE = "multilingual_metadata.jsonl"
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        raise FileNotFoundError("multilingual.index or multilingual_metadata.jsonl missing.")

print(f"Loading FAISS Index from {INDEX_FILE}...")
index = faiss.read_index(INDEX_FILE)
total_vectors = index.ntotal
print(f"Total Vectors Indexed: {total_vectors:,}")

print("Building disk seek map...")
doc_offsets = []
with open(METADATA_FILE, "rb") as f:
    offset = 0
    for line in f:
        doc_offsets.append(offset)
        offset += len(line)

def get_metadata_by_id(doc_idx: int) -> dict:
    if 0 <= doc_idx < len(doc_offsets):
        with open(METADATA_FILE, "rb") as f:
            f.seek(doc_offsets[doc_idx])
            line = f.readline().decode("utf-8", errors="ignore")
            if line.strip():
                try:
                    return json.loads(line)
                except Exception:
                    return {}
    return {}

model_source = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else FALLBACK_MODEL_NAME
print(f"Loading Pre-cached Neural Model from: {model_source}...")
tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
embed_model = AutoModel.from_pretrained(model_source, low_cpu_mem_usage=True)
embed_model.eval()

def encode_query(query_text: str) -> np.ndarray:
    """Authentic cross-lingual vector alignment."""
    formatted_query = f"query: {query_text.strip()}"
    inputs = tokenizer(formatted_query, return_tensors="pt", max_length=128, padding=True, truncation=True)
    with torch.inference_mode():
        outputs = embed_model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        return normalized.cpu().numpy().astype("float32")

# Warmup run
_ = encode_query("warmup")
print("Neural Engine Initialized and Warmup Complete!\n")


# ============================================================
# 3. INDIC SCRIPT CLASSIFIER
# ============================================================
LANG_ALIASES = {
    "od": "or", "ori": "or", "odia": "or", "oriya": "or",
    "tam": "ta", "tamil": "ta", "tel": "te", "telugu": "te",
    "kan": "kn", "kannada": "kn", "mal": "ml", "malayalam": "ml",
    "ben": "bn", "bengali": "bn", "bangla": "bn", "asm": "as",
    "assamese": "as", "asamiya": "as", "guj": "gu", "gujarati": "gu",
    "pan": "pa", "punjabi": "pa", "urd": "ur", "urdu": "ur",
    "hin": "hi", "hindi": "hi", "mar": "mr", "marathi": "mr",
    "nep": "ne", "nepali": "ne", "san": "sa", "sanskrit": "sa",
    "eng": "en", "english": "en"
}

SARVAM_BCP47_MAP = {
    "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN",
    "ml": "ml-IN", "mr": "mr-IN", "gu": "gu-IN", "kn": "kn-IN",
    "pa": "pa-IN", "or": "od-IN", "as": "as-IN", "ur": "ur-IN",
    "ne": "ne-IN", "sa": "sa-IN", "en": "en-IN"
}

def normalize_lang_code(code: Optional[str]) -> Optional[str]:
    if not code: return None
    c = code.strip().lower().split("-")[0].split("_")[0]
    return LANG_ALIASES.get(c, c)

def classify_indic_language(text: str, hint_lang: Optional[str] = None) -> str:
    norm_hint = normalize_lang_code(hint_lang)
    if norm_hint and norm_hint not in ["auto", "unknown", ""]:
        return norm_hint

    cleaned = re.sub(r"[।॥?!,.:;\"'()\-—]", " ", text).strip()
    words = set(cleaned.split())
    txt_blob = f" {cleaned} "

    if re.search(r"[\u0B00-\u0B7F]", cleaned): return "or"
    if re.search(r"[\u0B80-\u0BFF]", cleaned): return "ta"
    if re.search(r"[\u0C00-\u0C7F]", cleaned): return "te"
    if re.search(r"[\u0C80-\u0CFF]", cleaned): return "kn"
    if re.search(r"[\u0D00-\u0D7F]", cleaned): return "ml"
    if re.search(r"[\u0A80-\u0AFF]", cleaned): return "gu"
    if re.search(r"[\u0A00-\u0A7F]", cleaned): return "pa"
    if re.search(r"[\u0600-\u06FF]", cleaned): return "ur"

    if re.search(r"[\u0980-\u09FF]", cleaned):
        if any(c in cleaned for c in ["\u09F0", "\u09F1", "ৰ", "ৱ"]): return "as"
        if any(w in words for w in {"কি", "কৰ্পোৰেচন", "হৈছে", "আছে"}): return "as"
        return "bn"

    if re.search(r"[\u0900-\u097F]", cleaned):
        if "\u0933" in cleaned or "ळ" in cleaned: return "mr"
        if any(f" {m} " in txt_blob for m in ["काय", "म्हणजे", "कसा", "आहे", "नाही"]): return "mr"
        if "ः" in cleaned or any(f" {s} " in txt_blob for s in ["किमिति", "अस्ति", "भवति"]): return "sa"
        if any(p in cleaned for p in ["के हो", "हो निगम", "भनेको"]): return "ne"
        return "hi"

    return "en"


# ============================================================
# 4. TRUE VECTOR RETRIEVAL (RANK ORDERED)
# ============================================================
def retrieve_passages(query: str, top_k: int = 3, target_lang: Optional[str] = None):
    t0 = time.perf_counter()
    effective_lang = classify_indic_language(query, target_lang)

    q_emb = encode_query(query)
    scores, indices = index.search(q_emb, min(100, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < total_vectors:
            doc = get_metadata_by_id(int(idx))
            if not doc:
                continue
            
            doc_lang = str(doc.get("language", "")).strip().lower()
            if doc_lang == effective_lang:
                results.append({
                    "score": round(float(score), 4),
                    "language": doc_lang,
                    "query_id": doc.get("query_id"),
                    "text": doc.get("text", "")
                })
                if len(results) >= top_k:
                    break

    # Fallback to nearest semantic matches regardless of strict language tag
    if not results:
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < total_vectors:
                doc = get_metadata_by_id(int(idx))
                if doc:
                    results.append({
                        "score": round(float(score), 4),
                        "language": str(doc.get("language", effective_lang)),
                        "query_id": doc.get("query_id"),
                        "text": doc.get("text", "")
                    })
                    if len(results) >= top_k:
                        break

    ret_time = (time.perf_counter() - t0) * 1000.0
    print(f"[Neural RAG] Lang: {effective_lang.upper()} | Matches: {len(results)} | FAISS Latency: {ret_time:.2f}ms")
    return results, ret_time, effective_lang


# ============================================================
# 5. API ENDPOINTS
# ============================================================
class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = "auto"

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Voice-Enabled Multilingual Indic RAG Harness",
        "vectors_indexed": index.ntotal,
        "docs_url": "/docs"
    }

@app.post("/api/ask")
def process_text_query(req: QueryRequest):
    t_start = time.perf_counter()
    passages, ret_time, matched_lang = retrieve_passages(req.query, top_k=3, target_lang=req.language)
    total_time = (time.perf_counter() - t_start) * 1000.0

    if not passages or passages[0]["score"] < 0.70:
        return {
            "query": req.query,
            "language": matched_lang,
            "answer": "The query is outside the verified dataset knowledge base.",
            "grounded": False,
            "passages": [],
            "latency_ms": round(total_time, 2),
            "retrieval_ms": round(ret_time, 2),
            "passed_target_200ms": bool(total_time < 200.0)
        }

    return {
        "query": req.query,
        "language": matched_lang,
        "answer": passages[0]["text"],
        "grounded": True,
        "passages": passages,
        "latency_ms": round(total_time, 2),
        "retrieval_ms": round(ret_time, 2),
        "passed_target_200ms": bool(total_time < 200.0)
    }

@app.post("/api/voice-ask")
async def process_voice_query(
    file: UploadFile = File(...),
    language: Optional[str] = Form("auto")
):
    t_start = time.perf_counter()
    audio_data = await file.read()
    orig_filename = file.filename or ""

    resolved_hint = None
    if language and language.strip().lower() not in ["auto", "unknown", ""]:
        resolved_hint = normalize_lang_code(language)
    else:
        resolved_hint = normalize_lang_code(orig_filename)

    sarvam_lang_code = SARVAM_BCP47_MAP.get(resolved_hint, "unknown") if resolved_hint else "unknown"
    transcript = ""

    try:
        url = "https://api.sarvam.ai/speech-to-text"
        headers = {"api-subscription-key": SARVAM_API_KEY}
        filename = "audio.wav" if "wav" in orig_filename.lower() else "audio.mp3"
        files = {"file": (filename, audio_data, "audio/mpeg" if "mp3" in filename else "audio/wav")}
        data = {"model": "saaras:v3", "mode": "transcribe", "language_code": sarvam_lang_code}
        
        res = requests.post(url, headers=headers, files=files, data=data, timeout=15)
        if res.status_code == 200:
            transcript = res.json().get("transcript", "").strip()
            print(f"[Sarvam STT] Transcribed: '{transcript}'")
    except Exception as e:
        print(f"[STT Error] {e}")

    if not transcript:
        raise HTTPException(status_code=400, detail="Could not transcribe audio.")

    final_lang = classify_indic_language(transcript, resolved_hint)
    rag_response = process_text_query(QueryRequest(query=transcript, language=final_lang))
    rag_response["transcribed_text"] = transcript
    rag_response["detected_language"] = final_lang
    rag_response["audio_pipeline_total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
    return rag_response