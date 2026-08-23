import os
import time
import json
import re
import numpy as np
import faiss
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# ============================================================
# 1. RUNTIME CONFIGURATION (HF MODEL + HIGH PERFORMANCE)
# ============================================================
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "sk_8x94884j_MrW1uKlHhyOVd3Qf4tAhxopU")
HF_TOKEN = os.getenv("HF_TOKEN", "")

INDEX_FILE = os.path.join(os.path.dirname(__file__), "multilingual.index")
METADATA_FILE = os.path.join(os.path.dirname(__file__), "multilingual_metadata.jsonl")

# Direct Reliable Hugging Face Feature Extraction Endpoint
HF_MODEL_NAME = "intfloat/multilingual-e5-small"
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL_NAME}"

app = FastAPI(
    title="Voice-Enabled Multilingual Indic RAG Harness",
    description="Sub-200ms Indic Neural RAG with Hugging Face Model Integration",
    version="33.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent HTTP connection session for sub-80ms API calls
http_session = requests.Session()

# ============================================================
# 2. LOAD FAISS INDEX & FAST BYTE-SEEK TABLE
# ============================================================
print("=" * 60)
print(f"INITIALIZING INDIC RAG ENGINE WITH HF MODEL: {HF_MODEL_NAME}")
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

doc_offsets = []
docs_cache = {}

print("Mapping seek table for O(1) retrieval...")
with open(METADATA_FILE, "rb") as f:
    offset = 0
    for line in f:
        doc_offsets.append(offset)
        offset += len(line)

def get_metadata_by_id(doc_idx: int) -> dict:
    if doc_idx in docs_cache:
        return docs_cache[doc_idx]
    if 0 <= doc_idx < len(doc_offsets):
        with open(METADATA_FILE, "rb") as f:
            f.seek(doc_offsets[doc_idx])
            line = f.readline().decode("utf-8", errors="ignore")
            if line.strip():
                try:
                    data = json.loads(line)
                    docs_cache[doc_idx] = data
                    return data
                except Exception:
                    return {}
    return {}


# ============================================================
# 3. HUGGING FACE INFERENCE EMBEDDER (ACCURATE INDIC VECTORS)
# ============================================================

def encode_with_hf_model(query_text: str) -> Optional[np.ndarray]:
    """Fetches exact 384d semantic vectors from Hugging Face Inference."""
    token = os.getenv("HF_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    # multilingual-e5 models strictly require 'query: ' prefix
    payload = {
        "inputs": f"query: {query_text.strip()}",
        "options": {"wait_for_model": True, "use_cache": True}
    }
    
    try:
        res = http_session.post(
            "https://api-inference.huggingface.co/pipeline/feature-extraction/intfloat/multilingual-e5-small",
            headers=headers,
            json=payload,
            timeout=8.0
        )
        if res.status_code == 200:
            data = res.json()
            emb = np.array(data, dtype="float32")
            
            # Reshape based on HF feature extraction output format
            if emb.ndim == 2:
                emb = emb.mean(axis=0, keepdims=True)
            elif emb.ndim == 1:
                emb = np.expand_dims(emb, axis=0)
            elif emb.ndim == 3:
                emb = emb.mean(axis=1)

            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            return (emb / norm).astype("float32")
        else:
            print(f"[HF Error {res.status_code}] {res.text}")
    except Exception as e:
        print(f"[HF Exception] {e}")
    return None

# ============================================================
# 4. INDIC SCRIPT CLASSIFIER
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
# 5. SEMANTIC FAISS RETRIEVAL
# ============================================================
def retrieve_passages(query: str, top_k: int = 3, target_lang: Optional[str] = None):
    t0 = time.perf_counter()
    effective_lang = classify_indic_language(query, target_lang)

    # 1. Fetch exact semantic embedding from HF Model
    q_emb = encode_with_hf_model(query)
    results = []

    if q_emb is not None:
        scores, indices = index.search(q_emb, min(100, index.ntotal))
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < total_vectors:
                doc = get_metadata_by_id(int(idx))
                if not doc: continue
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

    # If strict language subset produced 0 hits, grab global top semantic hits
    if not results and q_emb is not None:
        scores, indices = index.search(q_emb, top_k)
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

    results.sort(key=lambda x: x["score"], reverse=True)
    final_passages = results[:top_k]
    ret_time = (time.perf_counter() - t0) * 1000.0
    print(f"[Neural RAG] Query: '{query[:30]}' | Lang: {effective_lang.upper()} | Matches: {len(final_passages)} | Latency: {ret_time:.2f}ms")
    return final_passages, ret_time, effective_lang


# ============================================================
# 6. FASTAPI ENDPOINTS
# ============================================================
class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = "auto"

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Voice-Enabled Multilingual Indic RAG Harness",
        "hf_model": HF_MODEL_NAME,
        "vectors_indexed": index.ntotal,
        "docs_url": "/docs"
    }

@app.post("/api/ask")
def process_text_query(req: QueryRequest):
    t_start = time.perf_counter()
    passages, ret_time, matched_lang = retrieve_passages(req.query, top_k=3, target_lang=req.language)
    total_time = (time.perf_counter() - t_start) * 1000.0

    if not passages or passages[0]["score"] < 0.60:
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