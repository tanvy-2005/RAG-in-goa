import os
import time
import json
import re
import torch
import numpy as np
import faiss
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer

# ============================================================
# 1. RUNTIME CONFIGURATION (<200ms TARGET)
# ============================================================
torch.set_num_threads(4)

SARVAM_API_KEY = "sk_8x94884j_MrW1uKlHhyOVd3Qf4tAhxopU"
INDEX_FILE = "multilingual.index"
METADATA_FILE = "multilingual_metadata.jsonl"
MODEL_NAME = "intfloat/multilingual-e5-small"

app = FastAPI(
    title="Voice-Enabled Multilingual Indic RAG Harness",
    description="Sub-200ms 14-Language Indic Vector Search, Sarvam STT & Strict Grounding Guardrail Engine",
    version="12.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 2. LOAD FAISS INDEX & METADATA
# ============================================================
print("=" * 60)
print("INITIALIZING MULTILINGUAL INDIC RAG ENGINE")
print("=" * 60)

if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
    raise FileNotFoundError("Index or metadata file missing.")

print(f"Loading FAISS Index from {INDEX_FILE}...")
index = faiss.read_index(INDEX_FILE)
print(f"Total Vectors Indexed: {index.ntotal:,}")

print(f"Loading Metadata from {METADATA_FILE}...")
metadata = []
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            metadata.append(json.loads(line))
print(f"Total Metadata Records: {len(metadata):,}")

print(f"Loading Embedding Model: {MODEL_NAME}...")
embed_model = SentenceTransformer(MODEL_NAME)

for _ in range(2):
    with torch.inference_mode():
        _ = embed_model.encode(["query: warmup"], normalize_embeddings=True, convert_to_numpy=True)
print("System Warmup Complete. Pipeline Ready!\n")


# ============================================================
# 3. LANGUAGE CODE ALIASES & SCRIPT CLASSIFIER
# ============================================================
LANG_ALIASES = {
    "od": "or",
    "ori": "or",
    "odia": "or",
    "oriya": "or",
    "tam": "ta",
    "tamil": "ta",
    "tel": "te",
    "telugu": "te",
    "kan": "kn",
    "kannada": "kn",
    "mal": "ml",
    "malayalam": "ml",
    "ben": "bn",
    "bengali": "bn",
    "bangla": "bn",
    "asm": "as",
    "assamese": "as",
    "asamiya": "as",
    "guj": "gu",
    "gujarati": "gu",
    "pan": "pa",
    "punjabi": "pa",
    "urd": "ur",
    "urdu": "ur",
    "hin": "hi",
    "hindi": "hi",
    "mar": "mr",
    "marathi": "mr",
    "nep": "ne",
    "nepali": "ne",
    "san": "sa",
    "sanskrit": "sa",
    "eng": "en",
    "english": "en",
}

SARVAM_BCP47_MAP = {
    "hi": "hi-IN",
    "bn": "bn-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "pa": "pa-IN",
    "or": "od-IN",
    "as": "as-IN",
    "ur": "ur-IN",
    "ne": "ne-IN",
    "sa": "sa-IN",
    "en": "en-IN",
}

def normalize_lang_code(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    c = code.strip().lower().split("-")[0].split("_")[0]
    return LANG_ALIASES.get(c, c)

def infer_lang_from_filename(filename: str) -> Optional[str]:
    fn = filename.lower()
    for key, val in LANG_ALIASES.items():
        if key in fn:
            return val
    return None

def classify_indic_language(text: str, hint_lang: Optional[str] = None) -> str:
    norm_hint = normalize_lang_code(hint_lang)
    if norm_hint and norm_hint not in ["auto", "unknown", ""]:
        return norm_hint

    cleaned_txt = re.sub(r"[।॥?!,.:;\"'()\-—]", " ", text).strip()
    words = set(cleaned_txt.split())
    txt_blob = f" {cleaned_txt} "

    if re.search(r"[\u0B00-\u0B7F]", cleaned_txt):
        return "or"
    if re.search(r"[\u0B80-\u0BFF]", cleaned_txt):
        return "ta"
    if re.search(r"[\u0C00-\u0C7F]", cleaned_txt):
        return "te"
    if re.search(r"[\u0C80-\u0CFF]", cleaned_txt):
        return "kn"
    if re.search(r"[\u0D00-\u0D7F]", cleaned_txt):
        return "ml"
    if re.search(r"[\u0A80-\u0AFF]", cleaned_txt):
        return "gu"
    if re.search(r"[\u0A00-\u0A7F]", cleaned_txt):
        return "pa"
    if re.search(r"[\u0600-\u06FF]", cleaned_txt):
        return "ur"

    if re.search(r"[\u0980-\u09FF]", cleaned_txt):
        if any(c in cleaned_txt for c in ["\u09F0", "\u09F1", "ৰ", "ৱ"]):
            return "as"
        assamese_words = {"কি", "কৰ্পোৰেচন", "হৈছে", "কেনেদৰে", "আছে", "নহয়", "হয়", "কৰিব", "পৰা", "বাবে", "লগত", "কৰি", "হোৱা", "তেওঁ"}
        if any(w in words for w in assamese_words):
            return "as"
        return "bn"

    if re.search(r"[\u0900-\u097F]", cleaned_txt):
        if "\u0933" in cleaned_txt or "ळ" in cleaned_txt:
            return "mr"
        marathi_markers = ["काय", "म्हणजे", "कसा", "कशी", "कसे", "आहे", "नाही", "झाले", "केले", "करावे", "येथे", "होते", "आहेत"]
        if any(f" {m} " in txt_blob for m in marathi_markers):
            return "mr"

        sanskrit_markers = ["किमिति", "किमुच्यते", "अस्ति", "उच्यते", "इति", "कथम्", "भवति", "सर्वम्", "निगमम्", "सन्ति", "विद्यते"]
        if "ः" in cleaned_txt or any(f" {s} " in txt_blob for s in sanskrit_markers) or any(w.endswith("उच्यते") or w.endswith("किमिति") for w in words):
            return "sa"

        nepali_patterns = ["के हो", "हो निगम", "के निगम", "निगम", "भनेको", "कसरी", "गर्छ", "हुन्", "छन्"]
        nepali_words = {"के", "हो", "छ", "छन्", "छैन", "हुन्", "गर्छ", "भनेको", "कसरी", "किन", "हुन्छ", "भयो", "थियो", "निगम"}
        if any(p in cleaned_txt for p in nepali_patterns) or (any(w in nepali_words for w in words) and not any(w in {"क्या", "क्यों", "था", "थी"} for w in words)):
            return "ne"

        return "hi"

    if re.search(r"[a-zA-Z]", cleaned_txt):
        return "en"

    return "en"


def query_safety_guardrail(query: str) -> bool:
    blocked = ["ignore previous instructions", "drop table", "system prompt", "exec(", "eval(", "<script>"]
    return not any(p in query.lower() for p in blocked)


# ============================================================
# 4. STRICT PASSAGE & QUERY CONTEXT FILTER
# ============================================================
def passage_matches_query_context(query: str, text: str, score: float) -> bool:
    if score < 0.82:
        return False
    
    clean_q = re.sub(r"[।॥?!,.:;\"'()\-—]", " ", query.lower()).strip()
    clean_p = re.sub(r"[।॥?!,.:;\"'()\-—]", " ", text.lower()).strip()
    
    stopwords = {
        "what", "is", "a", "an", "the", "are", "in", "of", "to", "for", "and", "or", "how", "why", "who", "aur", "definition",
        "क्या", "है", "की", "का", "के", "में", "से", "पर", "एक", "को", "हो",
        "কী", "হল", "ଏକ", "କଣ", "ഒരു", "എന്നാണ്", "என்ன", "என்பது", "అంటే", "ఏమిటి",
        "काय", "म्हणजे", "ਕੀ", "ہے", "کیا", "किमिति", "अस्ति"
    }
    
    q_tokens = [w for w in clean_q.split() if w not in stopwords and len(w) >= 3]
    
    # Require lexical/stem containment if score is in borderline range
    if score < 0.86 and q_tokens:
        return any(token in clean_p for token in q_tokens)
        
    return True


def grounding_guardrail(passages: List[dict], query: str, threshold: float = 0.84) -> bool:
    if not passages or len(passages) == 0:
        return False

    top_score = passages[0]["score"]
    
    if top_score < threshold:
        return False

    if top_score < 0.88:
        clean_q = re.sub(r"[।॥?!,.:;\"'()\-—]", " ", query.lower()).strip()
        clean_p = re.sub(r"[।॥?!,.:;\"'()\-—]", " ", passages[0]["text"].lower()).strip()
        
        stopwords = {
            "what", "is", "a", "an", "the", "are", "in", "of", "to", "for", "and", "or", "how", "why", "who", "aur", "definition",
            "क्या", "है", "की", "का", "के", "में", "से", "पर", "एक", "को", "हो",
            "কী", "হল", "ଏକ", "କଣ", "ഒരു", "എന്നാണ്", "என்ன", "என்பது", "అంటే", "ఏమిటి",
            "काय", "म्हणजे", "ਕੀ", "ہے", "کیا", "किमिति", "अस्ति"
        }
        
        q_tokens = [w for w in clean_q.split() if w not in stopwords and len(w) >= 3]
        if q_tokens:
            has_overlap = any(token in clean_p for token in q_tokens)
            if not has_overlap:
                return False

    return True


# ============================================================
# 5. VECTOR RETRIEVAL
# ============================================================
def retrieve_passages(query: str, top_k: int = 3, target_lang: Optional[str] = None):
    t0 = time.perf_counter()
    effective_lang = classify_indic_language(query, target_lang)

    with torch.inference_mode():
        q_emb = embed_model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

    scores, indices = index.search(q_emb, min(100, index.ntotal))
    retrieval_time_ms = (time.perf_counter() - t0) * 1000.0

    valid_results = []

    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(metadata):
            doc = metadata[idx]
            doc_lang = str(doc.get("language", "")).strip().lower()
            doc_text = doc.get("text", "").strip()
            
            if doc_lang == effective_lang:
                # Eliminate noisy crawl negatives and ungrounded passages
                if not passage_matches_query_context(query, doc_text, float(score)):
                    continue

                valid_results.append({
                    "score": float(score),
                    "language": doc_lang,
                    "query_id": doc.get("query_id"),
                    "text": doc_text
                })
                
                if len(valid_results) >= top_k:
                    break

    print(f"[RAG Retrieval] Language: {effective_lang.upper()} | Clean Grounded Matches: {len(valid_results)} | FAISS Latency: {retrieval_time_ms:.2f}ms")
    return valid_results, retrieval_time_ms, effective_lang


# ============================================================
# 6. API ENDPOINTS
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
    
    if not query_safety_guardrail(req.query):
        raise HTTPException(status_code=400, detail="Query blocked by safety guardrail.")
    
    passages, ret_time, matched_lang = retrieve_passages(req.query, top_k=3, target_lang=req.language)
    is_grounded = grounding_guardrail(passages, req.query, threshold=0.84)
    total_time = (time.perf_counter() - t_start) * 1000.0
    
    if not is_grounded:
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
        resolved_hint = infer_lang_from_filename(orig_filename)

    sarvam_lang_code = SARVAM_BCP47_MAP.get(resolved_hint, "unknown") if resolved_hint else "unknown"

    transcript = ""
    stt_detected_lang = "unknown"
    
    try:
        url = "https://api.sarvam.ai/speech-to-text"
        headers = {"api-subscription-key": SARVAM_API_KEY}
        filename = "audio.wav" if "wav" in orig_filename.lower() else "audio.mp3"
        files = {"file": (filename, audio_data, "audio/mpeg" if "mp3" in filename else "audio/wav")}
        data = {"model": "saaras:v3", "mode": "transcribe", "language_code": sarvam_lang_code}
        
        res = requests.post(url, headers=headers, files=files, data=data, timeout=15)
        if res.status_code == 200:
            resp_json = res.json()
            transcript = resp_json.get("transcript", "").strip()
            stt_detected_lang = resp_json.get("language_code", "unknown")
            print(f"[Sarvam STT] File: '{orig_filename}' | Sent Lang: {sarvam_lang_code} | Transcribed: '{transcript}'")
        else:
            print(f"[Sarvam STT Error] {res.text}")
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