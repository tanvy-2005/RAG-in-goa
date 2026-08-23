import os

# ============================================================
# 0. ENVIRONMENT / RESOURCE CONFIGURATION
# ============================================================

# Set these BEFORE importing torch.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import json
import re
import threading

import numpy as np
import faiss
import requests
import torch

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from transformers import AutoTokenizer, AutoModel


# ============================================================
# 1. RUNTIME CONFIGURATION
# ============================================================

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()

if HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN


# ============================================================
# 2. FILE / MODEL CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_FILE = os.path.join(
    BASE_DIR,
    "multilingual.index"
)

METADATA_FILE = os.path.join(
    BASE_DIR,
    "multilingual_metadata.jsonl"
)

MODEL_NAME = "intfloat/multilingual-e5-small"


# ============================================================
# 3. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Voice-Enabled Multilingual Indic RAG Harness",
    description="Multilingual Indic RAG retrieval engine using FAISS and multilingual-e5-small",
    version="39.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 4. LOAD FAISS INDEX
# ============================================================

print("=" * 60)
print("INITIALIZING MULTILINGUAL INDIC NEURAL RAG ENGINE")
print("=" * 60)

print(f"Looking for FAISS index:")
print(INDEX_FILE)

print(f"Looking for metadata:")
print(METADATA_FILE)


if not os.path.exists(INDEX_FILE):
    raise FileNotFoundError(
        f"FAISS index not found: {INDEX_FILE}"
    )

if not os.path.exists(METADATA_FILE):
    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_FILE}"
    )


print("\nLoading FAISS Index...")

index = faiss.read_index(INDEX_FILE)

total_vectors = index.ntotal

print(
    f"Total Vectors Indexed: {total_vectors:,}"
)


# ============================================================
# 5. BUILD METADATA DISK SEEK MAP
# ============================================================

doc_offsets = []
docs_cache = {}

print("Building disk seek map...")

with open(METADATA_FILE, "rb") as f:

    offset = 0

    for line in f:

        doc_offsets.append(offset)

        offset += len(line)


print(
    f"Metadata records available: {len(doc_offsets):,}"
)


if len(doc_offsets) != total_vectors:

    print(
        "WARNING: FAISS vector count and metadata record count "
        "do not match!"
    )

    print(
        f"FAISS vectors : {total_vectors:,}"
    )

    print(
        f"Metadata rows : {len(doc_offsets):,}"
    )


# ============================================================
# 6. METADATA READER
# ============================================================

def get_metadata_by_id(doc_idx: int) -> dict:

    if doc_idx in docs_cache:
        return docs_cache[doc_idx]

    if not (0 <= doc_idx < len(doc_offsets)):
        return {}

    try:

        with open(METADATA_FILE, "rb") as f:

            f.seek(doc_offsets[doc_idx])

            line = f.readline().decode(
                "utf-8",
                errors="ignore"
            )

        if not line.strip():
            return {}

        data = json.loads(line)

        # Small cache
        if len(docs_cache) < 1000:
            docs_cache[doc_idx] = data

        return data

    except Exception as e:

        print(
            f"[Metadata Error] index={doc_idx}: {e}"
        )

        return {}


# ============================================================
# 7. LAZY MULTILINGUAL EMBEDDING MODEL
# ============================================================

tokenizer = None
embed_model = None

model_lock = threading.Lock()


def load_embedding_model():

    global tokenizer
    global embed_model

    # Already loaded
    if tokenizer is not None and embed_model is not None:
        return

    with model_lock:

        # Check again after acquiring lock
        if tokenizer is not None and embed_model is not None:
            return

        print("\n" + "=" * 60)
        print("LOADING MULTILINGUAL EMBEDDING MODEL")
        print("=" * 60)

        auth_token = (
            HF_TOKEN
            if HF_TOKEN and not HF_TOKEN.startswith("hf_xxx")
            else None
        )

        print(
            f"Model: {MODEL_NAME}"
        )

        print("Loading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=True,
            token=auth_token
        )

        print("Tokenizer loaded!")

        print("Loading transformer model...")

        model = AutoModel.from_pretrained(
            MODEL_NAME,
            low_cpu_mem_usage=True,
            token=auth_token
        )

        print("Transformer model loaded!")

        # ----------------------------------------------------
        # Dynamic INT8 quantization
        # ----------------------------------------------------
        #
        # This reduces RAM usage for Linear layers.
        # It is especially useful on CPU-only Railway deployments.
        #

        try:

            print("Applying dynamic INT8 quantization...")

            model = torch.quantization.quantize_dynamic(
                model,
                {
                    torch.nn.Linear
                },
                dtype=torch.qint8
            )

            print(
                "Dynamic INT8 quantization applied!"
            )

        except Exception as e:

            print(
                f"WARNING: Quantization failed: {e}"
            )

            print(
                "Continuing with normal CPU model."
            )

        model.eval()

        embed_model = model

        print(
            "Multilingual embedding model ready!"
        )

        print("=" * 60)


# ============================================================
# 8. QUERY EMBEDDING
# ============================================================

def encode_query(query_text: str) -> np.ndarray:

    if not query_text or not query_text.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    # Load only when first request arrives.
    load_embedding_model()

    formatted = (
        "query: "
        + query_text.strip()
    )

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        max_length=128,
        padding=True,
        truncation=True
    )

    with torch.inference_mode():

        outputs = embed_model(
            **inputs
        )

        # Mean pooling
        mask = (
            inputs["attention_mask"]
            .unsqueeze(-1)
            .expand(
                outputs.last_hidden_state.size()
            )
            .float()
        )

        sum_embeddings = torch.sum(
            outputs.last_hidden_state * mask,
            dim=1
        )

        sum_mask = torch.clamp(
            mask.sum(dim=1),
            min=1e-9
        )

        mean_pooled = (
            sum_embeddings / sum_mask
        )

        # L2 normalization
        normalized = torch.nn.functional.normalize(
            mean_pooled,
            p=2,
            dim=1
        )

        embedding = (
            normalized
            .cpu()
            .numpy()
            .astype("float32")
        )

    return embedding


# ============================================================
# 9. LANGUAGE ALIASES
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
    "english": "en"
}


# ============================================================
# 10. SARVAM LANGUAGE MAP
# ============================================================

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
    "en": "en-IN"
}


# ============================================================
# 11. LANGUAGE NORMALIZATION
# ============================================================

def normalize_lang_code(
    code: Optional[str]
) -> Optional[str]:

    if not code:
        return None

    c = (
        code
        .strip()
        .lower()
        .split("-")[0]
        .split("_")[0]
    )

    return LANG_ALIASES.get(
        c,
        c
    )


# ============================================================
# 12. INDIC LANGUAGE CLASSIFIER
# ============================================================

def classify_indic_language(
    text: str,
    hint_lang: Optional[str] = None
) -> str:

    norm_hint = normalize_lang_code(
        hint_lang
    )

    if (
        norm_hint
        and norm_hint not in [
            "auto",
            "unknown",
            ""
        ]
    ):
        return norm_hint

    cleaned = re.sub(
        r"[।॥?!,.:;\"'()\-—]",
        " ",
        text
    ).strip()

    words = set(
        cleaned.split()
    )

    txt_blob = (
        f" {cleaned} "
    )

    # Odia
    if re.search(
        r"[\u0B00-\u0B7F]",
        cleaned
    ):
        return "or"

    # Tamil
    if re.search(
        r"[\u0B80-\u0BFF]",
        cleaned
    ):
        return "ta"

    # Telugu
    if re.search(
        r"[\u0C00-\u0C7F]",
        cleaned
    ):
        return "te"

    # Kannada
    if re.search(
        r"[\u0C80-\u0CFF]",
        cleaned
    ):
        return "kn"

    # Malayalam
    if re.search(
        r"[\u0D00-\u0D7F]",
        cleaned
    ):
        return "ml"

    # Gujarati
    if re.search(
        r"[\u0A80-\u0AFF]",
        cleaned
    ):
        return "gu"

    # Punjabi
    if re.search(
        r"[\u0A00-\u0A7F]",
        cleaned
    ):
        return "pa"

    # Urdu
    if re.search(
        r"[\u0600-\u06FF]",
        cleaned
    ):
        return "ur"

    # Bengali / Assamese
    if re.search(
        r"[\u0980-\u09FF]",
        cleaned
    ):

        if any(
            c in cleaned
            for c in [
                "ৰ",
                "ৱ"
            ]
        ):
            return "as"

        if any(
            w in words
            for w in [
                "কি",
                "কৰ্পোৰেচন",
                "হৈছে",
                "আছে"
            ]
        ):
            return "as"

        return "bn"

    # Devanagari
    if re.search(
        r"[\u0900-\u097F]",
        cleaned
    ):

        # Marathi
        if (
            "\u0933" in cleaned
            or "ळ" in cleaned
        ):
            return "mr"

        if any(
            f" {m} " in txt_blob
            for m in [
                "काय",
                "म्हणजे",
                "कसा",
                "आहे",
                "नाही"
            ]
        ):
            return "mr"

        # Sanskrit
        if (
            "ः" in cleaned
            or any(
                f" {s} " in txt_blob
                for s in [
                    "किमिति",
                    "अस्ति",
                    "भवति"
                ]
            )
        ):
            return "sa"

        # Nepali
        if any(
            p in cleaned
            for p in [
                "के हो",
                "हो निगम",
                "भनेको"
            ]
        ):
            return "ne"

        return "hi"

    # Default English
    return "en"


# ============================================================
# 13. FAISS RETRIEVAL
# ============================================================

def retrieve_passages(
    query: str,
    top_k: int = 3,
    target_lang: Optional[str] = None
):

    t0 = time.perf_counter()

    effective_lang = classify_indic_language(
        query,
        target_lang
    )

    # Generate query embedding
    q_emb = encode_query(
        query
    )

    search_k = min(
        100,
        index.ntotal
    )

    scores, indices = index.search(
        q_emb,
        search_k
    )

    lang_matches = []
    global_matches = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if not (
            0 <= idx < total_vectors
        ):
            continue

        doc = get_metadata_by_id(
            int(idx)
        )

        if not doc:
            continue

        doc_lang = normalize_lang_code(
            str(
                doc.get(
                    "language",
                    ""
                )
            )
        )

        item = {

            "score": round(
                float(score),
                4
            ),

            "language": (
                doc_lang
                or effective_lang
            ),

            "query_id": doc.get(
                "query_id"
            ),

            "text": doc.get(
                "text",
                ""
            )
        }

        global_matches.append(
            item
        )

        if doc_lang == effective_lang:
            lang_matches.append(
                item
            )

    # Prefer language-specific retrieval
    if len(lang_matches) >= top_k:

        results = lang_matches[:top_k]

    elif len(lang_matches) > 0:

        # Fill remaining slots globally
        results = lang_matches[:]

        used_ids = {
            x.get("query_id")
            for x in results
        }

        for item in global_matches:

            if (
                item.get("query_id")
                not in used_ids
            ):

                results.append(
                    item
                )

            if len(results) >= top_k:
                break

    else:

        results = global_matches[
            :top_k
        ]

    ret_time = (
        time.perf_counter()
        - t0
    ) * 1000.0

    print(
        f"[Neural RAG] "
        f"Query='{query[:50]}' | "
        f"Language={effective_lang} | "
        f"Matches={len(results)} | "
        f"Latency={ret_time:.2f}ms"
    )

    return (
        results,
        ret_time,
        effective_lang
    )


# ============================================================
# 14. REQUEST MODEL
# ============================================================

class QueryRequest(BaseModel):

    query: str

    language: Optional[str] = "auto"


# ============================================================
# 15. HEALTH CHECK
# ============================================================

@app.get("/")
def health_check():

    return {

        "status": "online",

        "service":
            "Voice-Enabled Multilingual Indic RAG Harness",

        "model":
            MODEL_NAME,

        "vectors_indexed":
            index.ntotal,

        "metadata_records":
            len(doc_offsets),

        "embedding_model_loaded":
            embed_model is not None,

        "docs_url":
            "/docs"
    }


# ============================================================
# 16. MODEL STATUS ENDPOINT
# ============================================================

@app.get("/api/status")
def status():

    return {

        "status": "online",

        "faiss_vectors":
            index.ntotal,

        "metadata_records":
            len(doc_offsets),

        "embedding_model_loaded":
            embed_model is not None,

        "model":
            MODEL_NAME
    }


# ============================================================
# 17. TEXT RAG ENDPOINT
# ============================================================

@app.post("/api/ask")
def process_text_query(
    req: QueryRequest
):

    if not req.query or not req.query.strip():

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    t_start = time.perf_counter()

    try:

        passages, ret_time, matched_lang = (
            retrieve_passages(
                req.query,
                top_k=3,
                target_lang=req.language
            )
        )

    except Exception as e:

        print(
            f"[Retrieval Error] {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {str(e)}"
        )

    total_time = (
        time.perf_counter()
        - t_start
    ) * 1000.0

    # --------------------------------------------------------
    # Grounding check
    # --------------------------------------------------------

    if (
        not passages
        or passages[0]["score"] < 0.35
    ):

        return {

            "query":
                req.query,

            "language":
                matched_lang,

            "answer":
                "The query is outside the verified dataset knowledge base.",

            "grounded":
                False,

            "passages":
                [],

            "latency_ms":
                round(
                    total_time,
                    2
                ),

            "retrieval_ms":
                round(
                    ret_time,
                    2
                ),

            "passed_target_200ms":
                bool(
                    total_time < 200.0
                )
        }

    # --------------------------------------------------------
    # Grounded response
    # --------------------------------------------------------

    return {

        "query":
            req.query,

        "language":
            matched_lang,

        "answer":
            passages[0]["text"],

        "grounded":
            True,

        "passages":
            passages,

        "latency_ms":
            round(
                total_time,
                2
            ),

        "retrieval_ms":
            round(
                ret_time,
                2
            ),

        "passed_target_200ms":
            bool(
                total_time < 200.0
            )
    }


# ============================================================
# 18. SARVAM VOICE QUERY
# ============================================================

@app.post("/api/voice-ask")
async def process_voice_query(

    file: UploadFile = File(...),

    language: Optional[str] = Form(
        "auto"
    )
):

    t_start = time.perf_counter()

    if not SARVAM_API_KEY:

        raise HTTPException(
            status_code=500,
            detail=(
                "SARVAM_API_KEY is not configured "
                "on the server."
            )
        )

    audio_data = await file.read()

    if not audio_data:

        raise HTTPException(
            status_code=400,
            detail="Uploaded audio file is empty."
        )

    orig_filename = (
        file.filename
        or ""
    )

    # --------------------------------------------------------
    # Determine requested language
    # --------------------------------------------------------

    resolved_hint = None

    if (
        language
        and language.strip().lower()
        not in [
            "auto",
            "unknown",
            ""
        ]
    ):

        resolved_hint = normalize_lang_code(
            language
        )

    else:

        # Try filename
        resolved_hint = normalize_lang_code(
            orig_filename
        )

    sarvam_lang_code = (
        SARVAM_BCP47_MAP.get(
            resolved_hint,
            "unknown"
        )
        if resolved_hint
        else "unknown"
    )

    transcript = ""

    # --------------------------------------------------------
    # Sarvam Speech-to-Text
    # --------------------------------------------------------

    try:

        url = (
            "https://api.sarvam.ai/"
            "speech-to-text"
        )

        headers = {

            "api-subscription-key":
                SARVAM_API_KEY
        }

        is_mp3 = (
            "mp3"
            in orig_filename.lower()
        )

        filename = (
            "audio.mp3"
            if is_mp3
            else "audio.wav"
        )

        content_type = (
            "audio/mpeg"
            if is_mp3
            else "audio/wav"
        )

        files = {

            "file": (
                filename,
                audio_data,
                content_type
            )
        }

        data = {

            "model":
                "saaras:v3",

            "mode":
                "transcribe",

            "language_code":
                sarvam_lang_code
        }

        print(
            f"[Sarvam STT] "
            f"Language={sarvam_lang_code}"
        )

        res = requests.post(

            url,

            headers=headers,

            files=files,

            data=data,

            timeout=15
        )

        print(
            f"[Sarvam STT] "
            f"HTTP {res.status_code}"
        )

        if res.status_code == 200:

            response_json = (
                res.json()
            )

            transcript = (
                response_json
                .get(
                    "transcript",
                    ""
                )
                .strip()
            )

            print(
                f"[Sarvam STT] "
                f"Transcribed: "
                f"'{transcript}'"
            )

        else:

            print(
                "[Sarvam STT] "
                f"Error response: "
                f"{res.text[:500]}"
            )

    except requests.Timeout:

        raise HTTPException(
            status_code=504,
            detail=(
                "Sarvam speech-to-text "
                "request timed out."
            )
        )

    except Exception as e:

        print(
            f"[STT Error] {e}"
        )

    # --------------------------------------------------------
    # Validate transcript
    # --------------------------------------------------------

    if not transcript:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not transcribe audio."
            )
        )

    # --------------------------------------------------------
    # Detect language
    # --------------------------------------------------------

    final_lang = classify_indic_language(
        transcript,
        resolved_hint
    )

    # --------------------------------------------------------
    # Run RAG
    # --------------------------------------------------------

    rag_response = process_text_query(

        QueryRequest(

            query=transcript,

            language=final_lang
        )
    )

    rag_response[
        "transcribed_text"
    ] = transcript

    rag_response[
        "detected_language"
    ] = final_lang

    rag_response[
        "audio_pipeline_total_ms"
    ] = round(

        (
            time.perf_counter()
            - t_start
        ) * 1000.0,

        2
    )

    return rag_response


# ============================================================
# 19. STARTUP MESSAGE
# ============================================================

@app.on_event("startup")
async def startup_event():

    print("\n")
    print("=" * 60)
    print("RAG-IN-GOA API STARTED")
    print("=" * 60)

    print(
        f"FAISS vectors : {index.ntotal:,}"
    )

    print(
        f"Metadata rows : {len(doc_offsets):,}"
    )

    print(
        f"Embedding model : {MODEL_NAME}"
    )

    print(
        "Embedding model will be loaded lazily "
        "on the first query."
    )

    print(
        "Server startup complete."
    )

    print("=" * 60)
    print("\n")