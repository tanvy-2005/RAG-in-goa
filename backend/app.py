import os
import time
import json
import re
import gc

# ============================================================
# MEMORY CONFIGURATION — MUST BE BEFORE TORCH IMPORT
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

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
# 1. PYTORCH MEMORY SETTINGS
# ============================================================

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Disable gradients globally
torch.set_grad_enabled(False)


# ============================================================
# 2. ENVIRONMENT VARIABLES
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()

if not SARVAM_API_KEY:
    print("WARNING: SARVAM_API_KEY is not configured.")


# ============================================================
# 3. FILE PATHS
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
# 4. FASTAPI
# ============================================================

app = FastAPI(
    title="RAG-in-Goa Multilingual RAG API",
    description="Multilingual Indic semantic retrieval using FAISS",
    version="39.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 5. LOAD FAISS INDEX
# ============================================================

print("=" * 60)
print("INITIALIZING MULTILINGUAL INDIC RAG ENGINE")
print("=" * 60)

if not os.path.exists(INDEX_FILE):
    raise FileNotFoundError(
        f"FAISS index not found: {INDEX_FILE}"
    )

if not os.path.exists(METADATA_FILE):
    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_FILE}"
    )


print(f"Loading FAISS index from: {INDEX_FILE}")

index = faiss.read_index(INDEX_FILE)

total_vectors = index.ntotal

print(
    f"Total vectors indexed: {total_vectors:,}"
)


# ============================================================
# 6. METADATA DISK SEEK MAP
# ============================================================

print("Building metadata disk seek map...")

doc_offsets = []

with open(
    METADATA_FILE,
    "rb"
) as f:

    offset = 0

    for line in f:

        doc_offsets.append(offset)

        offset += len(line)


print(
    f"Metadata records: {len(doc_offsets):,}"
)


# Small cache only
# Prevents unbounded memory growth.

docs_cache = {}

MAX_CACHE_SIZE = 100


def get_metadata_by_id(doc_idx: int):

    if doc_idx in docs_cache:

        return docs_cache[doc_idx]

    if not (
        0 <= doc_idx < len(doc_offsets)
    ):
        return {}


    try:

        with open(
            METADATA_FILE,
            "rb"
        ) as f:

            f.seek(
                doc_offsets[doc_idx]
            )

            line = f.readline()


        if not line:
            return {}


        data = json.loads(
            line.decode(
                "utf-8",
                errors="ignore"
            )
        )


        # Keep cache bounded
        if len(docs_cache) >= MAX_CACHE_SIZE:

            first_key = next(
                iter(docs_cache)
            )

            del docs_cache[first_key]


        docs_cache[doc_idx] = data

        return data


    except Exception as e:

        print(
            f"Metadata error: {e}"
        )

        return {}


# ============================================================
# 7. LOAD MULTILINGUAL E5 MODEL
# ============================================================

print()
print(
    f"Loading embedding model: {MODEL_NAME}"
)

auth_token = (
    HF_TOKEN
    if HF_TOKEN
    and not HF_TOKEN.startswith("hf_xxx")
    else None
)


# Tokenizer

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    token=auth_token
)


# Model

embed_model = AutoModel.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True,
    token=auth_token
)

embed_model.eval()


# ============================================================
# 8. DYNAMIC INT8 QUANTIZATION
# ============================================================

print()
print("Applying CPU INT8 quantization...")

try:

    embed_model = torch.quantization.quantize_dynamic(
        embed_model,
        {
            torch.nn.Linear
        },
        dtype=torch.qint8
    )

    print(
        "INT8 quantization enabled."
    )

except Exception as e:

    print(
        "INT8 quantization unavailable:"
    )

    print(e)

    print(
        "Continuing with FP32 model."
    )


# ============================================================
# 9. QUERY EMBEDDING
# ============================================================

def encode_query(
    query_text: str
) -> np.ndarray:

    """
    Generate multilingual-e5 query embedding.

    E5 requires:
        query: <text>

    Output:
        normalized float32 vector
    """

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


        hidden = (
            outputs.last_hidden_state
        )


        mask = (
            inputs["attention_mask"]
            .unsqueeze(-1)
            .expand(hidden.size())
            .float()
        )


        masked_embeddings = (
            hidden * mask
        )


        sum_embeddings = (
            masked_embeddings.sum(
                dim=1
            )
        )


        sum_mask = (
            mask.sum(
                dim=1
            ).clamp(
                min=1e-9
            )
        )


        mean_embedding = (
            sum_embeddings
            / sum_mask
        )


        normalized = torch.nn.functional.normalize(
            mean_embedding,
            p=2,
            dim=1
        )


        result = (
            normalized
            .cpu()
            .numpy()
            .astype(
                "float32"
            )
        )


    # Explicit cleanup
    del inputs
    del outputs

    return result


# ============================================================
# 10. WARMUP
# ============================================================

print()
print("Running embedding warmup...")

try:

    _ = encode_query(
        "warmup"
    )

    gc.collect()

    print(
        "Embedding warmup complete."
    )

except Exception as e:

    print(
        f"Warmup failed: {e}"
    )


print()
print("=" * 60)
print("RAG ENGINE READY")
print("=" * 60)
print()


# ============================================================
# 11. LANGUAGE DETECTION
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


def normalize_lang_code(
    code: Optional[str]
):

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


def classify_indic_language(
    text: str,
    hint_lang: Optional[str] = None
):

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

        if (
            "ळ" in cleaned
        ):

            return "mr"


        if any(
            w in words
            for w in [
                "काय",
                "म्हणजे",
                "कसा",
                "आहे",
                "नाही"
            ]
        ):

            return "mr"


        if (
            "ः" in cleaned
            or any(
                w in words
                for w in [
                    "किमिति",
                    "अस्ति",
                    "भवति"
                ]
            )
        ):

            return "sa"


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


    return "en"


# ============================================================
# 12. RETRIEVAL
# ============================================================

def retrieve_passages(
    query: str,
    top_k: int = 3,
    target_lang: Optional[str] = None
):

    t0 = time.perf_counter()


    effective_lang = (
        classify_indic_language(
            query,
            target_lang
        )
    )


    q_emb = encode_query(
        query
    )


    # Search only top 50 instead of 100
    search_k = min(
        50,
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

        if idx < 0:
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

            "language":
                doc_lang
                or effective_lang,

            "query_id":
                doc.get(
                    "query_id"
                ),

            "text":
                doc.get(
                    "text",
                    ""
                )
        }


        if (
            doc_lang
            == effective_lang
        ):

            lang_matches.append(
                item
            )


        global_matches.append(
            item
        )


    # Prefer requested language
    if lang_matches:

        results = lang_matches[
            :top_k
        ]

    else:

        results = global_matches[
            :top_k
        ]


    ret_time = (
        time.perf_counter()
        - t0
    ) * 1000.0


    print(
        f"[RAG] "
        f"Query='{query[:40]}' "
        f"Lang={effective_lang} "
        f"Results={len(results)} "
        f"Latency={ret_time:.2f}ms"
    )


    return (
        results,
        ret_time,
        effective_lang
    )


# ============================================================
# 13. REQUEST MODEL
# ============================================================

class QueryRequest(BaseModel):

    query: str

    language: Optional[str] = "auto"


# ============================================================
# 14. HEALTH CHECK
# ============================================================

@app.get("/")
def health_check():

    return {

        "status": "online",

        "service":
            "RAG-in-Goa Multilingual RAG",

        "vectors_indexed":
            index.ntotal,

        "embedding_model":
            MODEL_NAME,

        "endpoints": [
            "/api/ask",
            "/api/voice-ask"
        ]
    }


# ============================================================
# 15. TEXT QUERY
# ============================================================

@app.post("/api/ask")
def process_text_query(
    req: QueryRequest
):

    t_start = (
        time.perf_counter()
    )


    if not req.query.strip():

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )


    passages, ret_time, matched_lang = (
        retrieve_passages(
            req.query,
            top_k=3,
            target_lang=req.language
        )
    )


    total_time = (
        time.perf_counter()
        - t_start
    ) * 1000.0


    # Grounding threshold
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
                total_time < 200
        }


    answer = passages[0][
        "text"
    ]


    result = {

        "query":
            req.query,

        "language":
            matched_lang,

        "answer":
            answer,

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
            total_time < 200
    }


    # Cleanup temporary objects
    gc.collect()


    return result


# ============================================================
# 16. VOICE QUERY
# ============================================================

@app.post("/api/voice-ask")
async def process_voice_query(

    file: UploadFile = File(...),

    language: Optional[str] = Form("auto")
):

    t_start = (
        time.perf_counter()
    )


    # --------------------------------------------------------
    # Check API key
    # --------------------------------------------------------

    if not SARVAM_API_KEY:

        raise HTTPException(
            status_code=500,
            detail="SARVAM_API_KEY is not configured."
        )


    # --------------------------------------------------------
    # Read audio
    # --------------------------------------------------------

    audio_data = await file.read()

    if not audio_data:

        raise HTTPException(
            status_code=400,
            detail="Empty audio file."
        )


    # Limit audio size
    MAX_AUDIO_SIZE = 10 * 1024 * 1024

    if len(audio_data) > MAX_AUDIO_SIZE:

        raise HTTPException(
            status_code=413,
            detail="Audio file is too large. Maximum size is 10 MB."
        )


    orig_filename = (
        file.filename
        or "audio.wav"
    )


    # --------------------------------------------------------
    # Determine language
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

        resolved_hint = (
            normalize_lang_code(
                language
            )
        )


    # --------------------------------------------------------
    # Sarvam language
    # --------------------------------------------------------

    sarvam_lang_code = (
        SARVAM_BCP47_MAP.get(
            resolved_hint,
            "unknown"
        )
    )


    transcript = ""


    # --------------------------------------------------------
    # Determine file type
    # --------------------------------------------------------

    lower_filename = (
        orig_filename.lower()
    )


    if lower_filename.endswith(
        ".mp3"
    ):

        filename = "audio.mp3"

        content_type = (
            "audio/mpeg"
        )

    elif lower_filename.endswith(
        ".wav"
    ):

        filename = "audio.wav"

        content_type = (
            "audio/wav"
        )

    else:

        filename = "audio.wav"

        content_type = (
            "audio/wav"
        )


    # --------------------------------------------------------
    # Sarvam STT
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
            "[Sarvam] Sending audio..."
        )


        response = requests.post(

            url,

            headers=headers,

            files=files,

            data=data,

            timeout=10
        )


        print(
            f"[Sarvam] Status: "
            f"{response.status_code}"
        )


        if (
            response.status_code
            != 200
        ):

            print(
                "[Sarvam] Error:",
                response.text[:500]
            )

            raise HTTPException(
                status_code=502,
                detail=(
                    "Sarvam speech-to-text "
                    "request failed."
                )
            )


        response_json = (
            response.json()
        )


        transcript = (
            response_json
            .get(
                "transcript",
                ""
            )
            .strip()
        )


    except requests.Timeout:

        raise HTTPException(
            status_code=504,
            detail=(
                "Speech recognition "
                "timed out."
            )
        )


    except HTTPException:

        raise


    except Exception as e:

        print(
            "[STT Error]",
            repr(e)
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Speech recognition "
                "service failed."
            )
        )


    finally:

        # Release audio memory
        del audio_data

        gc.collect()


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


    print(
        f"[Sarvam STT] "
        f"Transcript: {transcript}"
    )


    # --------------------------------------------------------
    # Detect language
    # --------------------------------------------------------

    final_lang = (
        classify_indic_language(
            transcript,
            resolved_hint
        )
    )


    # --------------------------------------------------------
    # RAG retrieval
    # --------------------------------------------------------

    try:

        passages, ret_time, matched_lang = (
            retrieve_passages(
                transcript,
                top_k=3,
                target_lang=final_lang
            )
        )


        total_time = (
            time.perf_counter()
            - t_start
        ) * 1000.0


        if (
            not passages
            or passages[0]["score"] < 0.35
        ):

            answer = (
                "The query is outside "
                "the verified dataset "
                "knowledge base."
            )

            grounded = False

            passages = []


        else:

            answer = passages[0][
                "text"
            ]

            grounded = True


        result = {

            "query":
                transcript,

            "language":
                matched_lang,

            "answer":
                answer,

            "grounded":
                grounded,

            "passages":
                passages,

            "transcribed_text":
                transcript,

            "detected_language":
                final_lang,

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

            "audio_pipeline_total_ms":
                round(
                    total_time,
                    2
                ),

            "passed_target_200ms":
                total_time < 200
        }


        return result


    finally:

        gc.collect()