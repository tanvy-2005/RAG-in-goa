import os

# ============================================================
# MEMORY / CPU OPTIMIZATION
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

os.environ["ORT_INTRA_OP_NUM_THREADS"] = "1"
os.environ["ORT_INTER_OP_NUM_THREADS"] = "1"

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Prevent unnecessary tokenizer/model parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# IMPORTS
# ============================================================

import re
import json
import time
import statistics

from collections import deque
from typing import Optional, List, Dict, Any

import numpy as np
import faiss
import requests

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer


# ============================================================
# ENVIRONMENT
# ============================================================

SARVAM_API_KEY = os.getenv(
    "SARVAM_API_KEY",
    ""
).strip()


# ============================================================
# APPLICATION CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

INDEX_FILE = os.path.join(
    BASE_DIR,
    "multilingual.index"
)

METADATA_FILE = os.path.join(
    BASE_DIR,
    "multilingual_metadata.jsonl"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "intfloat/multilingual-e5-small"


# ============================================================
# RETRIEVAL CONFIG
# ============================================================

TOP_K = 3

# IMPORTANT:
# We only need a small candidate set because final output
# contains only TOP_K results.
SEARCH_K = 10

GROUNDING_THRESHOLD = 0.35

MAX_CACHE_SIZE = 50


# ============================================================
# LATENCY ANALYTICS
# ============================================================

LATENCY_WINDOW_SIZE = 500

latency_history = deque(
    maxlen=LATENCY_WINDOW_SIZE
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="RAG-in-Goa — Voice Enabled Multilingual RAG",
    description=(
        "HH Goa 2026 Task 2 submission. "
        "Voice → Sarvam STT → multilingual retrieval → "
        "grounded answer generation."
    ),
    version="41.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STARTUP
# ============================================================

print("=" * 70)
print("INITIALIZING RAG-IN-GOA MULTILINGUAL RAG ENGINE")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(INDEX_FILE):

    raise FileNotFoundError(
        f"FAISS index not found: {INDEX_FILE}"
    )


if not os.path.exists(METADATA_FILE):

    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_FILE}"
    )


# ============================================================
# LOAD FAISS
# ============================================================

print(
    f"Loading FAISS index from: {INDEX_FILE}"
)

index = faiss.read_index(
    INDEX_FILE
)

TOTAL_VECTORS = index.ntotal

print(
    f"Total vectors indexed: "
    f"{TOTAL_VECTORS:,}"
)

print(
    f"FAISS embedding dimension: "
    f"{index.d}"
)


# ============================================================
# METADATA SEEK MAP
# ============================================================

print(
    "Building metadata disk seek map..."
)

doc_offsets = []

with open(
    METADATA_FILE,
    "rb"
) as metadata_file:

    offset = 0

    for line in metadata_file:

        doc_offsets.append(
            offset
        )

        offset += len(line)


print(
    f"Metadata records: "
    f"{len(doc_offsets):,}"
)


# ============================================================
# VALIDATE INDEX / METADATA
# ============================================================

if TOTAL_VECTORS != len(doc_offsets):

    raise RuntimeError(
        "FAISS vector count does not match "
        "metadata record count. "
        f"Vectors={TOTAL_VECTORS}, "
        f"Metadata={len(doc_offsets)}"
    )


# ============================================================
# SMALL METADATA CACHE
# ============================================================

docs_cache: Dict[int, Dict[str, Any]] = {}


def get_metadata_by_id(
    doc_idx: int
):

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


        if len(docs_cache) >= MAX_CACHE_SIZE:

            oldest_key = next(
                iter(docs_cache)
            )

            del docs_cache[
                oldest_key
            ]


        docs_cache[
            doc_idx
        ] = data


        return data


    except Exception as e:

        print(
            f"Metadata read error: {e}"
        )

        return {}


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print()

print(
    f"Loading multilingual embedding model: "
    f"{MODEL_NAME}"
)


try:

    embedding_model = SentenceTransformer(
        MODEL_NAME,
        backend="onnx",
        model_kwargs={
            "file_name":
                "onnx/model_qint8_avx512_vnni.onnx"
        },
        device="cpu"
    )


    print(
        "Embedding model loaded successfully."
    )


except Exception as e:

    print(
        "FAILED TO LOAD EMBEDDING MODEL"
    )

    print(
        repr(e)
    )

    raise


# ============================================================
# IMPORTANT
#
# DO NOT run a model.encode() dimension test here.
#
# The model is known to produce 384-dimensional embeddings
# and the FAISS index is also 384-dimensional.
#
# Running another encode during startup wastes RAM/CPU.
# ============================================================


print(
    f"Embedding model dimension: {index.d}"
)


# ============================================================
# LANGUAGE ALIASES
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

    "konkani": "kok",
    "kok": "kok"
}


# ============================================================
# SARVAM LANGUAGE MAP
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
    "en": "en-IN",
    "kok": "kok-IN"
}


# ============================================================
# LANGUAGE NORMALIZATION
# ============================================================

def normalize_lang_code(
    code: Optional[str]
):

    if not code:

        return None


    c = (
        str(code)
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
# LANGUAGE DETECTION
# ============================================================

def classify_indic_language(
    text: str,
    hint_lang: Optional[str] = None
):

    normalized_hint = normalize_lang_code(
        hint_lang
    )


    if (
        normalized_hint
        and normalized_hint not in [
            "auto",
            "unknown",
            ""
        ]
    ):

        return normalized_hint


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
            word in words
            for word in [
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

        if "ळ" in cleaned:

            return "mr"


        if any(
            word in words
            for word in [
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
                word in words
                for word in [
                    "किमिति",
                    "अस्ति",
                    "भवति"
                ]
            )
        ):

            return "sa"


        # Nepali

        if any(
            phrase in cleaned
            for phrase in [
                "के हो",
                "हो निगम",
                "भनेको"
            ]
        ):

            return "ne"


        return "hi"


    return "en"


# ============================================================
# QUERY SANITIZATION
# ============================================================

def sanitize_query(
    query: str
):

    if query is None:

        return ""


    query = str(
        query
    ).strip()


    if len(query) > 1000:

        query = query[:1000]


    query = "".join(
        c
        for c in query
        if c.isprintable()
        or c.isspace()
    )


    return query.strip()


# ============================================================
# SAFETY GUARDRAIL
# ============================================================

UNSAFE_PATTERNS = [

    r"\bhow to make a bomb\b",
    r"\bhow to build a bomb\b",
    r"\bmake explosives\b",
    r"\bcreate malware\b",
    r"\bmake malware\b",
    r"\bsteal passwords\b",
    r"\bhack someone's account\b",
    r"\bhow to hack\b"

]


def safety_check(
    query: str
):

    q = query.lower()


    for pattern in UNSAFE_PATTERNS:

        if re.search(
            pattern,
            q
        ):

            return False


    return True


# ============================================================
# E5 QUERY EMBEDDING
# ============================================================

def encode_query(
    query: str
):

    formatted_query = (
        "query: "
        + query.strip()
    )


    start = time.perf_counter()


    # IMPORTANT:
    # batch_size=1 because we only embed one query.
    #
    # This avoids unnecessary memory allocation.

    embeddings = embedding_model.encode(

        [formatted_query],

        batch_size=1,

        normalize_embeddings=True,

        convert_to_numpy=True,

        show_progress_bar=False
    )


    embedding = np.asarray(
        embeddings[0],
        dtype=np.float32
    )


    # Ensure correct dimension

    if embedding.shape[0] != index.d:

        raise RuntimeError(
            "Embedding dimension mismatch: "
            f"model={embedding.shape[0]}, "
            f"FAISS={index.d}"
        )


    # Normalize

    norm = np.linalg.norm(
        embedding
    )


    if norm > 0:

        embedding /= norm


    elapsed = (
        time.perf_counter()
        - start
    ) * 1000


    return (
        embedding.reshape(
            1,
            -1
        ),
        elapsed
    )


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def remove_duplicate_results(
    results: List[Dict[str, Any]]
):

    seen = set()

    output = []


    for result in results:

        text = (
            result.get(
                "text",
                ""
            )
            .strip()
        )


        if not text:

            continue


        normalized = re.sub(
            r"\s+",
            " ",
            text.lower()
        )


        fingerprint = normalized[
            :500
        ]


        if fingerprint in seen:

            continue


        seen.add(
            fingerprint
        )


        output.append(
            result
        )


    return output


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_passages(
    query: str,
    top_k: int = TOP_K,
    target_lang: Optional[str] = None
):

    pipeline_start = time.perf_counter()


    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    effective_lang = classify_indic_language(
        query,
        target_lang
    )


    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    query_vector, embedding_ms = encode_query(
        query
    )


    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    search_start = time.perf_counter()


    scores, indices = index.search(
        query_vector,
        min(
            SEARCH_K,
            index.ntotal
        )
    )


    vector_search_ms = (
        time.perf_counter()
        - search_start
    ) * 1000


    language_matches = []

    global_matches = []


    # --------------------------------------------------------
    # Metadata filtering
    # --------------------------------------------------------

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


        doc_language = normalize_lang_code(
            str(
                doc.get(
                    "language",
                    ""
                )
            )
        )


        text = str(
            doc.get(
                "text",
                ""
            )
        ).strip()


        if not text:

            continue


        item = {

            "score":
                round(
                    float(score),
                    4
                ),

            "language":
                doc_language
                or effective_lang,

            "query_id":
                doc.get(
                    "query_id"
                ),

            "text":
                text
        }


        if (
            doc_language
            == effective_lang
        ):

            language_matches.append(
                item
            )


        global_matches.append(
            item
        )


    # --------------------------------------------------------
    # Language preference
    # --------------------------------------------------------

    if language_matches:

        candidates = language_matches

    else:

        candidates = global_matches


    # --------------------------------------------------------
    # Score filtering
    # --------------------------------------------------------

    candidates = [

        item

        for item in candidates

        if item["score"]
        >= GROUNDING_THRESHOLD

    ]


    # --------------------------------------------------------
    # Duplicate removal
    # --------------------------------------------------------

    candidates = remove_duplicate_results(
        candidates
    )


    # --------------------------------------------------------
    # Top K
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    results = candidates[
        :top_k
    ]


    total_ms = (
        time.perf_counter()
        - pipeline_start
    ) * 1000


    return {

        "results":
            results,

        "language":
            effective_lang,

        "embedding_ms":
            round(
                embedding_ms,
                2
            ),

        "vector_search_ms":
            round(
                vector_search_ms,
                2
            ),

        "retrieval_ms":
            round(
                total_ms,
                2
            )
    }


# ============================================================
# GROUNDED ANSWER
# ============================================================

def generate_grounded_answer(
    query: str,
    passages: List[Dict[str, Any]]
):

    if not passages:

        return {

            "answer":
                "The query is outside the "
                "verified dataset knowledge base.",

            "grounded":
                False
        }


    best = passages[0]


    score = float(
        best.get(
            "score",
            0
        )
    )


    if score < GROUNDING_THRESHOLD:

        return {

            "answer":
                "The query is outside the "
                "verified dataset knowledge base.",

            "grounded":
                False
        }


    text = (
        best.get(
            "text",
            ""
        ).strip()
    )


    if not text:

        return {

            "answer":
                "No grounded answer was found.",

            "grounded":
                False
        }


    sentences = re.split(
        r"(?<=[.!?।])\s+",
        text
    )


    if len(sentences) > 4:

        answer = " ".join(
            sentences[:4]
        ).strip()

    else:

        answer = text


    return {

        "answer":
            answer,

        "grounded":
            True
    }


# ============================================================
# REQUEST MODEL
# ============================================================

class QueryRequest(BaseModel):

    query: str

    language: Optional[str] = "auto"


# ============================================================
# LATENCY
# ============================================================

def record_latency(
    latency_ms: float
):

    latency_history.append(
        float(latency_ms)
    )


def percentile(
    values,
    p
):

    if not values:

        return 0.0


    ordered = sorted(
        values
    )


    index_position = (
        (len(ordered) - 1)
        * p
    )


    lower = int(
        index_position
    )


    upper = min(
        lower + 1,
        len(ordered) - 1
    )


    weight = (
        index_position
        - lower
    )


    return (
        ordered[lower]
        * (1 - weight)
        +
        ordered[upper]
        * weight
    )


# ============================================================
# ROOT HEALTH
# ============================================================

@app.get("/")
def health_check():

    return {

        "status":
            "online",

        "service":
            "RAG-in-Goa",

        "vectors_indexed":
            int(index.ntotal),

        "embedding_dimension":
            int(index.d),

        "embedding_model":
            MODEL_NAME,

        "embedding_backend":
            "sentence-transformers-onnx",

        "endpoints":
            [
                "/api/ask",
                "/api/voice-ask",
                "/api/analytics",
                "/api/health"
            ]
    }


# ============================================================
# HEALTH API
# ============================================================

@app.get("/api/health")
def api_health():

    return {

        "status":
            "healthy",

        "faiss":
            True,

        "vectors":
            int(index.ntotal),

        "dimension":
            int(index.d),

        "metadata_records":
            len(doc_offsets),

        "embedding_model":
            MODEL_NAME,

        "embedding_backend":
            "sentence-transformers-onnx",

        "sarvam_configured":
            bool(SARVAM_API_KEY)
    }


# ============================================================
# TEXT RAG
# ============================================================

@app.post("/api/ask")
def process_text_query(
    req: QueryRequest
):

    request_start = time.perf_counter()


    # --------------------------------------------------------
    # Sanitize
    # --------------------------------------------------------

    query = sanitize_query(
        req.query
    )


    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )


    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    if not safety_check(
        query
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "This query is not supported "
                "by the system safety policy."
            )
        )


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieval = retrieve_passages(

        query=query,

        top_k=TOP_K,

        target_lang=req.language
    )


    passages = retrieval[
        "results"
    ]


    # --------------------------------------------------------
    # Grounded answer
    # --------------------------------------------------------

    generated = generate_grounded_answer(

        query,

        passages
    )


    total_ms = (
        time.perf_counter()
        - request_start
    ) * 1000


    record_latency(
        total_ms
    )


    return {

        "query":
            query,

        "language":
            retrieval["language"],

        "answer":
            generated["answer"],

        "grounded":
            generated["grounded"],

        "passages":
            passages,

        "latency_ms":
            round(
                total_ms,
                2
            ),

        "retrieval_ms":
            retrieval["retrieval_ms"],

        "embedding_ms":
            retrieval["embedding_ms"],

        "vector_search_ms":
            retrieval["vector_search_ms"],

        "passed_target_200ms":
            total_ms < 200,

        "pipeline":
            [
                "input-validation",
                "language-detection",
                "e5-semantic-embedding",
                "faiss-vector-search",
                "metadata-language-filter",
                "duplicate-removal",
                "grounding-check",
                "extractive-grounded-answer"
            ]
    }


# ============================================================
# SARVAM STT
# ============================================================

def transcribe_with_sarvam(
    audio_data: bytes,
    filename: str,
    content_type: str,
    language: Optional[str]
):

    if not SARVAM_API_KEY:

        raise HTTPException(
            status_code=500,
            detail=(
                "SARVAM_API_KEY is not configured."
            )
        )


    normalized_language = normalize_lang_code(
        language
    )


    sarvam_language = (
        SARVAM_BCP47_MAP.get(
            normalized_language,
            "unknown"
        )
    )


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
            sarvam_language
    }


    start = time.perf_counter()


    try:

        response = requests.post(

            url,

            headers=headers,

            files=files,

            data=data,

            timeout=20
        )


    except requests.Timeout:

        raise HTTPException(

            status_code=504,

            detail=(
                "Sarvam speech recognition timed out."
            )
        )


    except requests.RequestException as e:

        print(
            "[Sarvam Request Error]",
            repr(e)
        )

        raise HTTPException(

            status_code=502,

            detail=(
                "Sarvam speech recognition failed."
            )
        )


    stt_ms = (
        time.perf_counter()
        - start
    ) * 1000


    if response.status_code != 200:

        print(
            "[Sarvam Error]",
            response.text[:500]
        )

        raise HTTPException(

            status_code=502,

            detail=(
                "Sarvam speech-to-text request failed."
            )
        )


    try:

        payload = response.json()

    except Exception:

        raise HTTPException(

            status_code=502,

            detail=(
                "Invalid response from Sarvam."
            )
        )


    transcript = (
        payload
        .get(
            "transcript",
            ""
        )
        .strip()
    )


    detected_language = (
        payload
        .get(
            "language_code"
        )
    )


    if not transcript:

        raise HTTPException(

            status_code=400,

            detail=(
                "Could not transcribe audio."
            )
        )


    return {

        "transcript":
            transcript,

        "detected_language":
            detected_language,

        "stt_ms":
            round(
                stt_ms,
                2
            )
    }


# ============================================================
# VOICE RAG
# ============================================================

@app.post("/api/voice-ask")
async def process_voice_query(

    file: UploadFile = File(...),

    language: Optional[str] = Form(
        "auto"
    )
):

    pipeline_start = time.perf_counter()


    # --------------------------------------------------------
    # Audio
    # --------------------------------------------------------

    audio_data = await file.read()


    if not audio_data:

        raise HTTPException(

            status_code=400,

            detail="Empty audio file."
        )


    MAX_AUDIO_SIZE = (
        10 * 1024 * 1024
    )


    if len(audio_data) > MAX_AUDIO_SIZE:

        raise HTTPException(

            status_code=413,

            detail=(
                "Audio file exceeds "
                "the 10 MB limit."
            )
        )


    original_filename = (
        file.filename
        or "audio.wav"
    )


    lower_filename = (
        original_filename.lower()
    )


    # --------------------------------------------------------
    # Audio type
    # --------------------------------------------------------

    if lower_filename.endswith(
        ".mp3"
    ):

        filename = "audio.mp3"

        content_type = "audio/mpeg"


    elif lower_filename.endswith(
        ".wav"
    ):

        filename = "audio.wav"

        content_type = "audio/wav"


    elif lower_filename.endswith(
        ".webm"
    ):

        filename = "audio.webm"

        content_type = "audio/webm"


    elif lower_filename.endswith(
        ".ogg"
    ):

        filename = "audio.ogg"

        content_type = "audio/ogg"


    elif lower_filename.endswith(
        ".m4a"
    ):

        filename = "audio.m4a"

        content_type = "audio/mp4"


    else:

        filename = "audio.wav"

        content_type = (
            file.content_type
            or "audio/wav"
        )


    # --------------------------------------------------------
    # STT
    # --------------------------------------------------------

    stt = transcribe_with_sarvam(

        audio_data=audio_data,

        filename=filename,

        content_type=content_type,

        language=language
    )


    transcript = stt[
        "transcript"
    ]


    detected_language = (

        normalize_lang_code(

            stt[
                "detected_language"
            ]
        )

        or

        classify_indic_language(

            transcript,

            language
        )
    )


    # Release audio memory

    del audio_data


    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    if not safety_check(
        transcript
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "The transcribed query is not "
                "supported by the system safety policy."
            )
        )


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieval = retrieve_passages(

        query=transcript,

        top_k=TOP_K,

        target_lang=detected_language
    )


    passages = retrieval[
        "results"
    ]


    # --------------------------------------------------------
    # Grounded answer
    # --------------------------------------------------------

    generated = generate_grounded_answer(

        transcript,

        passages
    )


    total_ms = (
        time.perf_counter()
        - pipeline_start
    ) * 1000


    record_latency(
        total_ms
    )


    return {

        "query":
            transcript,

        "transcribed_text":
            transcript,

        "language":
            retrieval["language"],

        "detected_language":
            detected_language,

        "answer":
            generated["answer"],

        "grounded":
            generated["grounded"],

        "passages":
            passages,

        "stt_ms":
            stt["stt_ms"],

        "retrieval_ms":
            retrieval["retrieval_ms"],

        "embedding_ms":
            retrieval["embedding_ms"],

        "vector_search_ms":
            retrieval["vector_search_ms"],

        "audio_pipeline_total_ms":
            round(
                total_ms,
                2
            ),

        "passed_target_200ms":
            total_ms < 200,

        "pipeline":
            [
                "voice-input",
                "Sarvam-Saaras-v3-STT",
                "language-detection",
                "E5-multilingual-embedding",
                "FAISS-semantic-retrieval",
                "metadata-aware-language-filter",
                "duplicate-removal",
                "grounding-check",
                "grounded-answer-generation"
            ]
    }


# ============================================================
# LATENCY ANALYTICS
# ============================================================

@app.get("/api/analytics")
def latency_analytics():

    values = list(
        latency_history
    )


    if not values:

        return {

            "samples":
                0,

            "message":
                "No requests measured yet."
        }


    under_200 = sum(

        1

        for value in values

        if value < 200
    )


    return {

        "samples":
            len(values),

        "p50_ms":
            round(
                percentile(
                    values,
                    0.50
                ),
                2
            ),

        "p70_ms":
            round(
                percentile(
                    values,
                    0.70
                ),
                2
            ),

        "p100_ms":
            round(
                max(values),
                2
            ),

        "average_ms":
            round(
                statistics.mean(values),
                2
            ),

        "minimum_ms":
            round(
                min(values),
                2
            ),

        "maximum_ms":
            round(
                max(values),
                2
            ),

        "target_ms":
            200,

        "samples_under_200ms":
            under_200,

        "percentage_under_200ms":
            round(
                (
                    under_200
                    / len(values)
                )
                * 100,
                2
            )
    }


# ============================================================
# ENGINE READY
# ============================================================

print()

print("=" * 70)
print("RAG-IN-GOA ENGINE READY")
print("=" * 70)

print(
    f"Vectors: {TOTAL_VECTORS:,}"
)

print(
    f"Metadata: {len(doc_offsets):,}"
)

print(
    f"Embedding: {MODEL_NAME}"
)

print(
    f"Embedding dimension: {index.d}"
)

print(
    "Embedding backend: SentenceTransformers ONNX"
)

print(
    "STT: Sarvam Saaras v3"
)

print(
    "Retrieval: FAISS + metadata-aware filtering"
)

print(
    "Guardrails: enabled"
)

print(
    "Latency analytics: enabled"
)

print("=" * 70)