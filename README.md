Voice-Enabled Multilingual Indic RAG Engine

A high-performance, sub-200ms Voice-Enabled Multilingual Retrieval-Augmented Generation (RAG) system built for Indic languages using FAISS, Sarvam AI STT, FastAPI, and React.

---

## 1. Tech Stack Overview

- Backend Engine: FastAPI, Uvicorn, Python 3.11+
- Vector Database: FAISS (IndexFlatIP, 384-dimensional dense inner product search)
- Embedding Model: intfloat/multilingual-e5-small (normalized embeddings via sentence-transformers)
- Speech-to-Text (STT): Sarvam AI (saaras:v3 API with native Indic script transcription)
- Frontend: React, Vite, Tailwind CSS, Lucide Icons
- Data Processing & Streaming: pyarrow, huggingface_hub, datasets

---

## 2. Dataset Processing Pipeline

- Dataset Source: ai4bharat/MSMARCO-XI (14 official Indic languages: Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Assamese, Kannada, Malayalam, Punjabi, Odia, Urdu, Nepali, Sanskrit + English).
- Streaming & Zero-Disk Overhead: Parquet validation splits were streamed using pyarrow batches and cached temporarily to prevent excessive disk utilization.
- Corpus Balancing: Extracted and normalized ~3,000 passages per language to build a balanced multilingual corpus of 44,948 indexed vectors.
- Prefix Formatting: All passage chunks were structured with the "passage: " prefix and all queries with "query: " to adhere to the E5 contrastive learning specification.

---

## 3. Architecture & End-to-End Workflow

[Voice Input (.wav/.mp3)] / [Text Input]
│
▼
Sarvam AI STT (saaras:v3)  ──► Native Indic Script Transcript
│
▼
Unicode Script Detection   ──► Auto-maps (hi, bn, ta, mr, gu, etc.)
│
▼
Safety Guardrail Check     ──► Blocks injections / malicious inputs
│
▼
Multilingual E5 Encoding   ──► PyTorch CPU inference (<25ms)
│
▼
FAISS IndexFlatIP Search   ──► Top-K Vector Cosine Retrieval (<10ms)
│
▼
Language Priority Filter   ──► Strictly isolates native language passages
│
▼
Grounding Guardrail        ──► Cosine score threshold check (>= 0.72)
│
▼
Structured JSON Response   ──► Answer + Sources + Latency Analytics


---

## 4. Implementation of All 6 PDF Evaluation Criteria

### 1. Vector Database Setup & Chunking Strategy
- Built an in-memory faiss.IndexFlatIP storing 44,948 normalized dense vectors mapped to multilingual_metadata.jsonl.
- Chunks are preserved with complete passage context, language tags, query mappings, and ground-truth relevance flags (is_selected).

### 2. Multilingual Semantic Search Harness
- Integrated intfloat/multilingual-e5-small supporting cross-lingual and monolingual queries across 14 Indic languages and English.
- Automatic Unicode range parsing maps input text to its native script and prioritizes target-language candidate passages during retrieval.

### 3. Speech-to-Text (STT) Integration
- Integrated Sarvam AI's saaras:v3 API on the /api/voice-ask endpoint.
- Configured with mode: "transcribe" to preserve native scripts (Devanagari, Bengali, Tamil, etc.) directly into the RAG pipeline.

### 4. Sub-200ms Latency Optimization
- P50 / P70 / P100 latency targets achieved via:
  - Restricting PyTorch CPU thread pool: torch.set_num_threads(4)
  - Zero-overhead inference mode: torch.inference_mode()
  - Model & FAISS warmup execution on application boot
- Retrieval latency consistently clocks between 30ms and 65ms, well below the 200ms threshold.

### 5. Grounding & Safety Guardrails
- Safety Guardrail: Sanitizes queries and blocks prompt injections, system prompt exfiltration, and SQL/code execution patterns with HTTP 400.
- Grounding Guardrail: Enforces a strict cosine similarity threshold (score >= 0.72) to block ungrounded responses and mitigate hallucinations.

### 6. Full-Stack Interface & Benchmarking
- Interactive FastAPI Swagger documentation available at /docs.
- React + Tailwind dashboard providing one-click voice recording, live transcription inspection, grounded source display, and latency monitoring.
- Automated benchmarking scripts (generate_final_report.py) to measure per-language Recall@5 and latency distributions.

---

## 5. Quick Start & Execution

### 1. Backend Setup
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Set Sarvam API Key
$env:SARVAM_API_KEY="your_sarvam_api_key"

# Run FastAPI Server
uvicorn app:app --reload --port 8000