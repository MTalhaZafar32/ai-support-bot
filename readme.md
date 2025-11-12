
# AI Support Bot — Guide

This document consolidates setup, ingestion, prompts, and operations for the AminQa RAG bot. It replaces scattered notes and keeps everything current.

## Table of Contents

* [Quick Reference – KB & Ingestion](#quick-reference--kb--ingestion)

  * [Where to Put Files](#where-to-put-files)
  * [Supported Formats](#supported-formats)
  * [Batch Ingest (CLI)](#batch-ingest-cli)
  * [Manual Ingest (API)](#manual-ingest-api)
* [1. Setup](#1-setup)
* [2. Configure](#2-configure)
* [3. Start Services](#3-start-services)
* [4. Load Knowledge Base](#4-load-knowledge-base)
* [5. Verify & Test](#5-verify--test)

  * [Connectivity Pings](#connectivity-pings)
  * [Ask a Question](#ask-a-question)
  * [Search Only (Optional)](#search-only-optional)
* [Architecture](#architecture)
* [Endpoints](#endpoints)
* [CLI Tools](#cli-tools)
* [Prompts & Answer Policy](#prompts--answer-policy)
* [Quality & Scoring (Light)](#quality--scoring-light)
* [Operations](#operations)

  * [Switch Embedding Model](#switch-embedding-model)
  * [Reset / Rebuild Collection](#reset--rebuild-collection)
  * [Clean Bad Local Model Caches](#clean-bad-local-model-caches)
* [Troubleshooting](#troubleshooting)
* [File Layout](#file-layout)
* [Quick Start (10 minutes)](#quick-start-10-minutes)

---

## Quick Reference – KB & Ingestion

### Where to Put Files

```
kb/   ← add .md / .txt here (recommended)
```

### Supported Formats

* **Ingest script:** `.md`, `.txt` (default)
* You can extend to PDFs later by adding a PDF → text step.

### Batch Ingest (CLI)

```bash
# from repo root
source api/.venv/bin/activate
export PYTHONPATH=$(pwd)

python -m api.scripts.ingest_kb kb \
  --collection kb_en \
  --exts ".md,.txt" \
  --batch-size 128
# → prints: Files: N, Chunks: M
```

### Manual Ingest (API)

```bash
# POST raw text (dev only)
curl -X POST http://localhost:8010/ingest \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"doc_1","text":"your raw text here"}'
```

---

## 1. Setup

```bash
cd ai-support-bot
python -m venv api/.venv
source api/.venv/bin/activate
pip install -r requirements.txt
```

**Ollama**

```bash
# install Ollama, then:
ollama pull phi3:mini
# server listens at http://localhost:11434
```

**Qdrant (Docker)**

```bash
docker compose up -d
# Qdrant: http://localhost:6333
```

---

## 2. Configure

Edit `.env` in project root:

```ini
EMB_PATH=sentence-transformers/all-MiniLM-L6-v2
EMB_DIM=384

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=kb_en

OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini

HOST=0.0.0.0
PORT=8010
```

> **Important:** `EMB_DIM` must match the embedding model.

---

## 3. Start Services

```bash
export PYTHONPATH=$(pwd)
uvicorn api.main:app --host 0.0.0.0 --port 8010 --reload
```

---

## 4. Load Knowledge Base

Put your tech-company docs in `kb/` (CRM, customer platform, cloud, data engineering/Power BI, web/app dev, SLAs, overview, etc.) as Markdown/text, then:

```bash
python -m api.scripts.ingest_kb kb --collection kb_en --exts ".md,.txt" --batch-size 128
```

---

## 5. Verify & Test

### Connectivity Pings

```bash
curl -sS http://localhost:8010/health
curl -sS http://localhost:8010/ping/qdrant
curl -sS http://localhost:8010/ping/ollama
curl -sS http://localhost:8010/ping/embeddings
```

### Ask a Question

```bash
curl -sS -X POST http://localhost:8010/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"What services does AminQa provide in cloud infrastructure?","top_k":4}'
```

### Search Only (Optional)

If you add `/search` in your API later, you can inspect retrieval without generation.

---

## Architecture

```
[ Streamlit UI ]  (optional)
        │ HTTP
        ▼
     [ FastAPI ]  api/main.py
        │
        ├─ chunk → embed → upsert      (api/rag/*.py)
        │      └─ SentenceTransformers: all-MiniLM-L6-v2 (384)
        ├─ vector search               Qdrant (Docker) @ :6333
        └─ generate grounded answer    Ollama (phi3:mini) @ :11434
```

---

## Endpoints

* `GET /` – returns active config (qdrant url, models, dims)
* `GET /health` – liveness
* `GET /ping/qdrant` – lists collections
* `GET /ping/ollama` – quick “pong” test
* `GET /ping/embeddings` – returns embedding vector dimension
* `POST /ingest` – dev helper to index a single text payload

  * body: `{"doc_id": "...", "text": "..." }`
* `POST /ask`

  * body: `{"query": "...", "top_k": 4}`
  * returns: `{"answer": "...", "sources": [{"doc_id":..., "chunk":..., "score":...}, ...]}`

---

## CLI Tools

* **Ingest folder**
  `python -m api.scripts.ingest_kb <kb_dir> --collection kb_en --exts ".md,.txt" --batch-size 128`

> The script handles: file walk → chunk → embed → Qdrant upsert (with `doc_id`, `chunk_id`, `text` payload).

---

## Prompts & Answer Policy

* System message used in `/ask`:

  * “Use **only** provided context. If not present, say **I don’t know**.”
* Context construction:

  * Top-k chunks joined as bullet points; concise to keep prompt small.
* You can customize prompt templates in `api/rag/prompts.py` (already scaffolded).

---

## Quality & Scoring (Light)

* Current API returns raw answer + citations.
* A simple “retrieval_avg_score” or heuristic scoring can be added in the UI layer:

  * Mean of Qdrant scores (COSINE)
  * Number of distinct docs in top-k
  * Penalize uncertainty phrases
* Keep it minimal until you need gating.

---

## Operations

### Switch Embedding Model

1. Stop API.
2. Update `.env`:

   * `EMB_PATH=...` and matching `EMB_DIM=...`
3. **Recreate collection** (see below) and **re-ingest**.

### Reset / Rebuild Collection

* If dimensions changed or you want a clean slate:

  * Stop API.
  * Either delete the collection via Qdrant UI/API **or** remove local volume:

    ```bash
    docker compose down
    rm -rf qdrant_storage
    docker compose up -d
    ```
  * Start API; run ingest again.

### Clean Bad Local Model Caches

If a model download was partial/corrupt:

```bash
rm -rf models/<folder>
rm -rf ~/.cache/huggingface/hub/models--*<name>*
rm -rf ~/.cache/torch/sentence_transformers/*<name>*
```

---

## Troubleshooting

* **`/favicon.ico` 404 in logs**
  Browser asks for favicon; safe to ignore.

* **Port already in use**
  Change `--port` or free it:
  `lsof -i :8010 | awk 'NR>1 {print $2}' | xargs -r kill -9`

* **Embedding dim mismatch**
  Symptoms: Qdrant upsert/search errors.
  Fix: ensure `EMB_DIM` matches `EMB_PATH`, rebuild collection, re-ingest.

* **Slow first answer**
  First Ollama call may be cold; subsequent calls are faster. Keep model warm.

* **Nothing retrieved**
  Check that ingest succeeded (non-zero chunks) and queries match KB language.

---

## File Layout

```
ai-support-bot/
├─ api/
│  ├─ main.py                  # FastAPI
│  ├─ settings.py              # pydantic-settings (reads ../.env)
│  ├─ rag/
│  │  ├─ chunker.py            # chunking
│  │  ├─ embed.py              # embedder + helpers
│  │  ├─ vector.py             # Qdrant helpers (ensure collection)
│  │  ├─ prompts.py            # prompt strings (customize here)
│  │  └─ utils.py
│  └─ scripts/
│     └─ ingest_kb.py          # CLI ingestion
├─ kb/                         # your docs (.md/.txt)
├─ qdrant_storage/             # Docker volume for Qdrant
├─ .env
├─ requirements.txt
└─ docker-compose.yml          # Qdrant
```

---

## Quick Start (10 minutes)

```bash
# 1) Python deps
python -m venv api/.venv
source api/.venv/bin/activate
pip install -r requirements.txt

# 2) Services
docker compose up -d            # Qdrant
ollama pull phi3:mini           # LLM

# 3) Configure
# edit .env as shown above

# 4) Run API
export PYTHONPATH=$(pwd)
uvicorn api.main:app --host 0.0.0.0 --port 8010 --reload

# 5) Ingest KB
python -m api.scripts.ingest_kb kb --collection kb_en --exts ".md,.txt" --batch-size 128

# 6) Ask
curl -sS -X POST http://localhost:8010/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"Give me an overview of AminQa services","top_k":4}'
```
