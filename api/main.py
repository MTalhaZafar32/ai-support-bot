"""
AI Support Bot — RAG API (answer-only)
--------------------------------------
Stack:
- Qdrant (vector DB)
- SentenceTransformers (embeddings)  -> settings.EMB_PATH
- Ollama (LLM)                       -> settings.OLLAMA_MODEL

Public endpoints for the UI:
- GET  /health
- GET  /config
- POST /ask

Light ops:
- GET  /ping/qdrant
- GET  /ping/ollama
- GET  /ping/embeddings
- GET  /stats
"""
from __future__ import annotations

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from statistics import mean
import requests

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams  # PointStruct not needed now

# Prefer new package; fall back to community if needed
try:
    from langchain_ollama import OllamaLLM as _Ollama
except Exception:  # pragma: no cover
    from langchain_community.llms import Ollama as _Ollama

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
# top of file with the other imports
from .rag.prompts import build_prompt
from qdrant_client.http.models import SearchParams  # <-- for exact search (optional)
import time
from qdrant_client.http.models import SearchParams


from .settings import settings

# -----------------------------------------------------------------------------
# App + CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="AI Support Bot — RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Lazy singletons
# -----------------------------------------------------------------------------
_qdrant: Optional[QdrantClient] = None
_llm: Optional[_Ollama] = None
_embed: Optional[SentenceTransformer] = None


def qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=settings.QDRANT_URL)
    return _qdrant


def llm() -> _Ollama:
    global _llm
    if _llm is None:
        _llm = _Ollama(
            base_url=settings.OLLAMA_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0.2,
            num_ctx=4096)
    return _llm


def embedder() -> SentenceTransformer:
    global _embed
    if _embed is None:
        _embed = SentenceTransformer(settings.EMB_PATH)
    return _embed

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_collection() -> None:
    """Create the collection if missing (assumes EMB_DIM is correct)."""
    client = qdrant()
    existing = {c.name for c in client.get_collections().collections}
    if settings.QDRANT_COLLECTION in existing:
        return
    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=settings.EMB_DIM, distance=Distance.COSINE),
    )


def embed_texts(texts: List[str]) -> List[List[float]]:
    return embedder().encode(texts, normalize_embeddings=True).tolist()


def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


def estimate_tokens(s: str) -> int:
    return max(1, len(s.split()))

# -----------------------------------------------------------------------------
# Basics
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "msg": "AI Support Bot is running.",
        "qdrant": settings.QDRANT_URL,
        "collection": settings.QDRANT_COLLECTION,
        "ollama_model": settings.OLLAMA_MODEL,
        "emb_model": settings.EMB_PATH,
        "emb_dim": settings.EMB_DIM,
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/config")
def config():
    return {
        "qdrant": settings.QDRANT_URL,
        "collection": settings.QDRANT_COLLECTION,
        "ollama_model": settings.OLLAMA_MODEL,
        "emb_model": settings.EMB_PATH,
        "emb_dim": settings.EMB_DIM,
    }

# --- light ops (optional) ----------------------------------------------------
@app.get("/ping/qdrant")
def ping_qdrant():
    try:
        resp = qdrant().get_collections()
        return {"ok": True, "collections": [c.name for c in resp.collections]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/ping/ollama")
def ping_ollama():
    """
    Cheap liveness check: list local Ollama models (no text generation).
    Fast and reliable for a status light.
    """
    try:
        r = requests.get(f"{settings.OLLAMA_URL}/api/tags", timeout=3)
        r.raise_for_status()
        data = r.json()
        models = [m.get("name") or m.get("model") for m in data.get("models", [])]
        return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/ping/embeddings")
def ping_embeddings():
    try:
        vec = embedder().encode(["hello world"])[0]
        return {"ok": True, "dim": len(vec)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/stats")
def stats():
    """Exact point count + vector config of the active collection."""
    info = qdrant().get_collection(settings.QDRANT_COLLECTION)
    count = qdrant().count(collection_name=settings.QDRANT_COLLECTION, exact=True).count
    return {
        "collection": settings.QDRANT_COLLECTION,
        "points": count,
        "vector_size": info.config.params.vectors.size,
        "distance": info.config.params.vectors.distance,
    }

# -----------------------------------------------------------------------------
# Retrieval & Generation (public for UI) — with timings
# -----------------------------------------------------------------------------
@app.post("/ask")
def ask(
    query: str = Body(..., embed=True, description="User question"),
    top_k: int = Body(4, embed=True, description="Number of chunks to return"),
    score_threshold: float = Body(0.50, embed=True, description="Min cosine score to keep"),
    exact_search: bool = Body(False, embed=True, description="Use exhaustive search while KB is small"),
    max_per_doc: int = Body(2, embed=True, description="Limit chunks per document"),
):
    """
    RAG flow with timing metrics:
      1) Embed query
      2) Retrieve candidates from Qdrant (optionally exact search)
      3) Sort, threshold, de-duplicate, diversify
      4) Prompt LLM with clean context
      5) Return answer, citations, metrics (incl. timings), and retrieved snippets
    """
    try:
        t0 = time.perf_counter()
        ensure_collection()

        # 1) Embed + 2) Retrieve (time this block)
        t_ret_start = time.perf_counter()
        qvec = embed_texts([query])[0]
        search = qdrant().search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=qvec,
            limit=max(20, top_k * 5),  # overfetch; we'll curate later
            with_payload=True,
            search_params=SearchParams(exact=exact_search) if exact_search else None,
        )
        t_ret_end = time.perf_counter()

        # sort high→low and collect raw scores
        search = sorted(search, key=lambda h: float(h.score), reverse=True)
        raw_scores = [float(h.score) for h in search]

        # 3) threshold + dedupe + diversify
        strong = [h for h in search if float(h.score) >= float(score_threshold)]

        seen_pairs = set()
        per_doc = {}
        curated = []
        for h in strong:
            doc = h.payload.get("doc_id")
            chk = h.payload.get("chunk_id")
            key = (doc, chk)
            if key in seen_pairs:
                continue
            per_doc[doc] = per_doc.get(doc, 0) + 1
            if per_doc[doc] > int(max_per_doc):
                continue
            seen_pairs.add(key)
            curated.append(h)

        curated = curated[:top_k]
        contexts = [h.payload["text"] for h in curated]

        if not contexts:
            t_empty = time.perf_counter()
            return {
                "ok": True,
                "answer": "I don’t know from the knowledge base.",
                "sources": [],
                "metrics": {
                    "retrieval_avg_score": round(sum(raw_scores) / len(raw_scores), 4) if raw_scores else 0.0,
                    "context_tokens_est": 0,
                    "prompt_tokens_est": 0,
                    "timings_ms": {
                        "retrieval_ms": round((t_ret_end - t_ret_start) * 1000, 2),
                        "generation_ms": 0.0,
                        "server_total_ms": round((t_empty - t0) * 1000, 2),
                    },
                },
                "retrieved": [],
            }

        # 4) Prompt (via prompts.py)
        prompt = build_prompt(contexts, query)

        # 5) Generate (time it)
        t_gen_start = time.perf_counter()
        answer = llm().invoke(prompt).strip()
        t_gen_end = time.perf_counter()
        t1 = time.perf_counter()

        citations = [
            {"doc_id": h.payload.get("doc_id"), "chunk": h.payload.get("chunk_id"), "score": float(h.score)}
            for h in curated
        ]
        retrieved = [
            {
                "doc_id": h.payload.get("doc_id"),
                "chunk": h.payload.get("chunk_id"),
                "score": float(h.score),
                "text": h.payload.get("text"),
            }
            for h in curated
        ]
        metrics = {
            "retrieval_avg_score": round(sum(float(h.score) for h in curated) / len(curated), 4),
            "context_tokens_est": sum(estimate_tokens(c) for c in contexts),
            "prompt_tokens_est": estimate_tokens(prompt),
            "timings_ms": {
                "retrieval_ms": round((t_ret_end - t_ret_start) * 1000, 2),
                "generation_ms": round((t_gen_end - t_gen_start) * 1000, 2),
                "server_total_ms": round((t1 - t0) * 1000, 2),
            },
        }

        return {"ok": True, "answer": answer, "sources": citations, "metrics": metrics, "retrieved": retrieved}

    except Exception as e:
        return {"ok": False, "error": str(e)}
