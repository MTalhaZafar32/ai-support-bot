# api/rag/embed.py
from __future__ import annotations
from typing import List
from sentence_transformers import SentenceTransformer

_embed: SentenceTransformer | None = None

def load_embedder(model_or_path: str) -> SentenceTransformer:
    """Create or reuse a single SentenceTransformer instance."""
    global _embed
    if _embed is None:
        _embed = SentenceTransformer(model_or_path)
    return _embed

def embed_texts(model: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    """Encode texts -> normalized vectors (as Python lists)."""
    return model.encode(texts, normalize_embeddings=True).tolist()
