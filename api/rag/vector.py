# api/rag/vector.py
from __future__ import annotations
from typing import List, Dict
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from api.settings import settings

_client: QdrantClient | None = None

def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.QDRANT_URL)
    return _client

def ensure_collection(collection_name: str, dim: int):
    q = client()
    existing = [c.name for c in q.get_collections().collections]
    if collection_name not in existing:
        q.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

def upsert_chunks(collection_name: str, points: List[Dict]):
    """
    points: [{'vector': [...], 'payload': {...}}, ...]
    """
    qpoints = [
        PointStruct(id=str(uuid.uuid4()), vector=p["vector"], payload=p["payload"])
        for p in points
    ]
    client().upsert(collection_name=collection_name, points=qpoints, wait=True)
