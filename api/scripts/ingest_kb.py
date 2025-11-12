# api/scripts/ingest_kb.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, List, Tuple
import typer

# our settings and rag helpers
from api.settings import settings
from api.rag.chunker import chunk_text
from api.rag.embed import load_embedder, embed_texts
from api.rag.vector import ensure_collection, upsert_chunks

app = typer.Typer(add_completion=False, help="Ingest local KB into Qdrant")


def _iter_files(root: Path, exts: Tuple[str, ...]) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _read_text_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


@app.command("load")
def load(
    kb_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder with KB files"),
    collection: str = typer.Option(settings.QDRANT_COLLECTION, "--collection", "-c"),
    exts: str = typer.Option(".md,.txt", "--exts", help="Comma-separated extensions to include"),
    batch_size: int = typer.Option(128, "--batch-size", "-b", help="Upsert batch size"),
):
    """
    Walk kb_dir → read text files → chunk → embed → upsert to Qdrant.
    """
    exts_tuple = tuple(s.strip().lower() for s in exts.split(",") if s.strip())
    typer.echo(f"KB path: {kb_dir} | exts: {exts_tuple} | collection: {collection}")

    # 1) make sure collection exists (dim must match current EMB_DIM)
    ensure_collection(collection_name=collection, dim=settings.EMB_DIM)

    # 2) load embedder once
    embedder = load_embedder(settings.EMB_PATH)

    total_files = 0
    total_chunks = 0
    batch_points: List[dict] = []

    for f in _iter_files(kb_dir, exts_tuple):
        total_files += 1
        doc_id = f.relative_to(kb_dir).as_posix()

        text = _read_text_file(f)
        chunks = chunk_text(text)
        vecs = embed_texts(embedder, chunks)

        # accumulate points
        for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
            batch_points.append(
                {
                    "vector": vec,
                    "payload": {"doc_id": doc_id, "chunk_id": i, "text": chunk},
                }
            )
        total_chunks += len(chunks)

        # flush in batches
        if len(batch_points) >= batch_size:
            upsert_chunks(collection_name=collection, points=batch_points)
            batch_points.clear()
            typer.echo(f"Upserted so far: {total_chunks} chunks")

    # flush remainder
    if batch_points:
        upsert_chunks(collection_name=collection, points=batch_points)

    typer.secho(f"Done. Files: {total_files}, Chunks: {total_chunks}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    # Allow running as a module or script
    app()
