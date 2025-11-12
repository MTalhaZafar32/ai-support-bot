# rag/utils.py
from pathlib import Path

def infer_title_from_text_or_name(text: str, fallback_name: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return Path(fallback_name).stem.replace("-", " ").replace("_", " ").strip().title()

def read_text_file(path: str) -> tuple[str, str]:
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() not in {".md", ".txt"}:
        raise ValueError("Only .md and .txt are supported (for now).")
    text = p.read_text(encoding="utf-8", errors="ignore")
    return text, str(p)
