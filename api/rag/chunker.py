# api/rag/chunker.py
from __future__ import annotations
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)
