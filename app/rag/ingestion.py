"""Offline, page-aware ingestion. No OCR: image-only PDF pages fail explicitly."""

import hashlib
import re
from pathlib import Path

from app.rag.documents import DocumentChunk


def load_chunks(path: Path, size: int = 1000, overlap: int = 150):
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < chunk size")
    chunks = []
    for file in sorted(path.rglob("*")):
        if file.suffix.lower() not in {".md", ".pdf"}:
            continue
        source = file.relative_to(path).as_posix()
        if file.suffix.lower() == ".pdf":
            import pymupdf

            with pymupdf.open(file) as doc:
                pages = [(i + 1, page.get_text(sort=True)) for i, page in enumerate(doc)]
            if any(not text.strip() for _, text in pages):
                raise ValueError(f"{source}: empty/image-only page; OCR required before ingestion")
        else:
            pages = [(0, file.read_text(encoding="utf-8"))]
        for page, text in pages:
            text = re.sub(r"[ \t]+", " ", text.replace("\r", "")).strip()
            start = 0
            while start < len(text):
                end = min(start + size, len(text))
                if end < len(text):
                    boundary = max(
                        text.rfind("\n", start + size // 2, end),
                        text.rfind(". ", start + size // 2, end),
                    )
                    if boundary > start:
                        end = boundary + 1
                body = text[start:end].strip()
                key = hashlib.sha256(f"{source}:{page}:{start}:{body}".encode()).hexdigest()
                if body:
                    chunks.append(DocumentChunk(key, source, body, page))
                if end == len(text):
                    break
                start = max(start + 1, end - overlap)
    if not chunks:
        raise ValueError(f"No PDF/Markdown text found in {path}")
    return chunks
