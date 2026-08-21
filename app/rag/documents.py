from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source: str
    text: str


def load_markdown_chunks(path: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for file in sorted(path.glob("*.md")):
        text = file.read_text(encoding="utf-8")
        blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
        for idx, block in enumerate(blocks):
            chunks.append(DocumentChunk(chunk_id=f"{file.stem}-{idx}", source=file.name, text=block))
    return chunks
