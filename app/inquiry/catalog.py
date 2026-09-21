"""Product knowledge base with local embedding-backed semantic retrieval."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.rag.embeddings import LocalHashEmbedding


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    aliases: list[str]
    description: str
    specifications: dict[str, str | int | float | bool]
    currency: str
    unit_price: float
    moq: int
    stock: int
    lead_time_days: int

    @property
    def searchable_text(self) -> str:
        specification_values = map(str, self.specifications.values())
        parts = [self.name, *self.aliases, self.description, *specification_values]
        return " ".join(parts)


@dataclass(frozen=True)
class ProductCandidate:
    product: Product
    score: float


class ProductKnowledgeBase:
    """A lightweight vector store; swap it for Chroma/Milvus behind this interface in production."""

    def __init__(self, products: list[Product], embedding: LocalHashEmbedding | None = None):
        self.products = products
        self.embedding = embedding or LocalHashEmbedding()
        self._vectors = [self.embedding.embed(item.searchable_text) for item in products]

    @classmethod
    def from_json(cls, path: Path) -> "ProductKnowledgeBase":
        raw_products = json.loads(path.read_text(encoding="utf-8"))
        return cls([Product(**item) for item in raw_products])

    def search(self, query: str, top_k: int = 3) -> list[ProductCandidate]:
        query_vector = self.embedding.embed(query)
        candidates = [
            ProductCandidate(product=product, score=float(np.dot(query_vector, vector)))
            for product, vector in zip(self.products, self._vectors, strict=True)
        ]
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]
