from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from app.rag.documents import DocumentChunk
from app.rag.embeddings import LocalHashEmbedding

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    vector_score: float
    bm25_score: float


class HybridRetriever:
    def __init__(self, chunks: list[DocumentChunk], embedder: LocalHashEmbedding | None = None):
        self.chunks = chunks
        self.embedder = embedder or LocalHashEmbedding()
        self.embeddings = [self.embedder.embed(chunk.text) for chunk in chunks]
        self.tokens = [self._tokenize(chunk.text) for chunk in chunks]
        self.doc_freq = self._doc_freq()
        self.avg_doc_len = sum(len(tokens) for tokens in self.tokens) / max(len(self.tokens), 1)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_vector = self.embedder.embed(query)
        query_tokens = self._tokenize(query)

        vector_ranked = self._rank_vector(query_vector)
        bm25_ranked = self._rank_bm25(query_tokens)
        fused = self._rrf([vector_ranked, bm25_ranked])

        results: list[RetrievalResult] = []
        for idx, fused_score in fused[:top_k]:
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=fused_score,
                    vector_score=dict(vector_ranked).get(idx, 0.0),
                    bm25_score=dict(bm25_ranked).get(idx, 0.0),
                )
            )
        return results

    def _rank_vector(self, query_vector: np.ndarray) -> list[tuple[int, float]]:
        scored = []
        for idx, vector in enumerate(self.embeddings):
            scored.append((idx, float(np.dot(query_vector, vector))))
        return sorted(scored, key=lambda item: item[1], reverse=True)

    def _rank_bm25(self, query_tokens: list[str]) -> list[tuple[int, float]]:
        scored = []
        total_docs = max(len(self.tokens), 1)
        for idx, doc_tokens in enumerate(self.tokens):
            counts = Counter(doc_tokens)
            doc_len = len(doc_tokens)
            score = 0.0
            for token in query_tokens:
                if token not in counts:
                    continue
                df = self.doc_freq[token]
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                tf = counts[token]
                denom = tf + 1.2 * (1 - 0.75 + 0.75 * doc_len / max(self.avg_doc_len, 1))
                score += idf * (tf * 2.2 / denom)
            scored.append((idx, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)

    @staticmethod
    def _rrf(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
        fused: dict[int, float] = defaultdict(float)
        raw_scores: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            for rank, (idx, score) in enumerate(ranking, start=1):
                if score <= 0:
                    continue
                fused[idx] += 1 / (k + rank)
                raw_scores[idx] += score
        return sorted(fused.items(), key=lambda item: (item[1], raw_scores[item[0]]), reverse=True)

    def _doc_freq(self) -> dict[str, int]:
        freq: dict[str, int] = defaultdict(int)
        for tokens in self.tokens:
            for token in set(tokens):
                freq[token] += 1
        return freq

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())
