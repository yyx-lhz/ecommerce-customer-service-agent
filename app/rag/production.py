from concurrent.futures import ThreadPoolExecutor

from app.rag.hybrid import RetrievalResult


def fuse(rankings, k=60):
    scores, chunks = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, (chunk, _) in enumerate(ranking, 1):
            key = chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            chunks[key] = chunk
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
    return [
        (chunks[key], score)
        for key, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


class ProductionRetriever:
    def __init__(self, settings, stores=None, embedder=None, reranker=None):
        from app.rag.models import BGEEmbedding, BGEReranker
        from app.rag.stores import SearchStores

        self.s = settings
        self.stores = stores or SearchStores(settings)
        self.stores.check()
        self.embedder = embedder or BGEEmbedding(settings)
        self.reranker = reranker or BGEReranker(settings)

    def search(self, query, top_k=5):
        if top_k <= 0 or not query.strip():
            return []
        if top_k > self.s.rerank_candidates:
            raise ValueError("top_k must not exceed rerank_candidates")
        vector = self.embedder.encode([query])[0]
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense = pool.submit(self.stores.dense_search, vector, self.s.dense_candidates)
            sparse = pool.submit(self.stores.sparse_search, query, self.s.sparse_candidates)
            dense, sparse = dense.result(), sparse.result()
        fused = fuse([dense, sparse], self.s.rrf_k)[: self.s.rerank_candidates]
        if not fused:
            return []
        scores = self.reranker.score(query, [c for c, _ in fused])
        ds = {c.chunk_id: s for c, s in dense}
        ss = {c.chunk_id: s for c, s in sparse}
        results = [
            RetrievalResult(
                c, float(score), ds.get(c.chunk_id, 0), ss.get(c.chunk_id, 0), rrf_score=rrf
            )
            for (c, rrf), score in zip(fused, scores, strict=True)
        ]
        return sorted(results, key=lambda r: (-r.score, -r.rrf_score, r.chunk.chunk_id))[:top_k]

    def close(self):
        self.stores.close()
