from dataclasses import asdict

from app.rag.documents import DocumentChunk

FIELDS = ["chunk_id", "source", "page", "text"]


class SearchStores:
    def __init__(self, settings):
        from elasticsearch import Elasticsearch
        from pymilvus import MilvusClient

        self.s = settings
        self.dense = MilvusClient(
            uri=settings.milvus_uri or f"http://{settings.milvus_host}:{settings.milvus_port}",
            token=settings.milvus_token,
            timeout=settings.search_timeout,
        )
        self.sparse = Elasticsearch(
            settings.elasticsearch_url, request_timeout=settings.search_timeout
        )

    def create(self, dimensions):
        from pymilvus import DataType

        # Refuse partial/repeated ingestion: choose a new version for every corpus build.
        if self.dense.has_collection(self.s.milvus_collection) or self.sparse.indices.exists(
            index=self.s.elasticsearch_index
        ):
            raise ValueError("Index already exists. Choose a new RAG_INDEX_VERSION for ingestion.")
        schema = self.dense.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimensions)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("source", DataType.VARCHAR, max_length=2048)
        schema.add_field("page", DataType.INT64)
        index = self.dense.prepare_index_params()
        index.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        self.dense.create_collection(
            self.s.milvus_collection, schema=schema, index_params=index, consistency_level="Strong"
        )
        self.sparse.indices.create(
            index=self.s.elasticsearch_index,
            settings={
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "similarity": {"policy_bm25": {"type": "BM25", "k1": 1.2, "b": 0.75}},
            },
            mappings={
                "_meta": {
                    "complete": False,
                    "embedding_model": self.s.embedding_model,
                    "model_max_length": self.s.model_max_length,
                },
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "text": {"type": "text", "analyzer": "standard", "similarity": "policy_bm25"},
                },
            },
        )

    def insert(self, chunks, vectors):
        from elasticsearch.helpers import bulk

        rows = [asdict(c) for c in chunks]
        self.dense.upsert(
            self.s.milvus_collection,
            [dict(row, vector=v.tolist()) for row, v in zip(rows, vectors, strict=True)],
        )
        bulk(
            self.sparse,
            (
                {"_index": self.s.elasticsearch_index, "_id": r["chunk_id"], "_source": r}
                for r in rows
            ),
            raise_on_error=True,
        )

    def finish(self):
        self.dense.flush(self.s.milvus_collection)
        self.sparse.indices.refresh(index=self.s.elasticsearch_index)
        self.sparse.indices.put_mapping(
            index=self.s.elasticsearch_index,
            body={
                "_meta": {
                    "complete": True,
                    "embedding_model": self.s.embedding_model,
                    "model_max_length": self.s.model_max_length,
                }
            },
        )

    def check(self):
        self.sparse.cluster.health(wait_for_status="yellow", timeout="20s")
        if not self.dense.has_collection(self.s.milvus_collection):
            raise RuntimeError("Milvus index missing; run ingestion")
        meta = self.sparse.indices.get_mapping(index=self.s.elasticsearch_index)[
            self.s.elasticsearch_index
        ]["mappings"].get("_meta", {})
        if (
            not meta.get("complete")
            or meta.get("embedding_model") != self.s.embedding_model
            or meta.get("model_max_length") != self.s.model_max_length
        ):
            raise RuntimeError("Incomplete ingestion or embedding configuration mismatch")
        count = self.sparse.count(index=self.s.elasticsearch_index)["count"]
        dense_count = self.dense.query(
            self.s.milvus_collection, filter="", output_fields=["count(*)"]
        )[0]["count(*)"]
        if not count or count != dense_count:
            raise RuntimeError("Empty/inconsistent indices; use a completed ingestion version")
        self.dense.load_collection(self.s.milvus_collection)

    def dense_search(self, vector, limit):
        hits = self.dense.search(
            self.s.milvus_collection,
            data=[vector.tolist()],
            limit=limit,
            output_fields=FIELDS,
            search_params={"metric_type": "COSINE", "params": {"ef": max(64, limit)}},
        )[0]
        return [(DocumentChunk(**h["entity"]), float(h["distance"])) for h in hits]

    def sparse_search(self, query, limit):
        hits = self.sparse.search(
            index=self.s.elasticsearch_index, size=limit, query={"match": {"text": query}}
        )["hits"]["hits"]
        return [(DocumentChunk(**h["_source"]), float(h["_score"])) for h in hits]

    def close(self):
        self.dense.close()
        self.sparse.close()
