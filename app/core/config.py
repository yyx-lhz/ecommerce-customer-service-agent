from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    rag_backend: Literal["local", "production"] = "local"
    rag_index_version: str = Field(default="v1", pattern=r"^[a-z0-9_]+$")
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    model_device: str = "cpu"
    model_max_length: int = Field(default=1024, ge=32, le=8192)
    embedding_batch_size: int = Field(default=8, ge=1)
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0)
    dense_candidates: int = Field(default=20, ge=5, le=1000)
    sparse_candidates: int = Field(default=20, ge=5, le=1000)
    rerank_candidates: int = Field(default=20, ge=5, le=1000)
    rrf_k: int = Field(default=60, ge=1)
    search_timeout: float = Field(default=30, gt=0)
    milvus_uri: str = ""
    milvus_token: str = ""

    @property
    def milvus_collection(self):
        return f"knowledge_{self.rag_index_version}"

    @property
    def elasticsearch_index(self):
        return f"knowledge_{self.rag_index_version}"

    @model_validator(mode="after")
    def validate_chunks(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    app_env: str = "local"
    openai_api_key: str | None = None
    embedding_provider: str = "local"
    redis_url: str = "redis://localhost:6379/0"
    elasticsearch_url: str = "http://localhost:9200"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    use_redis: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
