"""Real Hugging Face BGE models; downloads once, then uses the local cache."""

import numpy as np


class BGEEmbedding:
    def __init__(self, settings):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(settings.embedding_model, device=settings.model_device)
        self.model.max_seq_length = settings.model_max_length
        self.batch_size = settings.embedding_batch_size
        self.dimensions = self.model.get_sentence_embedding_dimension()

    def encode(self, texts):
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True
        )


class BGEReranker:
    def __init__(self, settings):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            settings.reranker_model,
            device=settings.model_device,
            max_length=settings.model_max_length,
        )

    def score(self, query, chunks):
        return np.asarray(self.model.predict([(query, c.text) for c in chunks])).reshape(-1)
