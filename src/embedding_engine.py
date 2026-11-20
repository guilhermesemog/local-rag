from sentence_transformers import SentenceTransformer
from functools import lru_cache
import logging

from errors import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        self.model_name = model_name
        if not device:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(model_name, device=device)
        logger.info("Loaded embedding model on device %s", device)

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                batch_size=32,
                normalize_embeddings=True,
            )

            if embeddings.shape[1] != 384:
                raise ValueError(
                    f"Unexpected embedding dimension: {embeddings.shape[1]}"
                )

        except Exception as e:
            logger.exception("Embedding generation failed")
            raise EmbeddingError() from e

        return embeddings.tolist()


@lru_cache(maxsize=1)
def get_embedding_engine() -> EmbeddingEngine:
    return EmbeddingEngine()
