import numpy as np
import pytest

from embedding_engine import EmbeddingEngine
from errors import EmbeddingError


class FakeModel:
    def __init__(self, values):
        self.values = values

    def encode(self, *_args, **_kwargs):
        return self.values


def test_embedding_engine_returns_vectors():
    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    engine.model = FakeModel(np.ones((1, 384)))
    assert engine.generate_embeddings(["text"]) == [[1.0] * 384]


def test_embedding_engine_maps_model_failure():
    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    engine.model = FakeModel(np.ones((1, 12)))
    with pytest.raises(EmbeddingError):
        engine.generate_embeddings(["text"])
