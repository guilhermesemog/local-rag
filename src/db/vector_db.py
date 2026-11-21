from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from typing import Any


@dataclass(frozen=True)
class SearchResult:
    content: str
    metadata: dict[str, Any]
    similarity: float

    def __post_init__(self):
        metadata = self.metadata
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        object.__setattr__(self, "metadata", metadata)


class VectorDB(ABC):
    @abstractmethod
    def ensure_embedding_identity(self, model_name: str, dimension: int):
        """Validate or initialize the embedding identity for this index."""
        raise NotImplementedError

    @abstractmethod
    def validate_vector_dimension(self, vector: list[float]):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def create_table(self):
        pass

    @abstractmethod
    def insert_documents(self, data: list[tuple]):
        """Atomically persist (content, metadata, embedding) rows."""
        pass

    @abstractmethod
    def search(self, query_vector: list[float], limit: int) -> list[SearchResult]:
        """Return matches using the shared SearchResult contract."""
        pass

    @abstractmethod
    def delete_all_documents(self):
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass
