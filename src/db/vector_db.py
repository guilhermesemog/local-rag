import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
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

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        """Return matches using the shared SearchResult contract."""

    @abstractmethod
    def list_documents(self) -> list[dict[str, Any]]:
        """Return indexed documents with chunk and page counts."""

    @abstractmethod
    def document_exists(self, document_id: str) -> bool:
        pass

    @abstractmethod
    def delete_documents(self, document_ids: list[str]) -> int:
        """Delete chunks for the given documents and return the row count."""

    @abstractmethod
    def replace_documents(self, document_id: str, data: list[tuple]) -> int:
        """Replace one document's chunks atomically and return the old count."""

    @abstractmethod
    def delete_all_documents(self):
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def create_job(self, job: dict[str, Any]):
        pass

    @abstractmethod
    def get_job(self, job_id: str) -> dict[str, Any] | None:
        pass

    @abstractmethod
    def list_jobs(self, status: str | None = None) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def claim_job(self, job_id: str) -> bool:
        pass

    @abstractmethod
    def update_job(self, job_id: str, **changes):
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        pass
