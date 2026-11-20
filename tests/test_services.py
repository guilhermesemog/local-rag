import pytest

from errors import EmptyDocumentError, LLMProviderError
from llm.dummy import DummyLLM
from services import LLMService
from db.vector_db import SearchResult


class FakeDB:
    def __init__(self, empty=False):
        self.empty = empty

    def is_empty(self):
        return self.empty

    def search(self, *_args, **_kwargs):
        return [
            SearchResult("context", {"document_name": "sample.pdf", "page": 2}, 1.0)
        ]


class FakeEngine:
    def generate_embeddings(self, _texts):
        return [[1.0] * 384]


def test_llm_service_empty_db():
    with pytest.raises(EmptyDocumentError):
        LLMService(DummyLLM(), FakeEngine(), FakeDB(empty=True)).answer("question")


def test_llm_service_answers_with_context():
    result = LLMService(DummyLLM(), FakeEngine(), FakeDB()).answer("question")
    assert result["results"] == "This is a dummy response."
    assert result["sources"][0]["document_name"] == "sample.pdf"
    assert result["sources"][0]["page"] == 2


def test_search_result_normalizes_json_string_metadata():
    result = SearchResult("context", '{"document_name": "old.pdf", "page": 1}', 0.9)
    assert result.metadata == {"document_name": "old.pdf", "page": 1}


def test_llm_provider_failure_is_mapped():
    class BrokenLLM:
        def generate_text(self, _prompt):
            raise RuntimeError("secret")

    with pytest.raises(LLMProviderError):
        LLMService(BrokenLLM(), FakeEngine(), FakeDB()).answer("question")
