import pytest

from db.vector_db import SearchResult
from errors import (
    EmptyDocumentError,
    IndexCompatibilityError,
    LLMProviderError,
    NoMatchingDocumentsError,
)
from llm.dummy import DummyLLM
from services import LLMService


class FakeDB:
    def __init__(self, empty=False, results=None):
        self.empty = empty
        self.results = results
        self.searches = []

    def is_empty(self):
        return self.empty

    def ensure_embedding_identity(self, _model_name, _dimension):
        return None

    def search(self, _vector, limit=None, document_ids=None):
        self.searches.append(document_ids)
        if self.results is not None:
            return self.results
        return [
            SearchResult("context", {"document_name": "sample.pdf", "page": 2}, 1.0)
        ]


class FakeEngine:
    model_name = "test-embedding"
    dimension = 384

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


def test_query_rejects_unknown_legacy_index():
    class LegacyDB(FakeDB):
        def ensure_embedding_identity(self, _model_name, _dimension):
            raise IndexCompatibilityError()

    with pytest.raises(IndexCompatibilityError):
        LLMService(DummyLLM(), FakeEngine(), LegacyDB()).answer("question")


def test_search_result_normalizes_json_string_metadata():
    result = SearchResult("context", '{"document_name": "old.pdf", "page": 1}', 0.9)
    assert result.metadata == {"document_name": "old.pdf", "page": 1}


def test_llm_provider_failure_is_mapped_without_logging_provider_content(caplog):
    class BrokenLLM:
        def generate_text(self, _prompt):
            raise RuntimeError("secret")

    with pytest.raises(LLMProviderError):
        LLMService(BrokenLLM(), FakeEngine(), FakeDB()).answer("question")
    assert "secret" not in caplog.text


def test_llm_service_passes_document_filter_and_history_to_prompt():
    class RecordingLLM:
        def __init__(self):
            self.prompt = None

        def generate_text(self, prompt):
            self.prompt = prompt
            return "answer"

    database = FakeDB()
    llm = RecordingLLM()
    service = LLMService(llm, FakeEngine(), database)

    service.answer(
        "current question",
        document_ids=["doc-a"],
        history=[{"question": "previous question", "answer": "previous answer"}],
    )

    assert database.searches == [["doc-a"]]
    assert "previous question" in llm.prompt
    assert "previous answer" in llm.prompt
    assert "current question" in llm.prompt


def test_filtered_query_without_matches_raises_safe_error():
    with pytest.raises(NoMatchingDocumentsError):
        LLMService(
            DummyLLM(),
            FakeEngine(),
            FakeDB(results=[]),
        ).answer("question", document_ids=["missing"])
