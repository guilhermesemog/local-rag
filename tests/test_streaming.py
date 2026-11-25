import json

import pytest

from db.vector_db import SearchResult
from errors import LLMProviderError
from llm.dummy import DummyLLM
from services import LLMService


class FakeDB:
    def __init__(self, empty=False):
        self.empty = empty
        self.searches = []

    def is_empty(self):
        return self.empty

    def ensure_embedding_identity(self, _model_name, _dimension):
        return None

    def search(self, _vector, limit=None, document_ids=None):
        self.searches.append(document_ids)
        return [
            SearchResult(
                "context",
                {
                    "document_id": "doc-1",
                    "document_name": "sample.pdf",
                    "page": 2,
                    "chunk": 3,
                },
                0.87,
            )
        ]


class FakeEngine:
    model_name = "test-embedding"
    dimension = 384

    def generate_embeddings(self, _texts):
        return [[1.0] * 384]


def parse_sse(text):
    frames = []
    for block in text.split("\n\n"):
        event = "message"
        data_lines = []
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if data_lines:
            frames.append({"event": event, "data": json.loads("\n".join(data_lines))})
    return frames


def test_dummy_stream_matches_generate_text():
    llm = DummyLLM()

    assert "".join(llm.stream_text("prompt")) == llm.generate_text("prompt")


def test_answer_stream_emits_sources_then_tokens():
    database = FakeDB()
    service = LLMService(DummyLLM(), FakeEngine(), database)

    events = list(
        service.answer_stream(
            "question",
            document_ids=["doc-1"],
            history=[{"question": "earlier", "answer": "reply"}],
        )
    )

    assert database.searches == [["doc-1"]]
    assert events[0]["type"] == "sources"
    assert events[0]["sources"][0]["document_name"] == "sample.pdf"
    assert events[0]["sources"][0]["page"] == 2

    tokens = [event["text"] for event in events if event["type"] == "token"]
    assert "".join(tokens) == "This is a dummy response."


def test_answer_stream_maps_provider_errors_without_leaking_content(caplog):
    class BrokenLLM:
        def stream_text(self, _prompt):
            raise RuntimeError("secret")

    service = LLMService(BrokenLLM(), FakeEngine(), FakeDB())

    with pytest.raises(LLMProviderError):
        list(service.answer_stream("question"))
    assert "secret" not in caplog.text
