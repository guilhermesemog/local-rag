import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def sample_pdf():
    return Path(__file__).resolve().parents[1] / "samples" / "sample.pdf"


@pytest.fixture
def sqlite_db(tmp_path):
    from db.sqlite_db import SQLiteVectorDB

    database = SQLiteVectorDB(str(tmp_path / "test.db"))
    yield database
    database.close()


@pytest.fixture
def fake_embeddings():
    return [[1.0] + [0.0] * 383]


@pytest.fixture
def postgres_db():
    if os.getenv("LOCAL_RAG_TEST_POSTGRES") != "1":
        pytest.skip("Set LOCAL_RAG_TEST_POSTGRES=1 and configure test database variables")
    from db.postgres_db import PostgresVectorDB

    database = PostgresVectorDB()
    yield database
    database.delete_all_documents()
    database.close()


@pytest.fixture
def api_client(monkeypatch, sqlite_db):
    from fastapi.testclient import TestClient
    import app
    from llm.dummy import DummyLLM

    monkeypatch.setattr(app, "llm", DummyLLM())
    class FakeEmbeddingEngine:
        def generate_embeddings(self, _texts):
            return [[1.0] * 384]

    monkeypatch.setattr(app, "get_embedding_engine", lambda: FakeEmbeddingEngine())
    services = __import__("services")
    monkeypatch.setattr(app, "DBService", lambda: services.DBService(db=sqlite_db))
    monkeypatch.setattr(
        app,
        "LLMService",
        lambda llm, embedding_engine=None: services.LLMService(
            llm, embedding_engine=embedding_engine, db=sqlite_db
        ),
    )
    return TestClient(app.app)
