import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


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
        pytest.skip(
            "Set LOCAL_RAG_TEST_POSTGRES=1 and configure test database variables"
        )
    from db.postgres_db import PostgresVectorDB

    database = PostgresVectorDB()
    yield database
    database.delete_all_documents()
    database.close()


@pytest.fixture
def api_client(monkeypatch, sqlite_db, tmp_path):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("LLM_MODEL", "dummy")
    import app
    import job_manager
    from llm.dummy import DummyLLM

    monkeypatch.setattr(app, "llm", DummyLLM())

    class FakeEmbeddingEngine:
        model_name = "test-embedding"
        dimension = 384

        def generate_embeddings(self, _texts):
            return [[1.0] * 384]

    monkeypatch.setattr(
        job_manager, "get_embedding_engine", lambda: FakeEmbeddingEngine()
    )
    monkeypatch.setattr(app.settings, "job_upload_dir", str(tmp_path / "job-uploads"))
    monkeypatch.setattr(job_manager, "get_vector_db", lambda: sqlite_db)
    monkeypatch.setattr(app, "close_vector_db", lambda: None)
    services = __import__("services")
    monkeypatch.setattr(app, "DBService", lambda: services.DBService(db=sqlite_db))
    monkeypatch.setattr(
        job_manager,
        "LLMService",
        lambda llm, embedding_engine=None: services.LLMService(
            llm, embedding_engine=embedding_engine, db=sqlite_db
        ),
    )
    manager = job_manager.JobManager(app.llm, app.settings.job_upload_dir)
    manager.start()
    app.app.state.job_manager = manager
    try:
        yield TestClient(app.app)
    finally:
        manager.stop()
