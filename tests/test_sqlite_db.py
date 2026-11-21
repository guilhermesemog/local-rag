import pytest
import sqlite3
import numpy as np

from errors import IndexCompatibilityError


def test_sqlite_insert_search_clear(sqlite_db, fake_embeddings):
    metadata = {
        "document_id": "doc-1",
        "document_name": "sample.pdf",
        "page": 2,
        "chunk": 1,
    }
    sqlite_db.insert_documents([("hello", metadata, fake_embeddings[0])])
    assert sqlite_db.is_empty() is False
    results = sqlite_db.search(fake_embeddings[0], limit=1)
    assert results[0].content == "hello"
    assert results[0].metadata == metadata
    sqlite_db.delete_all_documents()
    assert sqlite_db.is_empty() is True


def test_postgres_insert_search_clear(postgres_db, fake_embeddings):
    postgres_db.insert_documents([("test-only", {}, fake_embeddings[0])])
    assert postgres_db.search(fake_embeddings[0], limit=1)
    postgres_db.delete_all_documents()
    assert postgres_db.is_empty()


def test_sqlite_embedding_identity_blocks_model_switch(sqlite_db, fake_embeddings):
    sqlite_db.ensure_embedding_identity("model-a", 384)
    sqlite_db.insert_documents([("legacy vector", {}, fake_embeddings[0])])
    with pytest.raises(IndexCompatibilityError):
        sqlite_db.ensure_embedding_identity("model-b", 384)
    sqlite_db.delete_all_documents()
    sqlite_db.ensure_embedding_identity("model-b", 384)


def test_sqlite_rolls_back_invalid_batch(sqlite_db, fake_embeddings):
    rows = [("valid", {}, fake_embeddings[0]), (None, {}, fake_embeddings[0])]
    with pytest.raises(sqlite3.IntegrityError):
        sqlite_db.insert_documents(rows)
    assert sqlite_db.is_empty()


def test_sqlite_search_reads_persisted_vectors(tmp_path, fake_embeddings):
    from db.sqlite_db import SQLiteVectorDB

    path = tmp_path / "persist.db"
    first = SQLiteVectorDB(str(path))
    first.insert_documents([("persisted", {}, fake_embeddings[0])])
    first.close()
    reopened = SQLiteVectorDB(str(path))
    try:
        assert reopened.search(fake_embeddings[0], limit=1)[0].content == "persisted"
    finally:
        reopened.close()


def test_sqlite_migration_preserves_legacy_rows(tmp_path, fake_embeddings):
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT NOT NULL, "
        "metadata TEXT DEFAULT '{}', embedding BLOB)"
    )
    connection.execute(
        "INSERT INTO documents(id, content, metadata, embedding) VALUES(1, ?, '{}', ?)",
        ("legacy content", np.array(fake_embeddings[0], dtype="float32").tobytes()),
    )
    connection.commit()
    connection.close()

    from db.sqlite_db import SQLiteVectorDB

    database = SQLiteVectorDB(str(path))
    try:
        assert (
            database.search(fake_embeddings[0], limit=1)[0].content == "legacy content"
        )
        with pytest.raises(IndexCompatibilityError):
            database.ensure_embedding_identity("model", 384)
        assert database.is_empty() is False
    finally:
        database.close()
