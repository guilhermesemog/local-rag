import sqlite3

import numpy as np
import pytest

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


def _document_metadata(document_id, document_name, page=1, chunk=1):
    return {
        "document_id": document_id,
        "document_name": document_name,
        "page": page,
        "chunk": chunk,
    }


def test_sqlite_lists_filters_and_removes_documents(sqlite_db, fake_embeddings):
    sqlite_db.insert_documents(
        [
            ("a1", _document_metadata("doc-a", "a.pdf", 1, 1), fake_embeddings[0]),
            ("a2", _document_metadata("doc-a", "a.pdf", 2, 2), fake_embeddings[0]),
            ("b1", _document_metadata("doc-b", "b.pdf", 1, 1), fake_embeddings[0]),
        ]
    )

    documents = sqlite_db.list_documents()
    by_id = {document["document_id"]: document for document in documents}
    assert by_id["doc-a"]["document_name"] == "a.pdf"
    assert by_id["doc-a"]["chunks"] == 2
    assert by_id["doc-a"]["pages"] == 2
    assert by_id["doc-b"]["chunks"] == 1
    assert sqlite_db.document_exists("doc-a") is True
    assert sqlite_db.document_exists("missing") is False

    filtered = sqlite_db.search(
        fake_embeddings[0],
        limit=6,
        document_ids=["doc-a"],
    )
    assert {result.content for result in filtered} == {"a1", "a2"}

    assert sqlite_db.delete_documents(["doc-b"]) == 1
    assert sqlite_db.document_exists("doc-b") is False
    assert sqlite_db.delete_documents(["missing"]) == 0


def test_sqlite_lists_legacy_chunks_as_unidentified(sqlite_db, fake_embeddings):
    sqlite_db.insert_documents([("legacy", {}, fake_embeddings[0])])

    assert sqlite_db.list_documents() == [
        {
            "document_id": "",
            "document_name": "Unidentified chunks",
            "chunks": 1,
            "pages": 0,
        }
    ]
    assert sqlite_db.document_exists("") is False


def test_sqlite_replaces_documents_atomically(sqlite_db, fake_embeddings):
    sqlite_db.insert_documents(
        [
            ("old", _document_metadata("doc-old", "old.pdf"), fake_embeddings[0]),
            ("other", _document_metadata("doc-other", "other.pdf"), fake_embeddings[0]),
        ]
    )

    deleted = sqlite_db.replace_documents(
        "doc-old",
        [
            (
                "new",
                _document_metadata("doc-new", "new.pdf"),
                fake_embeddings[0],
            )
        ],
    )

    assert deleted == 1
    assert sqlite_db.document_exists("doc-old") is False
    assert sqlite_db.document_exists("doc-new") is True
    assert sqlite_db.document_exists("doc-other") is True

    with pytest.raises(IndexCompatibilityError):
        sqlite_db.replace_documents(
            "doc-new",
            [("broken", _document_metadata("doc-new", "new.pdf"), [1.0])],
        )
    assert sqlite_db.document_exists("doc-new") is True


def test_postgres_lists_filters_and_removes_documents(postgres_db, fake_embeddings):
    postgres_db.insert_documents(
        [
            ("a1", _document_metadata("doc-a", "a.pdf", 1, 1), fake_embeddings[0]),
            ("a2", _document_metadata("doc-a", "a.pdf", 2, 2), fake_embeddings[0]),
            ("b1", _document_metadata("doc-b", "b.pdf", 1, 1), fake_embeddings[0]),
        ]
    )

    by_id = {
        document["document_id"]: document for document in postgres_db.list_documents()
    }
    assert by_id["doc-a"]["chunks"] == 2
    assert by_id["doc-a"]["pages"] == 2
    assert [
        result.content
        for result in postgres_db.search(
            fake_embeddings[0], limit=6, document_ids=["doc-b"]
        )
    ] == ["b1"]
    assert postgres_db.delete_documents(["doc-a"]) == 2
    assert postgres_db.document_exists("doc-b") is True


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


def test_sqlite_jobs_persist_and_only_queued_jobs_can_be_cancelled(sqlite_db):
    sqlite_db.create_job(
        {"id": "queued-1", "job_type": "query", "payload": {"question": "hello"}}
    )
    assert sqlite_db.cancel_job("queued-1") is True
    assert sqlite_db.get_job("queued-1")["status"] == "cancelled"

    sqlite_db.create_job(
        {"id": "running-1", "job_type": "query", "payload": {"question": "hello"}}
    )
    assert sqlite_db.claim_job("running-1") is True
    assert sqlite_db.cancel_job("running-1") is False
    sqlite_db.update_job(
        "running-1",
        status="completed",
        phase="completed",
        progress=100,
        result={"results": "answer", "sources": []},
    )
    result = sqlite_db.get_job("running-1")
    assert result["status"] == "completed"
    assert result["result"] == {"results": "answer", "sources": []}
