import json


def test_empty_query_and_database_clear(api_client):
    response = api_client.post("/query", json={"question": "hello"})
    assert response.status_code == 400
    assert "detail" in response.json()
    assert api_client.get("/database").json() == {"is_empty": True}
    assert api_client.delete("/database").status_code == 401
    assert (
        api_client.delete(
            "/database", headers={"X-Admin-Key": "test-admin-key"}
        ).status_code
        == 200
    )


def test_upload_rejects_invalid_pdf(api_client):
    response = api_client.post(
        "/upload-file",
        files={"file": ("bad.pdf", b"not pdf", "application/pdf")},
    )
    assert response.status_code == 400


def _wait_for_job(api_client, job_id):
    import time

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = api_client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
        time.sleep(0.01)
    raise AssertionError("Background task did not finish in time")


def _insert_document(
    database, document_id, document_name, text="context", page=1, chunk=1
):
    database.ensure_embedding_identity("test-embedding", 384)
    database.insert_documents(
        [
            (
                text,
                json.dumps(
                    {
                        "document_id": document_id,
                        "document_name": document_name,
                        "page": page,
                        "chunk": chunk,
                    }
                ),
                [1.0] * 384,
            )
        ]
    )


def test_upload_returns_job_and_queryable_result(api_client, monkeypatch):
    import job_manager

    class FakeUpload:
        def __init__(self, *_args, **_kwargs):
            pass

        def insert_documents(self, replace_document_id=None):
            return True

    monkeypatch.setattr(job_manager, "UploadService", FakeUpload)
    monkeypatch.setattr(job_manager, "get_embedding_engine", lambda: object())
    # Avoid parser/model/database work; this verifies multipart routing and cleanup.
    response = api_client.post(
        "/upload-file",
        files={"file": ("ok.pdf", b"%PDF-1.7\n", "application/pdf")},
    )
    assert response.status_code == 202
    task = response.json()
    assert task["status"] in {"queued", "processing", "completed"}
    assert task["job_type"] == "upload"
    completed = _wait_for_job(api_client, task["id"])
    assert completed["status"] == "completed"
    assert (
        completed["result"]["status"] == "Documents inserted into the vector database."
    )


def test_job_not_found_and_cancel(api_client):
    assert api_client.get("/jobs/missing").status_code == 404

    manager = api_client.app.state.job_manager
    manager.db.create_job(
        {"id": "cancel-me", "job_type": "query", "payload": {"question": "x"}}
    )
    response = api_client.delete(
        "/jobs/cancel-me", headers={"X-Admin-Key": "test-admin-key"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_admin_actions_are_disabled_when_key_is_not_configured(api_client, monkeypatch):
    import app

    monkeypatch.setattr(app.settings, "admin_api_key", None)
    response = api_client.delete("/database", headers={"X-Admin-Key": "anything"})
    assert response.status_code == 503
    assert "ADMIN_API_KEY" in response.json()["detail"]


def test_wrong_admin_key_is_rejected(api_client):
    response = api_client.delete("/database", headers={"X-Admin-Key": "wrong"})
    assert response.status_code == 401


def test_query_rejects_blank_or_oversized_question(api_client):
    assert api_client.post("/query", json={"question": "  "}).status_code == 422
    assert api_client.post("/query", json={"question": "x" * 2001}).status_code == 422
    assert (
        api_client.post(
            "/query", json={"question": "ok", "unused_payload": "x"}
        ).status_code
        == 422
    )


def test_cors_allows_configured_local_origin_only(api_client):
    allowed = api_client.options(
        "/query",
        headers={
            "Origin": "http://localhost:5500",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5500"

    denied = api_client.options(
        "/query",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in denied.headers


def test_query_returns_background_job_with_answer(api_client, sqlite_db):
    sqlite_db.ensure_embedding_identity("test-embedding", 384)
    sqlite_db.insert_documents(
        [("sample context", json.dumps({"document_name": "sample.pdf"}), [1.0] * 384)]
    )
    response = api_client.post("/query", json={"question": "What is this?"})
    assert response.status_code == 202
    job = _wait_for_job(api_client, response.json()["id"])
    assert job["status"] == "completed"
    assert job["result"]["results"] == "This is a dummy response."
    assert job["result"]["sources"][0]["document_name"] == "sample.pdf"


def test_documents_endpoint_lists_document_and_chunk_counts(api_client, sqlite_db):
    _insert_document(sqlite_db, "doc-a", "a.pdf", "a1", 1, 1)
    _insert_document(sqlite_db, "doc-a", "a.pdf", "a2", 2, 2)
    _insert_document(sqlite_db, "doc-b", "b.pdf", "b1", 1, 1)

    response = api_client.get("/documents")

    assert response.status_code == 200
    data = response.json()
    assert data["document_count"] == 2
    assert data["chunk_count"] == 3
    by_id = {document["document_id"]: document for document in data["documents"]}
    assert by_id["doc-a"]["chunks"] == 2
    assert by_id["doc-a"]["pages"] == 2
    assert by_id["doc-b"]["chunks"] == 1


def test_document_removal_requires_admin_and_targets_one_document(
    api_client, sqlite_db
):
    _insert_document(sqlite_db, "doc-a", "a.pdf", "a1", 1, 1)
    _insert_document(sqlite_db, "doc-a", "a.pdf", "a2", 2, 2)
    _insert_document(sqlite_db, "doc-b", "b.pdf", "b1", 1, 1)

    assert api_client.delete("/documents/doc-a").status_code == 401

    removed = api_client.delete(
        "/documents/doc-a", headers={"X-Admin-Key": "test-admin-key"}
    )
    assert removed.status_code == 200
    assert removed.json()["deleted_chunks"] == 2

    remaining = api_client.get("/documents").json()["documents"]
    assert [document["document_id"] for document in remaining] == ["doc-b"]
    assert (
        api_client.delete(
            "/documents/missing", headers={"X-Admin-Key": "test-admin-key"}
        ).status_code
        == 404
    )


def test_document_replacement_submits_upload_job_with_target(
    api_client, sqlite_db, monkeypatch
):
    import job_manager

    _insert_document(sqlite_db, "doc-a", "a.pdf")
    captured = {}

    class FakeUpload:
        def __init__(self, file_path, **_kwargs):
            captured["file_path"] = file_path

        def insert_documents(self, replace_document_id=None):
            captured["replace_document_id"] = replace_document_id

    monkeypatch.setattr(job_manager, "UploadService", FakeUpload)

    response = api_client.post(
        "/documents/doc-a/replace",
        headers={"X-Admin-Key": "test-admin-key"},
        files={"file": ("new.pdf", b"%PDF-1.7\n", "application/pdf")},
    )

    assert response.status_code == 202
    job = _wait_for_job(api_client, response.json()["id"])
    assert job["status"] == "completed"
    assert captured["replace_document_id"] == "doc-a"

    assert (
        api_client.post(
            "/documents/doc-a/replace",
            files={"file": ("new.pdf", b"%PDF-1.7\n", "application/pdf")},
        ).status_code
        == 401
    )
    assert (
        api_client.post(
            "/documents/missing/replace",
            headers={"X-Admin-Key": "test-admin-key"},
            files={"file": ("new.pdf", b"%PDF-1.7\n", "application/pdf")},
        ).status_code
        == 404
    )


def test_query_filters_sources_and_records_history(api_client, sqlite_db):
    _insert_document(sqlite_db, "doc-a", "a.pdf", "alpha")
    _insert_document(sqlite_db, "doc-b", "b.pdf", "beta")

    response = api_client.post(
        "/query",
        json={
            "question": "What is alpha?",
            "document_ids": ["doc-a"],
            "history": [{"question": "previous question", "answer": "previous answer"}],
        },
    )

    assert response.status_code == 202
    job_id = response.json()["id"]
    job = _wait_for_job(api_client, job_id)
    assert job["status"] == "completed"
    assert {source["document_id"] for source in job["result"]["sources"]} == {"doc-a"}

    stored = api_client.app.state.job_manager.db.get_job(job_id)
    assert stored["payload"]["document_ids"] == ["doc-a"]
    assert stored["payload"]["history"] == [
        {"question": "previous question", "answer": "previous answer"}
    ]


def test_query_with_unknown_document_filter_fails_safely(api_client, sqlite_db):
    _insert_document(sqlite_db, "doc-a", "a.pdf")

    response = api_client.post(
        "/query",
        json={"question": "anything", "document_ids": ["missing"]},
    )

    assert response.status_code == 202
    job = _wait_for_job(api_client, response.json()["id"])
    assert job["status"] == "failed"
    assert "No chunks matched" in job["error"]


def test_query_validation_rejects_invalid_filters_and_history(api_client, sqlite_db):
    _insert_document(sqlite_db, "doc-a", "a.pdf")

    assert (
        api_client.post(
            "/query",
            json={"question": "hello", "document_ids": ["bad id"]},
        ).status_code
        == 422
    )
    assert (
        api_client.post(
            "/query",
            json={"question": "hello", "document_ids": ["id"] * 51},
        ).status_code
        == 422
    )
    assert (
        api_client.post(
            "/query",
            json={
                "question": "hello",
                "history": [{"question": "q", "answer": "a"}] * 5,
            },
        ).status_code
        == 422
    )
    assert (
        api_client.post(
            "/query",
            json={
                "question": "hello",
                "history": [
                    {"question": "q" * 1000, "answer": "a" * 1000} for _ in range(4)
                ],
            },
        ).status_code
        == 422
    )


def test_query_body_is_limited_to_64_kib(api_client):
    response = api_client.post(
        "/query",
        content=b"x" * (64 * 1024 + 1),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_query_worker_runs_while_upload_worker_is_busy(
    api_client, sqlite_db, monkeypatch
):
    import json
    import threading

    import job_manager

    started = threading.Event()
    release = threading.Event()

    class SlowUpload:
        def __init__(self, *_args, **_kwargs):
            pass

        def insert_documents(self, replace_document_id=None):
            started.set()
            assert release.wait(timeout=3)

    monkeypatch.setattr(job_manager, "UploadService", SlowUpload)
    sqlite_db.ensure_embedding_identity("test-embedding", 384)
    sqlite_db.insert_documents(
        [("sample context", json.dumps({"document_name": "sample.pdf"}), [1.0] * 384)]
    )
    upload = api_client.post(
        "/upload-file",
        files={"file": ("slow.pdf", b"%PDF-1.7\n", "application/pdf")},
    )
    try:
        assert upload.status_code == 202
        assert started.wait(timeout=2)
        query = api_client.post("/query", json={"question": "What is this?"})
        query_job = _wait_for_job(api_client, query.json()["id"])
        assert query_job["status"] == "completed"
        assert (
            api_client.get(f"/jobs/{upload.json()['id']}").json()["status"]
            == "processing"
        )
    finally:
        release.set()
    assert _wait_for_job(api_client, upload.json()["id"])["status"] == "completed"


def _parse_sse(text):
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


def test_query_stream_emits_progressive_events(api_client, sqlite_db):
    sqlite_db.ensure_embedding_identity("test-embedding", 384)
    sqlite_db.insert_documents(
        [
            (
                "sample context",
                json.dumps({"document_name": "sample.pdf", "page": 2}),
                [1.0] * 384,
            )
        ]
    )

    response = api_client.post("/query/stream", json={"question": "What is this?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = _parse_sse(response.text)
    events = [frame["event"] for frame in frames]
    assert events[0] == "job"
    assert "sources" in events
    assert "token" in events
    assert events[-1] == "done"

    done = frames[-1]["data"]
    assert done["results"] == "This is a dummy response."
    assert done["sources"][0]["document_name"] == "sample.pdf"


def test_query_stream_requires_documents(api_client):
    response = api_client.post("/query/stream", json={"question": "hello"})

    assert response.status_code == 400
    assert "detail" in response.json()
    assert api_client.post("/query/stream", json={"question": "  "}).status_code == 422


def test_job_stream_replays_completed_result(api_client):
    manager = api_client.app.state.job_manager
    manager.db.create_job(
        {"id": "completed-job", "job_type": "query", "payload": {"question": "x"}}
    )
    manager.db.update_job(
        "completed-job",
        status="completed",
        phase="completed",
        progress=100,
        result={"results": "cached answer", "sources": []},
    )

    response = api_client.get("/jobs/completed-job/stream")

    assert response.status_code == 200
    frames = _parse_sse(response.text)
    assert frames[0]["event"] == "job"
    assert frames[-1]["event"] == "done"
    assert frames[-1]["data"]["results"] == "cached answer"

    assert api_client.get("/jobs/missing/stream").status_code == 404


def test_health_and_web_interface_are_served_from_api(api_client):
    health = api_client.get("/health")
    home = api_client.get("/")
    script = api_client.get("/script.js")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert home.status_code == 200
    assert "Local RAG" in home.text
    assert "documentsList" in home.text
    assert "conversationList" in home.text
    assert "clearConversationButton" in home.text
    assert script.status_code == 200
    assert "window.location.origin" in script.text
    assert "/query/stream" in script.text
    assert "/documents" in script.text
    assert "localStorage" in script.text
