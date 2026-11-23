def test_empty_query_and_database_clear(api_client):
    response = api_client.post("/query", json={"question": "hello"})
    assert response.status_code == 400
    assert "detail" in response.json()
    assert api_client.get("/database").json() == {"is_empty": True}
    assert api_client.delete("/database").status_code == 200


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


def test_upload_returns_job_and_queryable_result(api_client, monkeypatch):
    import job_manager

    class FakeUpload:
        def __init__(self, *_args, **_kwargs):
            pass

        def insert_documents(self):
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
    response = api_client.delete("/jobs/cancel-me")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_query_returns_background_job_with_answer(api_client, sqlite_db):
    import json

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

        def insert_documents(self):
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


def test_health_and_web_interface_are_served_from_api(api_client):
    health = api_client.get("/health")
    home = api_client.get("/")
    script = api_client.get("/script.js")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert home.status_code == 200
    assert "Local RAG" in home.text
    assert script.status_code == 200
    assert "window.location.origin" in script.text
