import time
from threading import Event

import pytest

from errors import DocumentParseError
from job_manager import JobManager, JobQueueFull


def _wait_for_status(database, job_id, terminal, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = database.get_job(job_id)
        if job["status"] in terminal:
            return job
        time.sleep(0.01)
    raise AssertionError(f"Job {job_id} did not reach a terminal state")


def test_manager_resumes_queued_jobs_and_marks_interrupted_jobs_failed(
    sqlite_db, tmp_path, monkeypatch
):
    import job_manager

    monkeypatch.setattr(job_manager, "get_vector_db", lambda: sqlite_db)
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    interrupted_pdf = upload_dir / "interrupted.pdf"
    interrupted_pdf.write_bytes(b"pdf")
    sqlite_db.create_job(
        {
            "id": "interrupted",
            "job_type": "upload",
            "payload": {"file_path": str(interrupted_pdf)},
        }
    )
    sqlite_db.claim_job("interrupted")
    sqlite_db.create_job(
        {"id": "queued", "job_type": "query", "payload": {"question": "hello"}}
    )

    manager = JobManager(None, str(upload_dir))
    manager._run_job = lambda _job: {"results": "answer", "sources": []}
    manager.start()
    try:
        completed = _wait_for_status(sqlite_db, "queued", {"completed", "failed"})
        assert completed["status"] == "completed"
        assert completed["result"]["results"] == "answer"
        failed = sqlite_db.get_job("interrupted")
        assert failed["status"] == "failed"
        assert failed["phase"] == "interrupted"
        assert not interrupted_pdf.exists()
    finally:
        manager.stop()


def test_manager_rejects_jobs_when_type_queue_is_full(sqlite_db, tmp_path, monkeypatch):
    import job_manager

    monkeypatch.setattr(job_manager, "get_vector_db", lambda: sqlite_db)
    for index in range(job_manager.MAX_QUEUED_JOBS_PER_TYPE):
        sqlite_db.create_job(
            {
                "id": f"queued-{index}",
                "job_type": "query",
                "payload": {"question": "hello"},
            }
        )

    manager = JobManager(None, str(tmp_path / "uploads"))
    with pytest.raises(JobQueueFull):
        manager.submit("overflow", "query", {"question": "hello"})
    assert sqlite_db.get_job("overflow") is None


def test_database_clear_is_blocked_until_queued_jobs_are_cancelled(
    sqlite_db, tmp_path, monkeypatch, fake_embeddings
):
    import job_manager

    monkeypatch.setattr(job_manager, "get_vector_db", lambda: sqlite_db)
    sqlite_db.insert_documents([("context", {}, fake_embeddings[0])])
    manager = JobManager(None, str(tmp_path / "uploads"))
    manager.submit("queued", "query", {"question": "hello"})
    assert manager.clear_database_if_idle() is False
    assert sqlite_db.is_empty() is False
    assert manager.cancel("queued") is True
    assert manager.clear_database_if_idle() is True
    assert sqlite_db.is_empty() is True


def test_manager_exposes_safe_processing_failure(sqlite_db, tmp_path, monkeypatch):
    import job_manager

    monkeypatch.setattr(job_manager, "get_vector_db", lambda: sqlite_db)
    manager = JobManager(None, str(tmp_path / "uploads"))
    manager._run_job = lambda _job: (_ for _ in ()).throw(DocumentParseError())
    manager.start()
    try:
        manager.submit("failed", "upload", {"file_path": "/not-used.pdf"})
        failed = _wait_for_status(sqlite_db, "failed", {"failed"})
        assert failed["error"] == "The PDF could not be processed."
    finally:
        manager.stop()


def test_upload_and_query_workers_run_independently(sqlite_db, tmp_path, monkeypatch):
    import job_manager

    monkeypatch.setattr(job_manager, "get_vector_db", lambda: sqlite_db)
    upload_started = Event()
    finish_upload = Event()
    manager = JobManager(None, str(tmp_path / "uploads"))

    def run_job(job):
        if job["job_type"] == "upload":
            upload_started.set()
            assert finish_upload.wait(timeout=2)
            return {"status": "uploaded"}
        return {"results": "answer", "sources": []}

    manager._run_job = run_job
    manager.start()
    try:
        manager.submit("upload", "upload", {"file_path": "/not-used.pdf"})
        assert upload_started.wait(timeout=1)
        manager.submit("query", "query", {"question": "hello"})
        query = _wait_for_status(sqlite_db, "query", {"completed", "failed"})
        assert query["status"] == "completed"
        assert sqlite_db.get_job("upload")["status"] == "processing"
    finally:
        finish_upload.set()
        manager.stop()
