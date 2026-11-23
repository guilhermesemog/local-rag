"""Persistent asynchronous processing queues for upload and query jobs."""

import logging
from pathlib import Path
from queue import Queue
from threading import Lock, Thread

from db.factory_db import get_vector_db
from embedding_engine import get_embedding_engine
from errors import AppError
from services import DBService, LLMService, UploadService

logger = logging.getLogger(__name__)
MAX_QUEUED_JOBS_PER_TYPE = 20


class JobQueueFull(Exception):
    pass


class JobManager:
    def __init__(self, llm, upload_dir: str):
        self.llm = llm
        self.upload_dir = Path(upload_dir)
        self.db = get_vector_db()
        self.queues = {"upload": Queue(), "query": Queue()}
        self.workers = []
        self.admission_lock = Lock()

    def start(self):
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        interrupted = self.db.list_jobs("processing")
        for job in interrupted:
            self.db.update_job(
                job["id"],
                status="failed",
                phase="interrupted",
                error="Processing was interrupted when the server stopped.",
            )
            self._remove_upload(job)

        queued = self.db.list_jobs("queued")
        for job in queued:
            if job["job_type"] in self.queues:
                self.queues[job["job_type"]].put(job["id"])

        self.workers = [
            Thread(target=self._worker, args=("upload",), name="upload-job-worker"),
            Thread(target=self._worker, args=("query",), name="query-job-worker"),
        ]
        for worker in self.workers:
            worker.start()

    def stop(self):
        for queue in self.queues.values():
            queue.put(None)
        if self.workers:
            for worker in self.workers:
                worker.join()

    def submit(self, job_id: str, job_type: str, payload: dict):
        with self.admission_lock:
            queued = self.db.list_jobs("queued")
            if (
                sum(job["job_type"] == job_type for job in queued)
                >= MAX_QUEUED_JOBS_PER_TYPE
            ):
                raise JobQueueFull("The processing queue is full. Try again later.")
            self.db.create_job(
                {"id": job_id, "job_type": job_type, "payload": payload},
            )
            self.queues[job_type].put(job_id)
            return self.db.get_job(job_id)

    def clear_database_if_idle(self):
        with self.admission_lock:
            if self.db.list_jobs("queued") or self.db.list_jobs("processing"):
                return False
            DBService(db=self.db).clear_database()
            return True

    def get(self, job_id: str):
        return self.db.get_job(job_id)

    def cancel(self, job_id: str):
        cancelled = self.db.cancel_job(job_id)
        if cancelled:
            self._remove_upload(self.db.get_job(job_id))
        return cancelled

    def _worker(self, job_type: str):
        queue = self.queues[job_type]
        while True:
            job_id = queue.get()
            try:
                if job_id is None:
                    return
                claimed = self.db.claim_job(job_id)
                if not claimed:
                    continue
                job = self.db.get_job(job_id)
                try:
                    result = self._run_job(job)
                    self.db.update_job(
                        job_id,
                        status="completed",
                        phase="completed",
                        progress=100,
                        result=result,
                        error=None,
                    )
                except AppError as exc:
                    self.db.update_job(
                        job_id,
                        status="failed",
                        phase="failed",
                        error=exc.public_message,
                    )
                except Exception:
                    logger.exception("Background %s job failed (%s)", job_type, job_id)
                    self.db.update_job(
                        job_id,
                        status="failed",
                        phase="failed",
                        error="The task could not be completed due to a server error.",
                    )
                finally:
                    self._remove_upload(job)
            finally:
                queue.task_done()

    def _run_job(self, job: dict):
        job_id = job["id"]
        progress = lambda phase, value: self.db.update_job(
            job_id, phase=phase, progress=value
        )
        if job["job_type"] == "upload":
            service = UploadService(
                job["payload"]["file_path"],
                engine=get_embedding_engine(),
                progress_callback=progress,
            )
            service.insert_documents()
            return {"status": "Documents inserted into the vector database."}

        service = LLMService(self.llm, embedding_engine=get_embedding_engine())
        service.progress_callback = progress
        result = service.answer(job["payload"]["question"])
        if isinstance(result, dict):
            return result
        return {"results": result, "sources": []}

    def _remove_upload(self, job):
        if job and job.get("job_type") == "upload":
            file_path = job.get("payload", {}).get("file_path")
            if file_path:
                path = Path(file_path)
                try:
                    path = path.resolve()
                    if path.parent != self.upload_dir.resolve():
                        return
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove staged upload %s", path)


def public_job(job: dict):
    return {
        key: job[key]
        for key in (
            "id",
            "job_type",
            "status",
            "phase",
            "progress",
            "result",
            "error",
            "created_at",
            "updated_at",
            "started_at",
            "finished_at",
        )
    }
