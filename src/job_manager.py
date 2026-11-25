"""Persistent asynchronous processing queues for upload and query jobs."""

import asyncio
import logging
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from time import monotonic

from db.factory_db import get_vector_db
from embedding_engine import get_embedding_engine
from errors import AppError
from services import DBService, LLMService, UploadService
from streaming import (
    END,
    HEARTBEAT_FRAME,
    HEARTBEAT_SECONDS,
    StreamHub,
    format_sse,
)

logger = logging.getLogger(__name__)
MAX_QUEUED_JOBS_PER_TYPE = 20


class JobQueueFull(Exception):
    pass


def _event_data(event):
    return {key: value for key, value in event.items() if key != "type"}


class JobManager:
    def __init__(self, llm, upload_dir: str):
        self.llm = llm
        self.upload_dir = Path(upload_dir)
        self.db = get_vector_db()
        self.queues = {"upload": Queue(), "query": Queue()}
        self.workers = []
        self.admission_lock = Lock()
        self.streams = StreamHub()

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
                if job["job_type"] == "query":
                    self.streams.open(job["id"])
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
        self.streams.close_all()

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
            if job_type == "query":
                self.streams.open(job_id)
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
            job = self.db.get_job(job_id)
            self._remove_upload(job)
            self.streams.publish(
                job_id, {"type": "error", "detail": "Task cancelled."}
            )
            self.streams.close(job_id)
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
                    if job_type == "query":
                        job = self.db.get_job(job_id)
                        detail = (
                            "Task cancelled."
                            if job and job["status"] == "cancelled"
                            else "Task is no longer available."
                        )
                        self.streams.publish(
                            job_id, {"type": "error", "detail": detail}
                        )
                        self.streams.close(job_id)
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
                    if job_type == "query":
                        self.streams.publish(
                            job_id,
                            {
                                "type": "done",
                                "results": result.get("results", ""),
                                "sources": result.get("sources", []),
                            },
                        )
                        self.streams.close(job_id)
                except AppError as exc:
                    self.db.update_job(
                        job_id,
                        status="failed",
                        phase="failed",
                        error=exc.public_message,
                    )
                    if job_type == "query":
                        self.streams.publish(
                            job_id,
                            {"type": "error", "detail": exc.public_message},
                        )
                        self.streams.close(job_id)
                except Exception as exc:  # noqa: BLE001 - persist unexpected job failures
                    logger.error(
                        "Background %s job failed (%s, %s)",
                        job_type,
                        job_id,
                        type(exc).__name__,
                    )
                    self.db.update_job(
                        job_id,
                        status="failed",
                        phase="failed",
                        error="The task could not be completed due to a server error.",
                    )
                    if job_type == "query":
                        self.streams.publish(
                            job_id,
                            {
                                "type": "error",
                                "detail": (
                                    "The task could not be completed due to a "
                                    "server error."
                                ),
                            },
                        )
                        self.streams.close(job_id)
                finally:
                    self._remove_upload(job)
            finally:
                queue.task_done()

    def _run_job(self, job: dict):
        job_id = job["id"]

        def progress(phase, value):
            self.db.update_job(job_id, phase=phase, progress=value)
            self.streams.publish(
                job_id,
                {"type": "phase", "phase": phase, "progress": value},
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
        parts = []
        sources = []
        for event in service.answer_stream(job["payload"]["question"]):
            if event["type"] == "sources":
                sources = event["sources"]
                self.streams.publish(job_id, event)
            elif event["type"] == "token":
                parts.append(event["text"])
                self.streams.publish(job_id, event)
        return {"results": "".join(parts).strip(), "sources": sources}

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
                    logger.warning("Could not remove staged upload")

    async def stream_events(self, job_id: str, initial_job: dict | None = None):
        """Yield SSE frames for a task, replaying its channel or persisted state."""
        job = initial_job or self.get(job_id)
        if job is None:
            yield format_sse("error", {"detail": "Processing task not found."})
            return

        yield format_sse("job", public_job(job))

        channel = self.streams.get(job_id)
        if channel is not None:
            subscription = channel.subscribe()
            last_heartbeat = monotonic()
            try:
                while True:
                    event = subscription.get_nowait()
                    if event is END:
                        return
                    if event is None:
                        if monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
                            last_heartbeat = monotonic()
                            yield HEARTBEAT_FRAME
                        await asyncio.sleep(0.05)
                        continue
                    last_heartbeat = monotonic()
                    yield format_sse(event["type"], _event_data(event))
            finally:
                channel.unsubscribe(subscription)
            return

        last_phase = None
        last_progress = None
        last_heartbeat = monotonic()
        while True:
            job = await asyncio.to_thread(self.get, job_id)
            if job is None:
                yield format_sse("error", {"detail": "Processing task not found."})
                return
            if (job["phase"], job["progress"]) != (last_phase, last_progress):
                last_phase = job["phase"]
                last_progress = job["progress"]
                yield format_sse(
                    "phase",
                    {"phase": last_phase, "progress": last_progress},
                )
            if job["status"] == "completed":
                result = job.get("result") or {}
                if not isinstance(result, dict):
                    result = {"results": str(result), "sources": []}
                yield format_sse(
                    "done",
                    {
                        "results": result.get("results", ""),
                        "sources": result.get("sources", []),
                    },
                )
                return
            if job["status"] in {"failed", "cancelled"}:
                yield format_sse(
                    "error",
                    {"detail": job.get("error") or "The task did not complete."},
                )
                return
            if monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
                last_heartbeat = monotonic()
                yield HEARTBEAT_FRAME
            await asyncio.sleep(0.3)


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
