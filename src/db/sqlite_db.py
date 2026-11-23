import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

from db.vector_db import SearchResult, VectorDB
from errors import IndexCompatibilityError

MIGRATION_DIR = Path(__file__).parent / "migrations" / "sqlite"


class SQLiteVectorDB(VectorDB):
    def __init__(self, db_path: str = "sqlite.db", dimension: int = 384):
        self.dimension = dimension
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(db_path or ":memory:", check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.create_table()

    def create_table(self):
        with self.lock:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version TEXT PRIMARY KEY, applied_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS index_metadata "
                "(id INTEGER PRIMARY KEY CHECK (id = 1), model_name TEXT, dimension INTEGER)"
            )
            for migration in sorted(MIGRATION_DIR.glob("*.sql")):
                version = migration.stem
                applied = self.conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = ?", (version,)
                ).fetchone()
                if not applied:
                    try:
                        script = migration.read_text()
                        self.conn.executescript("BEGIN;\n" + script)
                        self.conn.execute(
                            "INSERT INTO schema_migrations(version) VALUES (?)",
                            (version,),
                        )
                        self.conn.commit()
                    except Exception:
                        self.conn.rollback()
                        raise

    def ensure_embedding_identity(self, model_name: str, dimension: int):
        if dimension != self.dimension:
            raise IndexCompatibilityError()
        with self.lock:
            row = self.conn.execute(
                "SELECT model_name, dimension FROM index_metadata WHERE id = 1"
            ).fetchone()
            count = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            if row is None:
                if count:
                    raise IndexCompatibilityError()
                self.conn.execute(
                    "INSERT INTO index_metadata(id, model_name, dimension) VALUES (1, ?, ?)",
                    (model_name, dimension),
                )
                self.conn.commit()
                return
            if row != (model_name, dimension):
                raise IndexCompatibilityError()

    def validate_vector_dimension(self, vector: list[float]):
        if len(vector) != self.dimension:
            raise IndexCompatibilityError()

    def insert_documents(self, data: list[tuple]):
        with self.lock:
            try:
                rows = []
                for content, metadata, embedding in data:
                    self.validate_vector_dimension(embedding)
                    rows.append(
                        (
                            content,
                            json.dumps(metadata)
                            if not isinstance(metadata, str)
                            else metadata,
                            np.asarray(embedding, dtype=np.float32).tobytes(),
                        )
                    )
                self.conn.executemany(
                    "INSERT INTO documents(content, metadata, embedding) VALUES (?, ?, ?)",
                    rows,
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def search(self, query_vector: list[float], limit: int = 5) -> list[SearchResult]:
        self.validate_vector_dimension(query_vector)
        query = np.asarray(query_vector, dtype=np.float32)
        if not np.linalg.norm(query):
            return []
        with self.lock:
            rows = self.conn.execute(
                "SELECT content, metadata, embedding FROM documents WHERE embedding IS NOT NULL"
            ).fetchall()
        scored = []
        for content, metadata, raw_embedding in rows:
            embedding = np.frombuffer(raw_embedding, dtype=np.float32)
            if embedding.size != self.dimension:
                raise IndexCompatibilityError()
            norm = np.linalg.norm(embedding)
            similarity = (
                float(np.dot(query, embedding) / (np.linalg.norm(query) * norm))
                if norm
                else 0.0
            )
            scored.append((similarity, content, metadata))
        scored.sort(key=lambda row: row[0], reverse=True)
        return [
            SearchResult(content, json.loads(metadata or "{}"), score)
            for score, content, metadata in scored[:limit]
        ]

    def delete_all_documents(self):
        with self.lock:
            try:
                self.conn.execute("DELETE FROM documents")
                self.conn.execute("DELETE FROM index_metadata")
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def is_empty(self) -> bool:
        with self.lock:
            return (
                self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
            )

    @staticmethod
    def _job_from_row(row):
        if row is None:
            return None
        keys = (
            "id",
            "job_type",
            "status",
            "phase",
            "progress",
            "payload",
            "result",
            "error",
            "created_at",
            "updated_at",
            "started_at",
            "finished_at",
        )
        job = dict(zip(keys, row))
        job["payload"] = json.loads(job["payload"])
        job["result"] = json.loads(job["result"]) if job["result"] else None
        return job

    def create_job(self, job: dict):
        with self.lock:
            self.conn.execute(
                "INSERT INTO processing_jobs(id, job_type, status, phase, progress, payload) "
                "VALUES (?, ?, 'queued', 'queued', 0, ?)",
                (job["id"], job["job_type"], json.dumps(job["payload"])),
            )
            self.conn.commit()

    def get_job(self, job_id: str):
        with self.lock:
            row = self.conn.execute(
                "SELECT id, job_type, status, phase, progress, payload, result, error, "
                "created_at, updated_at, started_at, finished_at "
                "FROM processing_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        return self._job_from_row(row)

    def list_jobs(self, status: str | None = None):
        query = (
            "SELECT id, job_type, status, phase, progress, payload, result, error, "
            "created_at, updated_at, started_at, finished_at FROM processing_jobs"
        )
        params = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at, id"
        with self.lock:
            rows = self.conn.execute(query, params).fetchall()
        return [self._job_from_row(row) for row in rows]

    def claim_job(self, job_id: str):
        with self.lock:
            cursor = self.conn.execute(
                "UPDATE processing_jobs SET status = 'processing', phase = 'starting', "
                "progress = 1, started_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ? AND status = 'queued'",
                (job_id,),
            )
            self.conn.commit()
            return cursor.rowcount == 1

    def update_job(self, job_id: str, **changes):
        allowed = {"status", "phase", "progress", "result", "error"}
        changes = {key: value for key, value in changes.items() if key in allowed}
        if not changes:
            return
        assignments = [f"{key} = ?" for key in changes]
        values = [
            json.dumps(value) if key == "result" and value is not None else value
            for key, value in changes.items()
        ]
        assignments.append("updated_at = CURRENT_TIMESTAMP")
        if changes.get("status") in {"completed", "failed", "cancelled"}:
            assignments.append("finished_at = CURRENT_TIMESTAMP")
        with self.lock:
            self.conn.execute(
                f"UPDATE processing_jobs SET {', '.join(assignments)} WHERE id = ?",
                (*values, job_id),
            )
            self.conn.commit()

    def cancel_job(self, job_id: str):
        with self.lock:
            cursor = self.conn.execute(
                "UPDATE processing_jobs SET status = 'cancelled', phase = 'cancelled', "
                "updated_at = CURRENT_TIMESTAMP, finished_at = CURRENT_TIMESTAMP "
                "WHERE id = ? AND status = 'queued'",
                (job_id,),
            )
            self.conn.commit()
            return cursor.rowcount == 1

    def close(self):
        with self.lock:
            self.conn.close()
