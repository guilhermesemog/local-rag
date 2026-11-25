import json
from contextlib import contextmanager
from pathlib import Path

from psycopg2 import pool
from psycopg2.extras import Json, execute_values

from db.vector_db import SearchResult, VectorDB
from errors import IndexCompatibilityError
from settings import Settings, get_settings

MIGRATION_DIR = Path(__file__).parent / "migrations" / "postgres"


class PostgresVectorDB(VectorDB):
    def __init__(self, dimension: int | None = None, settings: Settings | None = None):
        settings = settings or get_settings()
        self.dimension = dimension or settings.embedding_dimension
        if not settings.db_password:
            raise ValueError("DB_PASSWORD is required to connect to PostgreSQL")
        self.pool = pool.ThreadedConnectionPool(
            settings.postgres_pool_min,
            settings.postgres_pool_max,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password.get_secret_value(),
            host=settings.db_host,
            port=settings.db_port,
        )
        try:
            self.create_table()
        except Exception:
            self.pool.closeall()
            raise

    @contextmanager
    def _connection(self):
        connection = self.pool.getconn()
        try:
            yield connection
        except Exception:
            connection.rollback()
            raise
        finally:
            self.pool.putconn(connection)

    def create_table(self):
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "CREATE SCHEMA IF NOT EXISTS embeddings;"
                "CREATE TABLE IF NOT EXISTS embeddings.schema_migrations "
                "(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT NOW());"
                "CREATE TABLE IF NOT EXISTS embeddings.index_metadata "
                "(id SMALLINT PRIMARY KEY CHECK(id = 1), model_name TEXT, dimension INTEGER);"
            )
            connection.commit()
            for migration in sorted(MIGRATION_DIR.glob("*.sql")):
                version = migration.stem
                cursor.execute(
                    "SELECT 1 FROM embeddings.schema_migrations WHERE version = %s",
                    (version,),
                )
                if cursor.fetchone():
                    continue
                sql = migration.read_text().replace("${DIMENSION}", str(self.dimension))
                try:
                    cursor.execute(sql)
                    cursor.execute(
                        "INSERT INTO embeddings.schema_migrations(version) VALUES (%s)",
                        (version,),
                    )
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise

    def ensure_embedding_identity(self, model_name: str, dimension: int):
        if dimension != self.dimension:
            raise IndexCompatibilityError()
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute("LOCK TABLE embeddings.documents IN ACCESS EXCLUSIVE MODE")
            cursor.execute(
                "SELECT model_name, dimension FROM embeddings.index_metadata WHERE id = 1"
            )
            row = cursor.fetchone()
            cursor.execute("SELECT COUNT(*) FROM embeddings.documents")
            count = cursor.fetchone()[0]
            if row is None:
                if count:
                    raise IndexCompatibilityError()
                cursor.execute(
                    "ALTER TABLE embeddings.documents ALTER COLUMN embedding "
                    f"TYPE vector({dimension}) USING embedding::vector({dimension})"
                )
                cursor.execute("DROP INDEX IF EXISTS embeddings.document_embedding_idx")
                cursor.execute(
                    "CREATE INDEX document_embedding_idx ON embeddings.documents "
                    "USING hnsw (embedding vector_cosine_ops)"
                )
                cursor.execute(
                    "INSERT INTO embeddings.index_metadata(id, model_name, dimension) VALUES (1, %s, %s)",
                    (model_name, dimension),
                )
                connection.commit()
                return
            if row != (model_name, dimension):
                raise IndexCompatibilityError()
            connection.rollback()

    def validate_vector_dimension(self, vector: list[float]):
        if len(vector) != self.dimension:
            raise IndexCompatibilityError()

    def _serialize_rows(self, data: list[tuple]):
        rows = []
        for content, metadata, embedding in data:
            self.validate_vector_dimension(embedding)
            parsed_metadata = (
                json.loads(metadata) if isinstance(metadata, str) else metadata
            )
            rows.append((content, Json(parsed_metadata), embedding))
        return rows

    def insert_documents(self, data: list[tuple]):
        rows = self._serialize_rows(data)
        with self._connection() as connection:
            with connection.cursor() as cursor:
                execute_values(
                    cursor,
                    "INSERT INTO embeddings.documents(content, metadata, embedding) VALUES %s",
                    rows,
                )
            connection.commit()

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        self.validate_vector_dimension(query_vector)
        with self._connection() as connection:
            with connection.cursor() as cursor:
                if document_ids:
                    cursor.execute(
                        "SELECT content, metadata, 1 - (embedding <=> %s::vector) "
                        "FROM embeddings.documents "
                        "WHERE metadata->>'document_id' = ANY(%s) "
                        "ORDER BY embedding <=> %s::vector LIMIT %s",
                        (query_vector, list(document_ids), query_vector, limit),
                    )
                else:
                    cursor.execute(
                        "SELECT content, metadata, 1 - (embedding <=> %s::vector) "
                        "FROM embeddings.documents ORDER BY embedding <=> %s::vector LIMIT %s",
                        (query_vector, query_vector, limit),
                    )
                rows = cursor.fetchall()
            connection.rollback()
        return [SearchResult(row[0], row[1] or {}, float(row[2])) for row in rows]

    def list_documents(self):
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COALESCE(metadata->>'document_id', '') AS document_id, "
                    "CASE WHEN COALESCE(metadata->>'document_id', '') = '' "
                    "THEN 'Unidentified chunks' "
                    "ELSE COALESCE(NULLIF(metadata->>'document_name', ''), "
                    "'Unknown document') END AS document_name, "
                    "COUNT(*) AS chunks, "
                    "COUNT(DISTINCT NULLIF(metadata->>'page', '')) AS pages "
                    "FROM embeddings.documents GROUP BY 1, 2 ORDER BY 2, 1"
                )
                rows = cursor.fetchall()
            connection.rollback()
        return [
            {
                "document_id": row[0],
                "document_name": row[1],
                "chunks": int(row[2]),
                "pages": int(row[3]),
            }
            for row in rows
        ]

    def document_exists(self, document_id: str) -> bool:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT EXISTS(SELECT 1 FROM embeddings.documents "
                    "WHERE metadata->>'document_id' = %s)",
                    (document_id,),
                )
                exists = cursor.fetchone()[0]
            connection.rollback()
            return bool(exists)

    def delete_documents(self, document_ids: list[str]) -> int:
        if not document_ids:
            return 0
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM embeddings.documents "
                    "WHERE metadata->>'document_id' = ANY(%s)",
                    (list(document_ids),),
                )
                deleted = cursor.rowcount
            connection.commit()
            return deleted

    def replace_documents(self, document_id: str, data: list[tuple]) -> int:
        rows = self._serialize_rows(data)
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM embeddings.documents "
                    "WHERE metadata->>'document_id' = %s",
                    (document_id,),
                )
                deleted = cursor.rowcount
                execute_values(
                    cursor,
                    "INSERT INTO embeddings.documents(content, metadata, embedding) VALUES %s",
                    rows,
                )
            connection.commit()
            return deleted

    def delete_all_documents(self):
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM embeddings.documents")
                cursor.execute("DELETE FROM embeddings.index_metadata")
            connection.commit()

    def is_empty(self) -> bool:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM embeddings.documents")
                empty = cursor.fetchone()[0] == 0
            connection.rollback()
            return empty

    @staticmethod
    def _job_from_row(cursor, row):
        if row is None:
            return None
        job = dict(zip([column[0] for column in cursor.description], row))
        for key in ("created_at", "updated_at", "started_at", "finished_at"):
            if job[key] is not None:
                job[key] = job[key].isoformat()
        return job

    def create_job(self, job: dict):
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO embeddings.processing_jobs "
                    "(id, job_type, status, phase, progress, payload) "
                    "VALUES (%s, %s, 'queued', 'queued', 0, %s)",
                    (job["id"], job["job_type"], Json(job["payload"])),
                )
            connection.commit()

    def get_job(self, job_id: str):
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT id, job_type, status, phase, progress, payload, result, error, "
                    "created_at, updated_at, started_at, finished_at "
                    "FROM embeddings.processing_jobs WHERE id = %s",
                    (job_id,),
                )
                row = cursor.fetchone()
                result = self._job_from_row(cursor, row)
            connection.rollback()
            return result

    def list_jobs(self, status: str | None = None):
        query = (
            "SELECT id, job_type, status, phase, progress, payload, result, error, "
            "created_at, updated_at, started_at, finished_at "
            "FROM embeddings.processing_jobs"
        )
        params = ()
        if status:
            query += " WHERE status = %s"
            params = (status,)
        query += " ORDER BY created_at, id"
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                jobs = [self._job_from_row(cursor, row) for row in cursor.fetchall()]
            connection.rollback()
            return jobs

    def claim_job(self, job_id: str):
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE embeddings.processing_jobs SET status = 'processing', "
                    "phase = 'starting', progress = 1, started_at = NOW(), updated_at = NOW() "
                    "WHERE id = %s AND status = 'queued' RETURNING id",
                    (job_id,),
                )
                claimed = cursor.fetchone() is not None
            connection.commit()
            return claimed

    def update_job(self, job_id: str, **changes):
        allowed = {"status", "phase", "progress", "result", "error"}
        changes = {key: value for key, value in changes.items() if key in allowed}
        if not changes:
            return
        assignments = [f"{key} = %s" for key in changes]
        values = [
            Json(value) if key == "result" and value is not None else value
            for key, value in changes.items()
        ]
        assignments.append("updated_at = NOW()")
        if changes.get("status") in {"completed", "failed", "cancelled"}:
            assignments.append("finished_at = NOW()")
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE embeddings.processing_jobs SET {', '.join(assignments)} WHERE id = %s",
                    (*values, job_id),
                )
            connection.commit()

    def cancel_job(self, job_id: str):
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE embeddings.processing_jobs SET status = 'cancelled', "
                    "phase = 'cancelled', updated_at = NOW(), finished_at = NOW() "
                    "WHERE id = %s AND status = 'queued' RETURNING id",
                    (job_id,),
                )
                cancelled = cursor.fetchone() is not None
            connection.commit()
            return cancelled

    def close(self):
        if self.pool:
            self.pool.closeall()
            self.pool = None
