import json
import os
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from psycopg2 import pool
from psycopg2.extras import Json, execute_values

from db.vector_db import SearchResult, VectorDB
from errors import IndexCompatibilityError

MIGRATION_DIR = Path(__file__).parent / "migrations" / "postgres"


class PostgresVectorDB(VectorDB):
    def __init__(self, dimension: int = 384):
        load_dotenv()
        self.dimension = dimension
        self.pool = pool.ThreadedConnectionPool(
            int(os.getenv("POSTGRES_POOL_MIN", "1")),
            int(os.getenv("POSTGRES_POOL_MAX", "8")),
            dbname=os.getenv("DB_NAME", "semantic_search_db"),
            user=os.getenv("DB_USER", "vector_admin"),
            password=os.getenv("DB_PASSWORD", "vector_password"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5433"),
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
        with self._connection() as connection:
            with connection.cursor() as cursor:
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
                    sql = migration.read_text().replace(
                        "${DIMENSION}", str(self.dimension)
                    )
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
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "LOCK TABLE embeddings.documents IN ACCESS EXCLUSIVE MODE"
                )
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
                    cursor.execute(
                        "DROP INDEX IF EXISTS embeddings.document_embedding_idx"
                    )
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

    def insert_documents(self, data: list[tuple]):
        rows = []
        for content, metadata, embedding in data:
            self.validate_vector_dimension(embedding)
            parsed_metadata = (
                json.loads(metadata) if isinstance(metadata, str) else metadata
            )
            rows.append((content, Json(parsed_metadata), embedding))
        with self._connection() as connection:
            with connection.cursor() as cursor:
                execute_values(
                    cursor,
                    "INSERT INTO embeddings.documents(content, metadata, embedding) VALUES %s",
                    rows,
                )
            connection.commit()

    def search(self, query_vector: list[float], limit: int = 5) -> list[SearchResult]:
        self.validate_vector_dimension(query_vector)
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT content, metadata, 1 - (embedding <=> %s::vector) "
                    "FROM embeddings.documents ORDER BY embedding <=> %s::vector LIMIT %s",
                    (query_vector, query_vector, limit),
                )
                rows = cursor.fetchall()
            connection.rollback()
        return [SearchResult(row[0], row[1] or {}, float(row[2])) for row in rows]

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

    def close(self):
        if self.pool:
            self.pool.closeall()
            self.pool = None
