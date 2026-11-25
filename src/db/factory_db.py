from __future__ import annotations

import logging
from functools import lru_cache

from errors import DatabaseError
from settings import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_vector_db():
    settings = get_settings()

    try:
        if settings.db_option == "postgres":
            from db.postgres_db import PostgresVectorDB

            return PostgresVectorDB(settings=settings)
        from db.sqlite_db import SQLiteVectorDB

        return SQLiteVectorDB(
            db_path=settings.sqlite_db_path,
            dimension=settings.embedding_dimension,
        )
    except Exception as exc:
        logger.error("Vector database initialization failed (%s)", type(exc).__name__)
        raise DatabaseError() from exc


def close_vector_db():
    if get_vector_db.cache_info().currsize:
        database = get_vector_db()
        database.close()
    get_vector_db.cache_clear()
