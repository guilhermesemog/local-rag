import os
import logging
from functools import lru_cache

from dotenv import load_dotenv

from db.postgres_db import PostgresVectorDB
from db.sqlite_db import SQLiteVectorDB
from errors import DatabaseError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_vector_db() -> PostgresVectorDB | SQLiteVectorDB:
    load_dotenv()
    db_type = os.getenv("DB_OPTION", "sqlite").lower()
    db_path = os.getenv("SQLITE_DB_PATH")
    dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))

    try:
        if db_type == "postgres":
            return PostgresVectorDB(dimension=dimension)

        if db_type == "sqlite" and db_path:
            return SQLiteVectorDB(db_path=db_path, dimension=dimension)

        if db_type == "sqlite":
            return SQLiteVectorDB(dimension=dimension)
        raise ValueError(f"Unsupported DB_OPTION: {db_type}")
    except Exception as exc:
        logger.exception("Vector database initialization failed")
        raise DatabaseError() from exc


def close_vector_db():
    if get_vector_db.cache_info().currsize:
        database = get_vector_db()
        database.close()
    get_vector_db.cache_clear()
