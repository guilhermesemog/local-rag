import os
from dotenv import load_dotenv

from db.postgres_db import PostgresVectorDB
from db.sqlite_db import SQLiteVectorDB
from errors import DatabaseError
import logging

logger = logging.getLogger(__name__)


def get_vector_db() -> PostgresVectorDB | SQLiteVectorDB:
    load_dotenv()
    db_type = os.getenv("DB_OPTION", "sqlite").lower()
    db_path = os.getenv("SQLITE_DB_PATH")

    try:
        if db_type == "postgres":
            return PostgresVectorDB()

        if db_type == "sqlite" and db_path:
            return SQLiteVectorDB(db_path=db_path)

        return SQLiteVectorDB()
    except Exception as exc:
        logger.exception("Vector database initialization failed")
        raise DatabaseError() from exc
