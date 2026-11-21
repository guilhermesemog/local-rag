import pytest
from pydantic import ValidationError

from settings import Settings


def test_basic_settings_default_to_dummy_and_sqlite():
    settings = Settings(_env_file=None)
    assert settings.llm_model == "dummy"
    assert settings.db_option == "sqlite"
    assert settings.embedding_dimension == 384


def test_gemini_requires_credentials():
    with pytest.raises(ValidationError, match="GEMINI_API_KEY"):
        Settings(_env_file=None, llm_model="gemini")


def test_postgres_requires_password():
    with pytest.raises(ValidationError, match="DB_PASSWORD"):
        Settings(_env_file=None, db_option="postgres")


def test_invalid_chunk_configuration_is_rejected():
    with pytest.raises(ValidationError, match="CHUNK_SIZE"):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)
