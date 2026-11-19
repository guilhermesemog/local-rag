def test_sqlite_insert_search_clear(sqlite_db, fake_embeddings):
    sqlite_db.insert_documents([("hello", {}, fake_embeddings[0])])
    assert sqlite_db.is_empty() is False
    results = sqlite_db.search(fake_embeddings[0], limit=1)
    assert results[0][1] == "hello"
    sqlite_db.delete_all_documents()
    assert sqlite_db.is_empty() is True


def test_postgres_insert_search_clear(postgres_db, fake_embeddings):
    postgres_db.insert_documents([("test-only", {}, fake_embeddings[0])])
    assert postgres_db.search(fake_embeddings[0], limit=1)
    postgres_db.delete_all_documents()
    assert postgres_db.is_empty()
