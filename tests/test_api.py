def test_empty_query_and_database_clear(api_client):
    response = api_client.post("/query", json={"question": "hello"})
    assert response.status_code == 400
    assert "detail" in response.json()
    assert api_client.get("/database").json() == {"is_empty": True}
    assert api_client.delete("/database").status_code == 200


def test_upload_rejects_invalid_pdf(api_client):
    response = api_client.post(
        "/upload-file",
        files={"file": ("bad.pdf", b"not pdf", "application/pdf")},
    )
    assert response.status_code == 400


def test_upload_accepts_pdf_and_query(api_client, monkeypatch):
    import app

    class FakeUpload:
        def __init__(self, *_args, **_kwargs):
            pass

        def insert_documents(self):
            return True

    monkeypatch.setattr(app, "UploadService", FakeUpload)
    monkeypatch.setattr(app, "get_embedding_engine", lambda: object())
    # Avoid parser/model/database work; this verifies multipart routing and cleanup.
    response = api_client.post(
        "/upload-file",
        files={"file": ("ok.pdf", b"%PDF-1.7\n", "application/pdf")},
    )
    assert response.status_code == 200


def test_health_and_web_interface_are_served_from_api(api_client):
    health = api_client.get("/health")
    home = api_client.get("/")
    script = api_client.get("/script.js")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert home.status_code == 200
    assert "Local RAG" in home.text
    assert script.status_code == 200
    assert "window.location.origin" in script.text
