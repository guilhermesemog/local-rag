from langchain_core.documents import Document

from document_parser import DocumentParser


def test_parser_loads_pdf_and_splits(sample_pdf):
    parser = DocumentParser(str(sample_pdf))
    cleaned = parser.clean_documents()
    assert cleaned
    chunks = parser.split_documents(chunk_size=300, chunk_overlap=30)
    assert chunks
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all(chunk.metadata["document_name"] == sample_pdf.name for chunk in chunks)
    assert all(chunk.metadata["page"] >= 1 for chunk in chunks)
    assert all(chunk.metadata["chunk"] >= 1 for chunk in chunks)


def test_parser_cleans_whitespace():
    parser = DocumentParser.__new__(DocumentParser)
    parser._document = [
        Document(page_content="  hello\n\t world "),
        Document(page_content="  "),
    ]
    cleaned = parser.clean_documents()
    assert [doc.page_content for doc in cleaned] == ["hello\nworld"]


def test_cleaning_preserves_table_columns():
    parser = DocumentParser.__new__(DocumentParser)
    parser._document = [Document(page_content="Name   Count\nItem A   3")]
    cleaned = parser.clean_documents()
    assert cleaned[0].page_content == "Name Count\nItem A 3"


def test_chunk_configuration_rejects_invalid_overlap(monkeypatch):
    parser = DocumentParser.__new__(DocumentParser)
    parser._document = []
    from settings import Settings

    parser.settings = Settings(_env_file=None)
    import pytest

    with pytest.raises(ValueError, match="overlap"):
        parser.split_documents(chunk_size=100, chunk_overlap=100)
