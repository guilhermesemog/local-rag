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
    assert [doc.page_content for doc in cleaned] == ["hello world"]
