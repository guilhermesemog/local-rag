import re
from pathlib import Path
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class DocumentParser:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.document_id = str(uuid4())
        self._loader = PyPDFLoader(file_path)
        self._document = self._loader.load()
        for page in self._document:
            page.metadata.update(
                {
                    "document_id": self.document_id,
                    "document_name": self.file_path.name,
                    "page": int(page.metadata.get("page", 0)) + 1,
                }
            )

    def clean_documents(self, overwrite: bool = True):
        cleaned_docs = []
        for doc in self._document:
            cleaned_text = re.sub(r"\s+", " ", doc.page_content).strip()
            if cleaned_text:
                doc.page_content = cleaned_text
                cleaned_docs.append(doc)

        if overwrite:
            self._document = cleaned_docs

        return cleaned_docs

    def split_documents(self, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        split_docs = text_splitter.split_documents(self._document)
        for chunk_number, doc in enumerate(split_docs, start=1):
            doc.metadata["chunk"] = chunk_number
        return split_docs

    @property
    def document(self):
        return self._document
