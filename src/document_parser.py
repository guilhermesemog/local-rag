import os
import re
from pathlib import Path
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from errors import OCRUnavailableError


class DocumentParser:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.document_id = str(uuid4())
        self.ocr_enabled = os.getenv("OCR_ENABLED", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        self.ocr_language = os.getenv("OCR_LANG", "eng")
        self._loader = PyPDFLoader(file_path)
        self._document = self._loader.load()
        if self.ocr_enabled:
            self._ocr_empty_pages()
        for page in self._document:
            page.metadata.update(
                {
                    "document_id": self.document_id,
                    "document_name": self.file_path.name,
                    "page": int(page.metadata.get("page", 0)) + 1,
                    "extraction_method": page.metadata.get("extraction_method", "text"),
                }
            )

    def clean_documents(self, overwrite: bool = True):
        cleaned_docs = []
        for doc in self._document:
            cleaned_text = re.sub(r"[ \t\f\v]+", " ", doc.page_content)
            cleaned_text = re.sub(r" *\n *", "\n", cleaned_text).strip()
            if cleaned_text:
                doc.page_content = cleaned_text
                cleaned_docs.append(doc)

        if overwrite:
            self._document = cleaned_docs

        return cleaned_docs

    def split_documents(self, chunk_size=None, chunk_overlap=None):
        chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1000"))
        chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else int(os.getenv("CHUNK_OVERLAP", "200"))
        )
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "Chunk size must be positive and overlap smaller than size."
            )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        split_docs = text_splitter.split_documents(self._document)
        unique_chunks = []
        seen = set()
        for doc in split_docs:
            text = doc.page_content.strip()
            deduplication_key = (doc.metadata.get("page"), text)
            if not text or deduplication_key in seen:
                continue
            seen.add(deduplication_key)
            doc.page_content = text
            unique_chunks.append(doc)

        for chunk_number, doc in enumerate(unique_chunks, start=1):
            doc.metadata["chunk"] = chunk_number
        return unique_chunks

    def _ocr_empty_pages(self):
        pages_to_ocr = [
            index
            for index, page in enumerate(self._document)
            if not page.page_content.strip()
        ]
        if not pages_to_ocr:
            return
        try:
            import pypdfium2
            import pytesseract

            pdf = pypdfium2.PdfDocument(str(self.file_path))
            for index in pages_to_ocr:
                bitmap = pdf[index].render(scale=2)
                image = bitmap.to_pil()
                text = pytesseract.image_to_string(image, lang=self.ocr_language)
                self._document[index].page_content = text
                self._document[index].metadata["extraction_method"] = "ocr"
            pdf.close()
        except Exception as exc:
            raise OCRUnavailableError() from exc

    @property
    def document(self):
        return self._document
