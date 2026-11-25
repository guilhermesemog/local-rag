import json
import logging

from db.factory_db import get_vector_db
from document_parser import DocumentParser
from embedding_engine import EmbeddingEngine
from errors import (
    AppError,
    DatabaseError,
    DocumentParseError,
    EmptyDocumentError,
    LLMProviderError,
)
from llm.base import LLM
from settings import Settings, get_settings
from templates import chat_template

logger = logging.getLogger(__name__)


def build_llm(settings: Settings | None = None) -> LLM:
    settings = settings or get_settings()

    if settings.llm_model == "gemini":
        from llm.gemini import GeminiLLM

        return GeminiLLM(
            model_name=settings.gemini_model,
            api_key=settings.gemini_api_key.get_secret_value(),
            request_timeout=settings.gemini_request_timeout,
            retries=settings.gemini_retries,
        )
    if settings.llm_model == "ollama":
        from llm.ollama import OllamaLLM

        return OllamaLLM(
            model_name=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.ollama_temperature,
            request_timeout=settings.ollama_request_timeout,
        )
    if settings.llm_model == "dummy":
        from llm.dummy import DummyLLM

        return DummyLLM()
    if settings.llm_model == "transformers":
        from llm.transformers import TransformersLLM

        return TransformersLLM(model_name=settings.transformers_model)
    raise ValueError(f"Unsupported LLM_MODEL: {settings.llm_model}")


class UploadService:
    def __init__(self, file_path: str, db=None, engine=None, progress_callback=None):
        self.file_path = file_path
        settings = get_settings()
        self.parser = DocumentParser(file_path, settings=settings)
        self.engine = engine or EmbeddingEngine(settings=settings)
        self.db = db or get_vector_db()
        self.progress_callback = progress_callback or (lambda _phase, _progress: None)

    def insert_documents(self):
        self.progress_callback("checking_index", 5)
        self.db.ensure_embedding_identity(self.engine.model_name, self.engine.dimension)
        try:
            if self.parser is None:
                raise DocumentParseError()
            self.progress_callback("parsing_pdf", 15)
            cleaned_docs = self.parser.clean_documents()
        except Exception as exc:
            if isinstance(exc, AppError):
                raise
            logger.error("PDF parsing failed (%s)", type(exc).__name__)
            raise DocumentParseError() from exc

        if not cleaned_docs:
            raise EmptyDocumentError()

        try:
            self.progress_callback("splitting_text", 35)
            content = self.parser.split_documents()
        except Exception as exc:
            logger.error("PDF splitting failed (%s)", type(exc).__name__)
            raise DocumentParseError() from exc

        texts = [doc.page_content for doc in content]
        self.progress_callback("generating_embeddings", 55)
        embeddings = self.engine.generate_embeddings(texts)

        data_to_insert = []
        for i, doc in enumerate(content):
            metadata = {
                key: doc.metadata[key]
                for key in (
                    "document_id",
                    "document_name",
                    "page",
                    "chunk",
                    "extraction_method",
                )
                if key in doc.metadata
            }
            data_to_insert.append(
                (doc.page_content, json.dumps(metadata), embeddings[i])
            )

        try:
            self.progress_callback("indexing_chunks", 85)
            self.db.insert_documents(data_to_insert)
        except Exception as exc:
            logger.error("Document insertion failed (%s)", type(exc).__name__)
            raise DatabaseError() from exc
        return True


class LLMService:
    def __init__(
        self, llm: LLM, embedding_engine: EmbeddingEngine | None = None, db=None
    ):
        self.llm = llm
        self.db = db or get_vector_db()
        self.engine = embedding_engine or EmbeddingEngine(settings=get_settings())
        self.progress_callback = lambda _phase, _progress: None

    def _query(self, query, top_k=5):
        query_embedding = self.engine.generate_embeddings([query])[0]
        try:
            return self.db.search(query_embedding, limit=top_k)
        except Exception as exc:
            logger.error("Document search failed (%s)", type(exc).__name__)
            raise DatabaseError() from exc

    @staticmethod
    def _sources(similar_docs):
        return [
            {
                "document_id": doc.metadata.get("document_id"),
                "document_name": doc.metadata.get("document_name"),
                "page": doc.metadata.get("page"),
                "chunk": doc.metadata.get("chunk"),
                "text": doc.content,
                "similarity": doc.similarity,
            }
            for doc in similar_docs
        ]

    def _retrieve(self, question: str):
        self.progress_callback("checking_index", 5)
        self.db.ensure_embedding_identity(self.engine.model_name, self.engine.dimension)
        try:
            if self.db.is_empty():
                raise EmptyDocumentError()
        except EmptyDocumentError:
            raise
        except Exception as exc:
            logger.error(
                "Vector database availability check failed (%s)", type(exc).__name__
            )
            raise DatabaseError() from exc

        self.progress_callback("searching_documents", 30)
        similar_docs = self._query(question, top_k=6)
        context = "\n".join([doc.content for doc in similar_docs])

        prompt = chat_template.format(context=context, question=question)
        return prompt, self._sources(similar_docs)

    def answer(self, question: str):
        prompt, sources = self._retrieve(question)
        try:
            self.progress_callback("generating_answer", 65)
            response = self.llm.generate_text(prompt)
        except Exception as exc:
            logger.error("Language model generation failed (%s)", type(exc).__name__)
            raise LLMProviderError() from exc

        self.progress_callback("formatting_result", 90)
        return {"results": response, "sources": sources}

    def answer_stream(self, question: str):
        """Yield sources first, then answer fragments as the provider produces them."""
        prompt, sources = self._retrieve(question)
        yield {"type": "sources", "sources": sources}
        self.progress_callback("generating_answer", 65)
        try:
            for text in self.llm.stream_text(prompt):
                if text:
                    yield {"type": "token", "text": text}
        except Exception as exc:
            logger.error("Language model stream failed (%s)", type(exc).__name__)
            raise LLMProviderError() from exc

        self.progress_callback("formatting_result", 90)


class DBService:
    def __init__(self, db=None):
        self.db = db or get_vector_db()

    def check_empty(self) -> bool:
        try:
            return self.db.is_empty()
        except Exception as exc:
            logger.error("Vector database status check failed (%s)", type(exc).__name__)
            raise DatabaseError() from exc

    def clear_database(self):
        try:
            self.db.delete_all_documents()
        except Exception as exc:
            logger.error("Vector database cleanup failed (%s)", type(exc).__name__)
            raise DatabaseError() from exc
