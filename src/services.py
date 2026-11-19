import json
import os
import logging

from db.factory_db import get_vector_db
from embedding_engine import EmbeddingEngine
from document_parser import DocumentParser

from templates import chat_template

from llm.gemini import GeminiLLM
from llm.ollama import OllamaLLM
from llm.dummy import DummyLLM
from llm.transformers import TransformersLLM
from llm.base import LLM
from errors import (
    DatabaseError,
    DocumentParseError,
    EmptyDocumentError,
    LLMProviderError,
)

logger = logging.getLogger(__name__)

def build_llm(model_type: str | None = None, **kwargs) -> LLM:
    model_type = model_type or os.getenv("LLM_MODEL", "dummy")

    if model_type == "gemini":
        model_name = kwargs.get("model_name") or os.getenv("GEMINI_MODEL")
        if not model_name:
            raise ValueError("GEMINI_MODEL must be configured when LLM_MODEL=gemini")
        return GeminiLLM(model_name=model_name)
    elif model_type == "ollama":
        model_name = kwargs.get("model_name") or os.getenv("OLLAMA_MODEL", "llama3")
        base_url = kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        temperature = float(kwargs.get("temperature", os.getenv("OLLAMA_TEMPERATURE", "0.7")))
        return OllamaLLM(model_name=model_name, base_url=base_url, temperature=temperature)
    elif model_type == "dummy":
        return DummyLLM()
    elif model_type in ("transformers", "hugging-face"):
        model_name = kwargs.get("model_name") or os.getenv("TRANSFORMERS_MODEL")
        if not model_name:
            raise ValueError("TRANSFORMERS_MODEL must be configured when LLM_MODEL=transformers")
        return TransformersLLM(model_name=model_name)
    else:
        return TransformersLLM(model_name=kwargs.get("model_name", model_type))

class UploadService:
    def __init__(self, file_path: str, db = None, engine = None):
        self.file_path = file_path
        self.parser = DocumentParser(file_path)
        self.engine = engine or EmbeddingEngine()
        self.db = db or get_vector_db()

    def insert_documents(self):
        try:
            if self.parser is None:
                raise DocumentParseError()
            cleaned_docs = self.parser.clean_documents()
        except Exception as exc:
            if isinstance(exc, DocumentParseError):
                raise
            logger.exception("PDF parsing failed")
            raise DocumentParseError() from exc

        if not cleaned_docs:
            raise EmptyDocumentError()

        try:
            content = self.parser.split_documents()
        except Exception as exc:
            logger.exception("PDF splitting failed")
            raise DocumentParseError() from exc

        texts = [doc.page_content for doc in content]
        embeddings = self.engine.generate_embeddings(texts)

        data_to_insert = []
        for i, doc in enumerate(content):
            data_to_insert.append((doc.page_content, json.dumps({}), embeddings[i]))

        try:
            self.db.insert_documents(data_to_insert)
        except Exception as exc:
            logger.exception("Document insertion failed")
            raise DatabaseError() from exc
        return True


class LLMService:
    def __init__(self, llm: LLM, embedding_engine: EmbeddingEngine | None = None, db = None):
        self.llm = llm
        self.db = db or get_vector_db()
        self.engine = embedding_engine or EmbeddingEngine()
        
    def _query(self, query, top_k=5):
        query_embedding = self.engine.generate_embeddings([query])[0]
        try:
            return self.db.search(query_embedding, limit=top_k)
        except Exception as exc:
            logger.exception("Document search failed")
            raise DatabaseError() from exc

    def answer(self, question: str):
        try:
            if self.db.is_empty():
                raise EmptyDocumentError()
        except EmptyDocumentError:
            raise
        except Exception as exc:
            logger.exception("Vector database availability check failed")
            raise DatabaseError() from exc

        similar_docs = self._query(question, top_k=6)
        context = "\n".join([doc[1] for doc in similar_docs])

        prompt = chat_template.format(context=context, question=question)
        try:
            response = self.llm.generate_text(prompt)
        except Exception as exc:
            logger.exception("Language model generation failed")
            raise LLMProviderError() from exc

        return response


class DBService:
    def __init__(self, db = None):
        self.db = db or get_vector_db()

    def check_empty(self) -> bool:
        try:
            return self.db.is_empty()
        except Exception as exc:
            logger.exception("Vector database status check failed")
            raise DatabaseError() from exc
        
    def clear_database(self):
        try:
            self.db.delete_all_documents()
        except Exception as exc:
            logger.exception("Vector database cleanup failed")
            raise DatabaseError() from exc
