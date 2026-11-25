"""Application errors that can be safely mapped to HTTP responses."""


class AppError(Exception):
    status_code = 500
    public_message = "An unexpected error occurred."


class DocumentParseError(AppError):
    status_code = 400
    public_message = "The PDF could not be processed."


class OCRUnavailableError(AppError):
    status_code = 503
    public_message = (
        "OCR is enabled, but its Python or system dependencies are unavailable."
    )


class EmptyDocumentError(AppError):
    status_code = 400
    public_message = "The PDF contains no processable text."


class NoMatchingDocumentsError(AppError):
    status_code = 400
    public_message = (
        "No chunks matched the selected documents. Adjust the document filter."
    )


class EmbeddingError(AppError):
    status_code = 500
    public_message = "Document processing failed while generating embeddings."


class DatabaseError(AppError):
    status_code = 503
    public_message = "The vector database is unavailable."


class IndexCompatibilityError(AppError):
    status_code = 409
    public_message = (
        "The stored embedding index is incompatible or unidentified. "
        "Clear the vector database and reindex documents with the configured model."
    )


class LLMProviderError(AppError):
    status_code = 502
    public_message = "The language model provider failed to generate a response."
