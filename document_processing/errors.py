class PDFValidationError(ValueError):
    """The uploaded file is not a usable PDF."""


class ExtractionError(RuntimeError):
    """Text extraction failed for a reason other than invalid input."""


class DuplicateDocumentError(ValueError):
    """The document (by file hash or by normalised text) is already in the library."""

    def __init__(self, message: str, existing_doc_id: str | None = None):
        super().__init__(message)
        self.existing_doc_id = existing_doc_id
