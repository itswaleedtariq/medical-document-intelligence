class PDFIngestionError(Exception):
    """Base exception for PDF ingestion errors."""


class UnsupportedFileTypeError(PDFIngestionError):
    """Raised when the uploaded file is not an accepted PDF."""


class InvalidPDFError(PDFIngestionError):
    """Raised when uploaded data is not a readable PDF."""


class EmptyPDFError(PDFIngestionError):
    """Raised when an uploaded PDF contains no data or pages."""


class EncryptedPDFError(PDFIngestionError):
    """Raised when a PDF requires a password."""


class PDFTooLargeError(PDFIngestionError):
    """Raised when an upload exceeds the configured size limit."""


class PDFPageLimitError(PDFIngestionError):
    """Raised when a PDF exceeds the configured page limit."""
