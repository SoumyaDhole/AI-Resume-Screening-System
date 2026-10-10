"""PDF text extraction helpers for resume files."""

from __future__ import annotations

import pymupdf


def validate_resume_text(text: str) -> bool:
    """Return whether extracted text contains usable non-whitespace content."""
    return bool(text and text.strip())


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF, raising ValueError for invalid or empty files."""
    if not pdf_bytes:
        raise ValueError("The uploaded file is empty.")

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            text = "\n".join(page.get_text() for page in document)
    except (pymupdf.FileDataError, RuntimeError, ValueError) as error:
        raise ValueError("This file could not be read as a PDF.") from error

    if not validate_resume_text(text):
        raise ValueError("No readable text was found. The PDF may be scanned or empty.")
    return text.strip()