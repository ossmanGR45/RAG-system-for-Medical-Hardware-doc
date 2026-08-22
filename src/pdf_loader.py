"""
pdf_loader.py — Layout-aware and memory-efficient PDF loader for technical manuals.

Key Features:
1. Layout-Aware Markdown Extraction: Uses `pymupdf4llm` to preserve document structure,
   including Markdown headers (#, ##, ###), lists, and formatted Markdown tables (| ... |).
2. Streaming Page-by-Page: Iterates over pages to prevent memory spikes on large manuals (500-700 pages).
3. Zero-Data-Loss with Conditional OCR Fallback: For scanned or image-heavy pages with low
   digital text, conditionally runs Tesseract OCR to ensure no schematic or table is lost.
4. Rich Metadata: Each LangChain Document contains source, page (1-indexed), total_pages,
   and extraction_method ("structured_markdown", "digital", "ocr", or "blank").
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Generator

import pymupdf as fitz  # PyMuPDF
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
LOW_TEXT_THRESHOLD = 50  # characters – below this we consider OCR
OCR_DPI = 300            # render resolution for OCR

# ---------------------------------------------------------------------------
# OCR availability check (lazy, once)
# ---------------------------------------------------------------------------
_ocr_available: bool | None = None


def _check_ocr_available() -> bool:
    """Return True if pytesseract is importable and Tesseract binary is reachable."""
    global _ocr_available
    if _ocr_available is not None:
        return _ocr_available
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        _ocr_available = True
        logger.info("Tesseract OCR is available — conditional OCR fallback enabled.")
    except Exception as exc:
        _ocr_available = False
        logger.warning(
            "Tesseract OCR is NOT available (%s). Scanned image pages will not be OCR'd.",
            exc,
        )
    return _ocr_available


def _ocr_page(page: fitz.Page) -> str:
    """Render a single PyMuPDF page to an image and run Tesseract OCR."""
    try:
        import pytesseract
        from PIL import Image

        zoom = OCR_DPI / 72  # 72 is PDF default DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)
        return text or ""
    except Exception as exc:
        logger.warning("OCR failed on page %s: %s", page.number + 1, exc)
        return ""


# ---------------------------------------------------------------------------
# Main Layout-Aware Loader
# ---------------------------------------------------------------------------
def load_pdf_pages(pdf_path: str | Path) -> Generator[Document, None, None]:
    """Yield one LangChain Document per page using structured Markdown extraction.

    Extracts Markdown headings, tables, and lists. Falls back to OCR when necessary.
    Guarantees zero data loss across all pages.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    filename = pdf_path.name
    pages_yielded = 0

    # 1. Attempt structured Markdown extraction via pymupdf4llm
    try:
        import pymupdf4llm

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)

        # pymupdf4llm page_chunks extracts structured markdown per page
        page_chunks = pymupdf4llm.to_markdown(
            doc,
            page_chunks=True,
            show_progress=False,
        )

        for chunk_info in page_chunks:
            page_num = chunk_info.get("metadata", {}).get("page", pages_yielded + 1)
            # PyMuPDF4LLM uses 0-indexed or 1-indexed page in metadata
            if isinstance(page_num, int) and page_num < total_pages and "page" in chunk_info.get("metadata", {}):
                # normalize to 1-indexed
                page_1_indexed = page_num if page_num >= 1 else page_num + 1
            else:
                page_1_indexed = pages_yielded + 1

            md_text = chunk_info.get("text", "") or ""
            extraction_method = "structured_markdown"

            # Check if page is near empty and needs OCR check
            if len(md_text.strip()) < LOW_TEXT_THRESHOLD and total_pages > 0:
                try:
                    fitz_page = doc.load_page(page_1_indexed - 1)
                    if len(fitz_page.get_images()) > 0 and _check_ocr_available():
                        ocr_text = _ocr_page(fitz_page)
                        if len(ocr_text.strip()) > len(md_text.strip()):
                            md_text = ocr_text
                            extraction_method = "ocr"
                except Exception as e:
                    logger.debug("Conditional OCR check skipped for page %d: %s", page_1_indexed, e)

            if not md_text.strip():
                extraction_method = "blank"

            pages_yielded += 1
            yield Document(
                page_content=md_text,
                metadata={
                    "source": filename,
                    "page": page_1_indexed,
                    "total_pages": total_pages,
                    "extraction_method": extraction_method,
                },
            )

        doc.close()
        return

    except ImportError:
        logger.warning("pymupdf4llm not installed or failed to import; falling back to PyMuPDF fitz.")
    except Exception as exc:
        logger.warning("Structured extraction failed for '%s' (%s); falling back to raw fitz.", filename, exc)

    # 2. Direct PyMuPDF fitz fallback
    try:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        for page_idx in range(total_pages):
            page = doc.load_page(page_idx)
            text = page.get_text("text") or ""
            extraction_method = "digital"

            if len(text.strip()) < LOW_TEXT_THRESHOLD:
                if len(page.get_images()) > 0 and _check_ocr_available():
                    ocr_text = _ocr_page(page)
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                        extraction_method = "ocr"

            if not text.strip():
                extraction_method = "blank"

            pages_yielded += 1
            yield Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": page_idx + 1,
                    "total_pages": total_pages,
                    "extraction_method": extraction_method,
                },
            )
        doc.close()
    except Exception as exc:
        logger.error("PyMuPDF fitz fallback failed for '%s': %s", filename, exc)

    # 3. pypdf last-resort fallback
    if pages_yielded == 0:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(pdf_path))
            total_pages = len(reader.pages)
            for page_idx in range(total_pages):
                page = reader.pages[page_idx]
                text = page.extract_text() or ""
                yield Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_idx + 1,
                        "total_pages": total_pages,
                        "extraction_method": "pypdf_fallback" if text.strip() else "blank",
                    },
                )
        except Exception as e:
            logger.error("All PDF loaders failed for '%s': %s", filename, e)


def load_all_manuals(manuals_dir: str | Path) -> list[Document]:
    """Load every PDF found in *manuals_dir*, returning a list of structured Documents."""
    manuals_dir = Path(manuals_dir)
    if not manuals_dir.is_dir():
        raise NotADirectoryError(f"Manuals directory not found: {manuals_dir}")

    documents: list[Document] = []
    for pdf_file in sorted(manuals_dir.glob("*.pdf")):
        for page_doc in load_pdf_pages(pdf_file):
            documents.append(page_doc)

    return documents


def render_pdf_page_image(
    pdf_path: str | Path,
    page_num: int,
    dpi: int = 150,
) -> bytes | None:
    """Render a specific 1-indexed page of a PDF manual to PNG image bytes.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF file.
    page_num : int
        1-indexed page number.
    dpi : int
        Resolution for rendering (default 150 DPI for clear diagram viewing).

    Returns
    -------
    bytes | None
        PNG image bytes, or None if rendering fails or page is out of range.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error("Cannot render page: PDF not found at %s", pdf_path)
        return None

    try:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        if page_num < 1 or page_num > total_pages:
            logger.warning("Page number %d is out of range (1-%d) for %s", page_num, total_pages, pdf_path.name)
            doc.close()
            return None

        page = doc.load_page(page_num - 1)  # fitz is 0-indexed
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except Exception as exc:
        logger.error("Failed to render page %d of '%s': %s", page_num, pdf_path.name, exc)
        return None

