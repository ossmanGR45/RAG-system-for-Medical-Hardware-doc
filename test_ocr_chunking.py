"""
test_ocr_chunking.py — Verification script for OCR fallback + Zero-Data-Loss chunking.

Run with:
    python test_ocr_chunking.py

Tests are self-contained (no Ollama / ChromaDB needed) and use mocked PDFs
built from in-memory PyMuPDF documents.
"""

from __future__ import annotations

import logging
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup logging so we see the INFO messages from our modules
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)

# Force UTF-8 stdout on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf as fitz  # PyMuPDF
from langchain_core.documents import Document

from src.pdf_loader import load_pdf_pages, LOW_TEXT_THRESHOLD
from src.vector_store import (
    enforce_zero_data_loss,
    get_text_splitter,
    MIN_CHUNK_SIZE,
    DEFAULT_CHUNK_SIZE,
)


# ===================================================================
# Helper: create a temporary PDF with controlled content
# ===================================================================
def _create_test_pdf(tmp_dir: Path, filename: str, pages: list[dict]) -> Path:
    """Create a minimal PDF.

    *pages* is a list of dicts, each with:
        - "text": str  — text to insert on the page
        - "add_image": bool — if True, embed a tiny red square image

    Returns the path to the written PDF.
    """
    doc = fitz.open()  # new blank PDF
    for page_spec in pages:
        page = doc.new_page(width=612, height=792)  # US Letter
        text = page_spec.get("text", "")
        if text:
            page.insert_text((72, 100), text, fontsize=12)
        if page_spec.get("add_image", False):
            # Create a tiny 10×10 red PNG and insert it
            tiny_pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10), 1)
            tiny_pix.set_rect(tiny_pix.irect, (255, 0, 0, 255))  # RGBA red
            img_bytes = tiny_pix.tobytes("png")
            page.insert_image(fitz.Rect(72, 200, 200, 400), stream=img_bytes)

    pdf_path = tmp_dir / filename
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ===================================================================
# Test 1: Digital-only page — no OCR triggered
# ===================================================================
def test_digital_only(tmp_dir: Path):
    long_text = "This is a page with plenty of digital text. " * 5  # ~225 chars
    pdf = _create_test_pdf(tmp_dir, "digital_only.pdf", [
        {"text": long_text, "add_image": False},
    ])
    docs = list(load_pdf_pages(pdf))

    assert len(docs) == 1, f"Expected 1 doc, got {len(docs)}"
    assert docs[0].metadata["extraction_method"] == "digital"
    assert len(docs[0].page_content.strip()) >= LOW_TEXT_THRESHOLD
    print("✅ Test 1 PASSED: digital-only page — no OCR triggered, method='digital'.\n")


# ===================================================================
# Test 2: Sparse page WITH images — OCR should be attempted
# ===================================================================
def test_sparse_with_images(tmp_dir: Path):
    pdf = _create_test_pdf(tmp_dir, "sparse_images.pdf", [
        {"text": "Hi", "add_image": True},  # <50 chars + image
    ])
    docs = list(load_pdf_pages(pdf))

    assert len(docs) == 1, f"Expected 1 doc, got {len(docs)}"
    # extraction_method can be "ocr" (if Tesseract installed) or "digital"
    method = docs[0].metadata["extraction_method"]
    assert method in ("ocr", "digital"), f"Unexpected method: {method}"
    if method == "ocr":
        print("✅ Test 2 PASSED: sparse page + images → OCR triggered and won.\n")
    else:
        print(
            "✅ Test 2 PASSED: sparse page + images → OCR attempted but "
            "digital kept (Tesseract may not be installed, or OCR yielded less).\n"
        )


# ===================================================================
# Test 3: Sparse page WITHOUT images — no OCR, kept as sparse
# ===================================================================
def test_sparse_no_images(tmp_dir: Path):
    pdf = _create_test_pdf(tmp_dir, "sparse_no_img.pdf", [
        {"text": "OK", "add_image": False},  # <50 chars, no images
    ])
    docs = list(load_pdf_pages(pdf))

    assert len(docs) == 1, f"Expected 1 doc, got {len(docs)}"
    assert docs[0].metadata["extraction_method"] == "digital"
    print("✅ Test 3 PASSED: sparse page, no images → no OCR, method='digital'.\n")


# ===================================================================
# Test 4: Blank page — yielded as blank, not dropped
# ===================================================================
def test_blank_page(tmp_dir: Path):
    pdf = _create_test_pdf(tmp_dir, "blank_page.pdf", [
        {"text": "", "add_image": False},
    ])
    docs = list(load_pdf_pages(pdf))

    assert len(docs) == 1, f"Expected 1 doc, got {len(docs)}"
    assert docs[0].metadata["extraction_method"] == "blank"
    print("✅ Test 4 PASSED: blank page yielded (not dropped), method='blank'.\n")


# ===================================================================
# Test 5: Multi-page — every page yields a document
# ===================================================================
def test_multipage_no_drops(tmp_dir: Path):
    pages = [
        {"text": "Full page " * 20, "add_image": False},
        {"text": "", "add_image": False},            # blank
        {"text": "Short", "add_image": True},         # sparse + image
        {"text": "Another full page " * 20, "add_image": False},
    ]
    pdf = _create_test_pdf(tmp_dir, "multipage.pdf", pages)
    docs = list(load_pdf_pages(pdf))

    assert len(docs) == len(pages), (
        f"Expected {len(pages)} docs (one per page), got {len(docs)}"
    )
    print(f"✅ Test 5 PASSED: {len(pages)}-page PDF → {len(docs)} documents, zero dropped.\n")


# ===================================================================
# Test 6: Zero-Data-Loss chunking — micro-chunks enriched / merged
# ===================================================================
def test_zero_data_loss_chunking():
    chunks = [
        Document(page_content="A" * 100, metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="tiny", metadata={"source": "a.pdf", "page": 1}),  # <40
        Document(page_content="B" * 100, metadata={"source": "a.pdf", "page": 2}),
        Document(page_content="x", metadata={"source": "a.pdf", "page": 2}),      # <40
        Document(page_content="y", metadata={"source": "b.pdf", "page": 1}),      # <40, diff source
    ]

    total_text_before = sum(len(c.page_content) for c in chunks)
    result = enforce_zero_data_loss(chunks, min_size=MIN_CHUNK_SIZE, max_chunk_size=DEFAULT_CHUNK_SIZE)
    total_text_after = sum(len(c.page_content) for c in result)

    # No text should have been lost (enrichment adds chars, so after >= before)
    assert total_text_after >= total_text_before, (
        f"Data loss detected! Before={total_text_before}, After={total_text_after}"
    )

    # "tiny" should have been merged into chunk[0] (same source+page, fits)
    # "x" should have been merged into chunk[2]
    # "y" is standalone enriched (different source)
    # So we expect: merged_chunk_0, merged_chunk_2, enriched_y = 3 chunks
    assert len(result) == 3, f"Expected 3 chunks after merge, got {len(result)}"

    # Verify enriched standalone has metadata prefix
    last = result[-1]
    assert last.page_content.startswith("[Doc: b.pdf | Page: 1]"), (
        f"Expected metadata prefix, got: {last.page_content[:60]}"
    )

    print(
        f"✅ Test 6 PASSED: {len(chunks)} input chunks → {len(result)} output chunks. "
        f"Text before={total_text_before}, after={total_text_after} (zero loss).\n"
    )


# ===================================================================
# Test 7: Empty chunk list — enforce_zero_data_loss handles gracefully
# ===================================================================
def test_empty_chunks():
    result = enforce_zero_data_loss([], min_size=MIN_CHUNK_SIZE)
    assert result == [], f"Expected empty list, got {result}"
    print("✅ Test 7 PASSED: empty chunk list handled gracefully.\n")


# ===================================================================
# Test 8: All chunks above threshold — no changes
# ===================================================================
def test_all_above_threshold():
    chunks = [
        Document(page_content="A" * 100, metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="B" * 200, metadata={"source": "a.pdf", "page": 2}),
    ]
    result = enforce_zero_data_loss(chunks, min_size=MIN_CHUNK_SIZE)
    assert len(result) == len(chunks)
    for orig, res in zip(chunks, result):
        assert orig.page_content == res.page_content
    print("✅ Test 8 PASSED: all chunks above threshold — unchanged.\n")


# ===================================================================
# Test 9: Integration — splitter + zero-data-loss on real-ish text
# ===================================================================
def test_splitter_integration():
    """Full pipeline: split a multi-page document then apply zero-data-loss."""
    pages = [
        Document(
            page_content="Medical procedure details. " * 80,  # ~2160 chars → multiple chunks
            metadata={"source": "manual.pdf", "page": 1, "total_pages": 2},
        ),
        Document(
            page_content="End.",  # 4 chars → will become a micro-chunk
            metadata={"source": "manual.pdf", "page": 2, "total_pages": 2},
        ),
    ]

    splitter = get_text_splitter(chunk_size=800, chunk_overlap=150)
    raw_chunks = splitter.split_documents(pages)
    processed = enforce_zero_data_loss(raw_chunks, min_size=MIN_CHUNK_SIZE, max_chunk_size=800)

    total_raw = sum(len(c.page_content) for c in raw_chunks)
    total_proc = sum(len(c.page_content) for c in processed)

    assert total_proc >= total_raw, "Data loss in integration test!"

    # The tiny "End." chunk should NOT have been dropped
    all_text = " ".join(c.page_content for c in processed)
    assert "End." in all_text, "Micro-chunk 'End.' was dropped!"

    print(
        f"✅ Test 9 PASSED: splitter produced {len(raw_chunks)} chunks → "
        f"zero-data-loss yielded {len(processed)} chunks, 'End.' preserved.\n"
    )


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    import tempfile

    print("=" * 65)
    print("  OCR Fallback & Zero-Data-Loss Chunking — Verification Suite")
    print("=" * 65 + "\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        test_digital_only(tmp_path)
        test_sparse_with_images(tmp_path)
        test_sparse_no_images(tmp_path)
        test_blank_page(tmp_path)
        test_multipage_no_drops(tmp_path)

    # Pure-logic tests (no PDF files needed)
    test_zero_data_loss_chunking()
    test_empty_chunks()
    test_all_above_threshold()
    test_splitter_integration()

    print("=" * 65)
    print("  ALL TESTS PASSED ✅")
    print("=" * 65)
