"""
chunking.py — Context-Preserving, Header-Aware & Parent-Child Chunking Pipeline.

Key Features:
1. Header-Aware Splitting: Parses Markdown headers (#, ##, ###, ####) to identify document structure.
2. Hierarchical Breadcrumbs: Constructs breadcrumb context (e.g. `[Doc: X.pdf > Header 1 > Header 2 | Page: Y]`)
   and prepends it to chunk text for dense/sparse retrieval grounding.
3. Table Preservation: Identifies Markdown tables (| ... |) and preserves them as atomic parent chunks.
   For large tables, child chunks retain the table header row so columns are never severed from context.
4. Parent-Child Hierarchy:
   - Parent Chunks (~1000–1500 chars): Full logical section/table sent to LLM context.
   - Child Chunks (~300–450 chars): High-precision units for dense & sparse hybrid search.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import NamedTuple

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
DEFAULT_PARENT_CHUNK_SIZE = 1200
DEFAULT_PARENT_OVERLAP = 150
DEFAULT_CHILD_CHUNK_SIZE = 380
DEFAULT_CHILD_OVERLAP = 60

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]


class ChunkHierarchy(NamedTuple):
    """Container holding parents (for LLM context) and child chunks (for vector/BM25 search)."""
    parents: list[Document]
    children: list[Document]
    parent_map: dict[str, Document]


# ---------------------------------------------------------------------------
# Helper: Markdown Table Detection & Header Extraction
# ---------------------------------------------------------------------------
def _is_table_block(text: str) -> bool:
    """Return True if text block appears to be a Markdown table."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < 2:
        return False
    # Check if lines start and end with '|'
    table_lines = [l for l in lines if l.startswith("|") and l.endswith("|")]
    return len(table_lines) >= 2


def _extract_table_header(table_text: str) -> str:
    """Extract header and separator line from a markdown table."""
    lines = [line.strip() for line in table_text.split("\n") if line.strip()]
    if len(lines) >= 2 and lines[0].startswith("|") and lines[1].startswith("|"):
        return f"{lines[0]}\n{lines[1]}\n"
    return ""


# ---------------------------------------------------------------------------
# Helper: Construct Breadcrumb
# ---------------------------------------------------------------------------
def build_breadcrumb(metadata: dict) -> str:
    """Generate a clean hierarchical breadcrumb string from document metadata.

    Example:
        `[Doc: manual.pdf > Troubleshooting > Error Codes | Page: 12]`
    """
    source = metadata.get("source", "Manual")
    page = metadata.get("page", "?")

    header_parts = []
    for _, header_name in HEADERS_TO_SPLIT_ON:
        val = metadata.get(header_name)
        if val:
            # Clean up header text
            clean_val = re.sub(r"\s+", " ", str(val)).strip()
            header_parts.append(clean_val)

    if header_parts:
        hierarchy_str = " > ".join(header_parts)
        return f"[Doc: {source} > {hierarchy_str} | Page: {page}]"
    else:
        return f"[Doc: {source} | Page: {page}]"


# ---------------------------------------------------------------------------
# Main Hierarchical Chunking Pipeline
# ---------------------------------------------------------------------------
def create_hierarchical_chunks(
    page_documents: list[Document],
    parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
    parent_overlap: int = DEFAULT_PARENT_OVERLAP,
    child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    child_overlap: int = DEFAULT_CHILD_OVERLAP,
) -> ChunkHierarchy:
    """Transform page-level documents into hierarchical Parent-Child chunks.

    1. Applies Markdown header splitting to preserve document outline.
    2. Builds parent chunks (~1200 chars) retaining whole sections and tables.
    3. Builds child chunks (~380 chars) with prepended breadcrumbs and table headers.
    4. Connects each child to its parent via `parent_id`.

    Parameters
    ----------
    page_documents : list[Document]
        Page-level Documents produced by `pdf_loader.py`.

    Returns
    -------
    ChunkHierarchy
        Named tuple with parents list, child chunks list, and a parent_id lookup map.
    """
    if not page_documents:
        return ChunkHierarchy(parents=[], children=[], parent_map={})

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    parent_recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
    )

    child_recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )

    parents: list[Document] = []
    children: list[Document] = []
    parent_map: dict[str, Document] = {}

    for page_doc in page_documents:
        page_text = page_doc.page_content or ""
        if not page_text.strip():
            continue

        base_meta = page_doc.metadata.copy()

        # 1. Split page by Markdown headers if any exist
        try:
            header_splits = markdown_splitter.split_text(page_text)
        except Exception:
            header_splits = []

        if not header_splits:
            # If no markdown headers were recognized, treat full page as a section
            header_splits = [Document(page_content=page_text, metadata={})]

        for h_doc in header_splits:
            # Merge base page metadata (source, page) with header metadata
            section_meta = {**base_meta, **h_doc.metadata}
            section_text = h_doc.page_content

            # 2. Check if section is an intact table
            is_table = _is_table_block(section_text)
            table_header = _extract_table_header(section_text) if is_table else ""

            # 3. Create Parent chunks for this section
            if len(section_text) <= parent_chunk_size:
                section_parents = [Document(page_content=section_text, metadata=section_meta)]
            else:
                section_parents = parent_recursive_splitter.create_documents(
                    texts=[section_text],
                    metadatas=[section_meta],
                )

            for p_doc in section_parents:
                parent_id = str(uuid.uuid4())
                breadcrumb = build_breadcrumb(p_doc.metadata)

                # Format Parent Document with header breadcrumb for clean LLM prompt context
                parent_doc = Document(
                    page_content=p_doc.page_content,
                    metadata={
                        **p_doc.metadata,
                        "parent_id": parent_id,
                        "breadcrumb": breadcrumb,
                        "is_table": is_table,
                    },
                )
                parents.append(parent_doc)
                parent_map[parent_id] = parent_doc

                # 4. Create Child Chunks linked to this Parent
                raw_children = child_recursive_splitter.split_text(p_doc.page_content)
                if not raw_children:
                    raw_children = [p_doc.page_content]

                for child_idx, child_text in enumerate(raw_children):
                    # For tables, preserve the table header row on subsequent splits
                    enriched_content = child_text
                    if is_table and table_header and not child_text.startswith(table_header[:20]):
                        enriched_content = f"{table_header}{child_text}"

                    # Prepend breadcrumb context to child text for search relevance
                    searchable_content = f"{breadcrumb}\n{enriched_content}"

                    child_doc = Document(
                        page_content=searchable_content,
                        metadata={
                            **p_doc.metadata,
                            "parent_id": parent_id,
                            "child_index": child_idx,
                            "breadcrumb": breadcrumb,
                            "is_table": is_table,
                        },
                    )
                    children.append(child_doc)

    logger.info(
        "Hierarchical chunking complete: %d parent context windows, %d searchable child chunks.",
        len(parents),
        len(children),
    )
    return ChunkHierarchy(parents=parents, children=children, parent_map=parent_map)
