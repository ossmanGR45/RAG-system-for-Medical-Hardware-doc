"""
vector_store.py — Vector store & Hierarchical Pipeline Manager for ChromaDB & BM25.

Key Features:
1. Integrates with `src.chunking` to create Header-Aware Parent-Child chunks.
2. Embeds Child Chunks into ChromaDB using Ollama (`nomic-embed-text`).
3. Persists Parent Documents & Child Chunks to disk (JSON) to avoid re-indexing.
4. Builds and exposes `HybridParentRetriever` (Dense + BM25 + RRF + Cross-Encoder).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from src.chunking import ChunkHierarchy, create_hierarchical_chunks
from src.hybrid_retriever import HybridParentRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
DEFAULT_COLLECTION_NAME = "medical_manuals"
PARENT_STORE_FILENAME = "parent_store.json"
CHILD_STORE_FILENAME = "child_store.json"


def get_embeddings(model: str = DEFAULT_EMBEDDING_MODEL) -> OllamaEmbeddings:
    """Return an OllamaEmbeddings instance for the given model with base_url."""
    return OllamaEmbeddings(model=model, base_url="http://localhost:11434")


def chroma_store_exists(
    persist_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> bool:
    """Check whether a Chroma store and parent document cache already exist on disk."""
    persist_path = Path(persist_dir)
    sqlite_file = persist_path / "chroma.sqlite3"
    parent_file = persist_path / PARENT_STORE_FILENAME

    if not sqlite_file.exists() or not parent_file.exists():
        return False

    try:
        client = chromadb.PersistentClient(path=str(persist_path))
        collection = client.get_collection(name=collection_name)
        return collection.count() > 0
    except Exception:
        return False


def _save_store_cache(
    persist_dir: str,
    parent_map: dict[str, Document],
    child_documents: list[Document],
):
    """Save parent map and child documents as JSON for instant reload."""
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    parent_data = {
        pid: {"content": doc.page_content, "metadata": doc.metadata}
        for pid, doc in parent_map.items()
    }
    child_data = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in child_documents
    ]

    with open(persist_path / PARENT_STORE_FILENAME, "w", encoding="utf-8") as f:
        json.dump(parent_data, f, ensure_ascii=False, indent=2)

    with open(persist_path / CHILD_STORE_FILENAME, "w", encoding="utf-8") as f:
        json.dump(child_data, f, ensure_ascii=False, indent=2)


def _load_store_cache(persist_dir: str) -> tuple[dict[str, Document], list[Document]]:
    """Load parent map and child documents from disk cache."""
    persist_path = Path(persist_dir)
    parent_file = persist_path / PARENT_STORE_FILENAME
    child_file = persist_path / CHILD_STORE_FILENAME

    if not parent_file.exists() or not child_file.exists():
        return {}, []

    with open(parent_file, "r", encoding="utf-8") as f:
        parent_data = json.load(f)

    with open(child_file, "r", encoding="utf-8") as f:
        child_data = json.load(f)

    parent_map = {
        pid: Document(page_content=item["content"], metadata=item["metadata"])
        for pid, item in parent_data.items()
    }

    child_documents = [
        Document(page_content=item["content"], metadata=item["metadata"])
        for item in child_data
    ]

    return parent_map, child_documents


def build_hierarchical_hybrid_pipeline(
    page_documents: list[Document],
    persist_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[Chroma, HybridParentRetriever]:
    """Index page documents into hierarchical Parent-Child stores and build a Hybrid Retriever.

    If the store is already cached on disk, loads directly from disk.
    """
    embeddings = get_embeddings(embedding_model)

    if chroma_store_exists(persist_dir, collection_name):
        logger.info("Loading existing Chroma vector store & parent cache from '%s'...", persist_dir)
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        parent_map, child_documents = _load_store_cache(persist_dir)
        retriever = HybridParentRetriever(
            vector_store=vector_store,
            child_documents=child_documents,
            parent_map=parent_map,
        )
        return vector_store, retriever

    # --- Fresh Ingestion & Chunking ---
    if not page_documents:
        raise ValueError("No extractable pages found in manual.")

    logger.info("Creating hierarchical Parent-Child chunks...")
    hierarchy: ChunkHierarchy = create_hierarchical_chunks(page_documents)

    if not hierarchy.children:
        raise ValueError("No searchable chunks could be generated from the document.")

    logger.info("Indexing %d child chunks into ChromaDB...", len(hierarchy.children))
    vector_store = Chroma.from_documents(
        documents=hierarchy.children,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )

    # Persist cache
    _save_store_cache(persist_dir, hierarchy.parent_map, hierarchy.children)

    # Initialize Hybrid Retriever
    retriever = HybridParentRetriever(
        vector_store=vector_store,
        child_documents=hierarchy.children,
        parent_map=hierarchy.parent_map,
    )

    return vector_store, retriever
