"""
debug_retrieval.py — Diagnostic & Verification Tool for RAG Ingestion & Hybrid Retrieval.

Usage:
    python debug_retrieval.py --manual "my_manual.pdf" --query "What is error code E-205?"
    python debug_retrieval.py --query "test query"  (uses first available manual in data/manuals/)

Features:
1. Inspects Dense (Chroma) search results & distance metrics.
2. Inspects Sparse (BM25) keyword hits & relevance scores.
3. Shows Reciprocal Rank Fusion (RRF) candidate rankings.
4. Shows resolved Parent Document windows.
5. Displays Cross-Encoder Re-ranking scores & confidence.
6. Displays the final assembled context and generates the LLM response.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAG-Debugger")

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.pdf_loader import load_pdf_pages
from src.vector_store import build_hierarchical_hybrid_pipeline
from src.rag_chain import build_rag_chain, format_context_docs


def run_diagnostics(manual_name: str | None, query: str, run_llm: bool = True):
    manuals_dir = BASE_DIR / "data" / "manuals"
    manuals = list(manuals_dir.glob("*.pdf"))

    if not manuals:
        print(f"\n[ERROR] No PDF manuals found in {manuals_dir}. Please place a manual there.")
        return

    if manual_name:
        manual_path = manuals_dir / manual_name
        if not manual_path.exists():
            print(f"\n[ERROR] Specified manual '{manual_name}' not found in {manuals_dir}.")
            return
    else:
        manual_path = manuals[0]
        manual_name = manual_path.name

    print("\n" + "=" * 80)
    print(f"🔬 RAG RETRIEVAL & INGESTION DIAGNOSTIC REPORT")
    print(f"📄 Manual: {manual_name}")
    print(f"🔎 Query:  '{query}'")
    print("=" * 80)

    # 1. Ingest / Load Pipeline
    chroma_dir = str(BASE_DIR / "chroma_db" / manual_name.replace(".pdf", ""))
    print(f"\n[1/5] Ingesting / Loading Index from: {chroma_dir}...")
    pages = list(load_pdf_pages(manual_path))
    print(f"      Extracted {len(pages)} pages using layout-aware structured parser.")

    vector_store, hybrid_retriever = build_hierarchical_hybrid_pipeline(
        page_documents=pages,
        persist_dir=chroma_dir,
    )

    # 2. Run Retrieval Diagnostics
    print("\n[2/5] Running Hybrid Retrieval & Diagnostics...")
    diag = hybrid_retriever.retrieve_with_diagnostics(query)

    print("\n--- A. DENSE RETRIEVAL (Chroma Vector Search) ---")
    if diag["dense_results"]:
        for i, res in enumerate(diag["dense_results"][:5], 1):
            doc = res["doc"]
            score = res["score"]
            meta = doc.metadata
            print(f"  #{i} [Dist: {score:.4f}] Page {meta.get('page')}: {meta.get('breadcrumb')}")
            snippet = doc.page_content.replace("\n", " ")[:120]
            print(f"     Snippet: {snippet}...\n")
    else:
        print("  (No dense results returned)")

    print("--- B. SPARSE RETRIEVAL (BM25 Lexical Keyword Search) ---")
    if diag["sparse_results"]:
        for i, res in enumerate(diag["sparse_results"][:5], 1):
            doc = res["doc"]
            score = res["score"]
            meta = doc.metadata
            print(f"  #{i} [BM25 Score: {score:.4f}] Page {meta.get('page')}: {meta.get('breadcrumb')}")
            snippet = doc.page_content.replace("\n", " ")[:120]
            print(f"     Snippet: {snippet}...\n")
    else:
        print("  (No BM25 keyword matches found for technical tokens)")

    print(f"--- C. RECIPROCAL RANK FUSION & PARENT RESOLUTION ---")
    print(f"  Total Fused Child Candidates: {diag['fused_child_count']}")
    print(f"  Unique Candidate Parent Windows: {diag['candidate_parent_count']}")

    print("\n--- D. CROSS-ENCODER RE-RANKING (Top Parent Contexts) ---")
    if diag["reranked_results"]:
        for i, res in enumerate(diag["reranked_results"], 1):
            doc = res["doc"]
            score = res["score"]
            meta = doc.metadata
            is_tbl = "📊 [TABLE]" if meta.get("is_table") else "📝 [SECTION]"
            print(f"  Rank #{i} [Rerank Score: {score:+.4f}] {is_tbl} Page {meta.get('page')}: {meta.get('breadcrumb')}")
            print(f"  Content Preview:\n{doc.page_content[:250]}...\n")
    else:
        print("  (No reranked results)")

    # 3. Formatted Context Window
    print("[3/5] Formatted Context Window Sent to LLM:")
    formatted_ctx = format_context_docs(diag["final_documents"])
    print(formatted_ctx)

    # 4. Generation Test
    if run_llm:
        print("[4/5] Invoking LLM (llama3.2:3b, temp=0.0) with Strict Grounding...")
        rag_chain, _ = build_rag_chain(hybrid_retriever)
        try:
            answer = rag_chain.invoke(query)
            print("\n" + "=" * 80)
            print("💡 MODEL RESPONSE:")
            print("=" * 80)
            print(answer)
            print("=" * 80)
        except Exception as e:
            print(f"[ERROR] LLM Generation failed: {e}")

    print("\n[5/5] Diagnostic completed successfully.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug RAG Ingestion & Retrieval")
    parser.add_argument("--manual", type=str, default=None, help="Name of PDF manual in data/manuals/")
    parser.add_argument("--query", type=str, default="What is the maintenance procedure or error codes?", help="Test query")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM generation step")

    args = parser.parse_args()
    run_diagnostics(args.manual, args.query, run_llm=not args.no_llm)
