"""
test_enhancements.py — Unit Tests for Phase 2 Enhancements.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from langchain_core.documents import Document

from src.query_expansion import expand_query, extract_and_expand_error_codes
from src.pdf_loader import render_pdf_page_image
from src.hybrid_retriever import HybridParentRetriever
from src.rag_chain import build_rag_chain, FALLBACK_MESSAGE


class TestPhase2Enhancements(unittest.TestCase):

    def test_acronym_expansion(self):
        # Query with PSU and OVP
        res = expand_query("What causes PSU OVP failure?")
        self.assertIn("power supply unit", res.expanded_query.lower())
        self.assertIn("overvoltage protection", res.expanded_query.lower())
        self.assertIn("power supply unit", res.expanded_terms)

    def test_error_code_expansion(self):
        # Query mentioning E205
        res = expand_query("Troubleshoot E205")
        self.assertIn("E-205", res.detected_error_codes)
        self.assertIn("ERR-205", res.detected_error_codes)
        self.assertIn("Error 205", res.detected_error_codes)
        self.assertIn("E-205", res.expanded_query)

    def test_hex_error_code_expansion(self):
        codes = extract_and_expand_error_codes("Device halted with 0x8004")
        self.assertIn("0x8004", codes)

    def test_confidence_threshold_filtering(self):
        doc1 = Document(page_content="Valid technical manual troubleshooting section.", metadata={"page": 1})
        
        class MockVectorStore:
            def similarity_search_with_score(self, query, k=15):
                return [(doc1, 0.5)]

        # Set a very high threshold that will reject everything
        retriever_strict = HybridParentRetriever(
            vector_store=MockVectorStore(),
            child_documents=[doc1],
            parent_map={"p1": doc1},
            confidence_threshold=10.0,  # Unattainable high cutoff
        )
        diag = retriever_strict.retrieve_with_diagnostics("Random irrelevant query")
        self.assertFalse(diag["passed_confidence"])
        self.assertEqual(len(diag["final_documents"]), 0)

    def test_rag_chain_fallback(self):
        # Mock empty retriever (rejected by threshold)
        class EmptyRetriever:
            def invoke(self, query):
                return []

        chain, _ = build_rag_chain(EmptyRetriever())
        ans = chain.invoke("Unrelated question")
        self.assertIn("Information not available in the provided manual", ans)


if __name__ == "__main__":
    unittest.main()
