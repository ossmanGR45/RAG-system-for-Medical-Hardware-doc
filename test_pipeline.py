"""
test_pipeline.py — Unit & Integration Test Suite for the Refactored RAG Pipeline.
"""

from __future__ import annotations

import unittest
from langchain_core.documents import Document

from src.chunking import create_hierarchical_chunks, build_breadcrumb, _is_table_block
from src.hybrid_retriever import technical_tokenize, HybridParentRetriever


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        # Sample markdown manual page with headers, tables, and error codes
        self.sample_markdown = """\
# Chapter 3: Troubleshooting & Diagnostics

## Section 3.1: System Error Codes

When the device detects an abnormal state, it displays an error code on the LCD.

| Error Code | Fault Description | Corrective Action | Part Number |
| :--- | :--- | :--- | :--- |
| **E-101** | Overheating Detected | Verify fan operation & clear ventilation filter | FAN-4020 |
| **E-205** | High-Voltage Lamp Ignition Failure | Replace Xenon lamp module & check ballast cable | LAMP-9812A |
| **E-309** | Video Output Sync Loss | Reseat SDI cable or replace DSP processor board | DSP-1100 |

## Section 3.2: Maintenance & Fuse Ratings

Ensure the main power is disconnected prior to servicing.

### Power Supply Specifications
The power supply accepts 110-240V AC at 50/60Hz.
Fuse specification: 250V 5A Slow-Blow (Part #FUSE-2505).
"""
        self.page_doc = Document(
            page_content=self.sample_markdown,
            metadata={"source": "STORZ_Manual_v2.pdf", "page": 42, "total_pages": 120},
        )

    def test_table_detection(self):
        table_snippet = "| Col1 | Col2 |\n| --- | --- |\n| Val1 | Val2 |"
        self.assertTrue(_is_table_block(table_snippet))
        self.assertFalse(_is_table_block("Just a normal text paragraph without pipes."))

    def test_technical_tokenizer(self):
        query = "What causes error E-205 with part #LAMP-9812A at 110-240V?"
        tokens = technical_tokenize(query)
        self.assertIn("e-205", tokens)
        self.assertIn("lamp-9812a", tokens)
        self.assertIn("110-240v", tokens)

    def test_hierarchical_chunking_and_breadcrumbs(self):
        hierarchy = create_hierarchical_chunks([self.page_doc])

        self.assertGreater(len(hierarchy.parents), 0)
        self.assertGreater(len(hierarchy.children), 0)

        # Verify breadcrumbs
        for child in hierarchy.children:
            self.assertIn("Doc: STORZ_Manual_v2.pdf", child.page_content)
            self.assertIn("Page: 42", child.page_content)
            self.assertIn("parent_id", child.metadata)

        # Verify parent mapping
        for child in hierarchy.children:
            parent_id = child.metadata["parent_id"]
            self.assertIn(parent_id, hierarchy.parent_map)
            parent = hierarchy.parent_map[parent_id]
            self.assertIsInstance(parent, Document)

    def test_bm25_exact_keyword_retrieval(self):
        hierarchy = create_hierarchical_chunks([self.page_doc])

        # Mock vector store for HybridParentRetriever test
        class MockVectorStore:
            def similarity_search_with_score(self, query, k=15):
                # Return child documents with mock distance scores
                return [(hierarchy.children[0], 0.45)]

        retriever = HybridParentRetriever(
            vector_store=MockVectorStore(),
            child_documents=hierarchy.children,
            parent_map=hierarchy.parent_map,
            final_top_k=2,
        )

        diag = retriever.retrieve_with_diagnostics("E-205 Xenon lamp ignition")

        # Verify BM25 found matching candidate
        self.assertGreater(len(diag["sparse_results"]), 0)
        top_sparse_doc = diag["sparse_results"][0]["doc"]
        self.assertIn("E-205", top_sparse_doc.page_content)

        # Verify final re-ranked documents resolved to parent containing the table
        self.assertGreater(len(diag["final_documents"]), 0)
        final_doc = diag["final_documents"][0]
        self.assertIn("High-Voltage Lamp Ignition Failure", final_doc.page_content)
        self.assertIn("LAMP-9812A", final_doc.page_content)


if __name__ == "__main__":
    unittest.main()
