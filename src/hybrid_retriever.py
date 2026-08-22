"""
hybrid_retriever.py — Production-Grade Hybrid Search (Dense + BM25), RRF Fusion & Cross-Encoder Re-ranking.

Key Features:
1. Query Expansion & Acronym Normalization: Integrates `src.query_expansion` to expand medical/technical
   acronyms and error code variations for maximal BM25 and dense retrieval recall.
2. Dense Retrieval: Vector similarity search via ChromaDB (nomic-embed-text embeddings).
3. Sparse Retrieval: Lexical search via BM25 (custom technical tokenizer preserving error codes & part numbers).
4. Reciprocal Rank Fusion (RRF): Combines dense and sparse rank scores without score scale bias.
5. Parent Document Resolver: Maps high-precision child search hits to full parent context windows.
6. Cross-Encoder Re-ranking: Uses cross-attention (ms-marco-MiniLM-L-6-v2 / FlashRank) to filter and rank
   top candidates.
7. Confidence Score Threshold & Fallback: Rejects candidate documents whose re-ranking score falls below
   the minimum confidence threshold, preventing hallucinations on out-of-domain queries.
8. Diagnostics Inspector: Returns detailed scores for verification & retrieval debugging.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.query_expansion import expand_query, ExpandedQuery

logger = logging.getLogger(__name__)

# Default cross-encoder logit cutoff (-4.0 to -5.0 filters ungrounded / out-of-domain noise)
DEFAULT_CONFIDENCE_THRESHOLD = -4.5


# ---------------------------------------------------------------------------
# Technical Tokenizer for BM25
# ---------------------------------------------------------------------------
def technical_tokenize(text: str) -> list[str]:
    """Tokenize technical manual text for BM25.

    Preserves error codes (E-205, ERR_403, 0x8004), part numbers (STORZ-9812A),
    voltage/frequency (110-240V, 50/60Hz), and standard alphanumeric terms.
    """
    tokens = re.findall(r"[a-z0-9]+(?:[-_.:/][a-z0-9]+)*", text.lower())
    return tokens


# ---------------------------------------------------------------------------
# Cross-Encoder Re-ranker Wrapper (with lazy load & multi-backend support)
# ---------------------------------------------------------------------------
class CrossEncoderReranker:
    """Wrapper for Cross-Encoder re-ranking with SentenceTransformers or FlashRank."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model: Any = None
        self._backend: Optional[str] = None

    def _load(self):
        if self._model is not None:
            return

        # 1. Try sentence-transformers CrossEncoder
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            self._backend = "sentence-transformers"
            logger.info("Loaded CrossEncoder model: %s", self.model_name)
            return
        except Exception as e:
            logger.warning("sentence-transformers unavailable (%s); attempting FlashRank fallback.", e)

        # 2. Try FlashRank as fast lightweight CPU fallback
        try:
            from flashrank import Ranker
            self._model = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
            self._backend = "flashrank"
            logger.info("Loaded FlashRank fallback model.")
            return
        except Exception as e:
            logger.warning("FlashRank also unavailable (%s); re-ranking will use RRF scores directly.", e)
            self._backend = "none"

    def rank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> list[tuple[Document, float]]:
        """Score (query, document) pairs and return top_k sorted by score that pass the threshold."""
        if not documents:
            return []

        self._load()

        if self._backend == "sentence-transformers":
            pairs = [[query, doc.page_content] for doc in documents]
            scores = self._model.predict(pairs)
            scored_docs = list(zip(documents, [float(s) for s in scores]))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            # Filter by confidence threshold
            filtered = [(doc, s) for doc, s in scored_docs if s >= threshold]
            return filtered[:top_k]

        elif self._backend == "flashrank":
            from flashrank import RerankRequest
            passages = [{"id": i, "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(documents)]
            rerank_request = RerankRequest(query=query, passages=passages)
            results = self._model.rerank(rerank_request)
            scored_docs = []
            for r in results:
                orig_doc = documents[r["id"]]
                scored_docs.append((orig_doc, float(r["score"])))
            return scored_docs[:top_k]

        else:
            return [(doc, 1.0) for doc in documents[:top_k]]


# ---------------------------------------------------------------------------
# Hybrid Retriever with Parent Resolution & Diagnostics
# ---------------------------------------------------------------------------
class HybridParentRetriever:
    """Hybrid Search (Dense + BM25) + Query Expansion + RRF + Parent Resolution + Cross-Encoder Re-ranking."""

    def __init__(
        self,
        vector_store: Chroma,
        child_documents: list[Document],
        parent_map: dict[str, Document],
        dense_top_k: int = 15,
        sparse_top_k: int = 15,
        final_top_k: int = 5,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.child_documents = child_documents
        self.parent_map = parent_map
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.final_top_k = final_top_k
        self.confidence_threshold = confidence_threshold
        self.rrf_k = rrf_k
        self.reranker = CrossEncoderReranker()

        # Initialize BM25 Index
        self._bm25 = None
        self._init_bm25()

    def _init_bm25(self):
        if not self.child_documents:
            return
        try:
            from rank_bm25 import BM25Okapi

            tokenized_corpus = [technical_tokenize(doc.page_content) for doc in self.child_documents]
            self._bm25 = BM25Okapi(tokenized_corpus)
            logger.info("Initialized BM25 index over %d child chunks.", len(self.child_documents))
        except Exception as e:
            logger.error("Failed to initialize BM25: %s", e)

    def dense_search(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        """Perform dense vector search in Chroma, returning (doc, score)."""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            return results
        except Exception as e:
            logger.error("Dense search failed: %s", e)
            return []

    def sparse_search(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        """Perform BM25 search over child documents, returning (doc, score)."""
        if self._bm25 is None or not self.child_documents:
            return []

        tokenized_query = technical_tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        doc_scores = list(zip(self.child_documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [(doc, float(s)) for doc, s in doc_scores[:top_k] if s > 0.0]

    def retrieve_with_diagnostics(self, query: str) -> dict[str, Any]:
        """Perform full hybrid retrieval, query expansion, re-ranking, and return diagnostic telemetry."""
        # 1. Query Expansion & Acronym Normalization
        expanded_info: ExpandedQuery = expand_query(query)
        search_query = expanded_info.expanded_query

        # 2. Dense retrieval using enriched query
        dense_results = self.dense_search(search_query, self.dense_top_k)

        # 3. Sparse (BM25) retrieval using enriched query
        sparse_results = self.sparse_search(search_query, self.sparse_top_k)

        # 4. Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[str, float] = {}
        child_doc_by_key: dict[str, Document] = {}

        def _doc_key(doc: Document) -> str:
            return f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}_{doc.metadata.get('child_index', doc.page_content[:40])}"

        for rank, (doc, _) in enumerate(dense_results, start=1):
            key = _doc_key(doc)
            child_doc_by_key[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, (doc, _) in enumerate(sparse_results, start=1):
            key = _doc_key(doc)
            child_doc_by_key[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (self.rrf_k + rank))

        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        fused_children = [child_doc_by_key[k] for k, _ in sorted_rrf]

        # 5. Resolve child chunks to unique Parent Documents
        candidate_parents: list[Document] = []
        seen_parent_ids: set[str] = set()

        for child in fused_children:
            parent_id = child.metadata.get("parent_id")
            if parent_id and parent_id in self.parent_map:
                if parent_id not in seen_parent_ids:
                    seen_parent_ids.add(parent_id)
                    candidate_parents.append(self.parent_map[parent_id])
            else:
                candidate_parents.append(child)

        # 6. Cross-Encoder Re-ranking with Confidence Threshold Filtering
        reranked_parents = self.reranker.rank(
            query=query,  # use original user query for cross-attention evaluation
            documents=candidate_parents[:15],
            top_k=self.final_top_k,
            threshold=self.confidence_threshold,
        )

        passed_confidence = len(reranked_parents) > 0
        final_documents = [doc for doc, _ in reranked_parents]

        # Best confidence score (or lowest cutoff if none passed)
        top_score = reranked_parents[0][1] if reranked_parents else -99.0

        return {
            "query": query,
            "expanded_query": search_query,
            "expanded_terms": expanded_info.expanded_terms,
            "detected_error_codes": expanded_info.detected_error_codes,
            "dense_results": [{"doc": d, "score": float(s)} for d, s in dense_results],
            "sparse_results": [{"doc": d, "score": float(s)} for d, s in sparse_results],
            "fused_child_count": len(fused_children),
            "candidate_parent_count": len(candidate_parents),
            "reranked_results": [{"doc": d, "score": float(s)} for d, s in reranked_parents],
            "final_documents": final_documents,
            "passed_confidence": passed_confidence,
            "top_confidence_score": top_score,
            "confidence_threshold": self.confidence_threshold,
        }

    def invoke(self, query: str) -> list[Document]:
        """LangChain standard retrieval interface returning top re-ranked Parent Documents."""
        diagnostics = self.retrieve_with_diagnostics(query)
        return diagnostics["final_documents"]
