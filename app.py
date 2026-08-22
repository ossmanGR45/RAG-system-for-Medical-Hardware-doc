"""
app.py — Modern Streamlit UI for Medical Equipment Manual RAG System.

Features:
- Structured Layout-Aware Ingestion (`pymupdf4llm`) preserving Markdown tables & headers.
- Context-Preserving Header-Aware & Parent-Child Chunking with hierarchical breadcrumbs.
- Domain Adaptation: Medical/Technical Acronym Expansion & Error Code Normalization.
- Visual PDF Page & Diagram Preview: On-demand high-res rendering of cited manual pages.
- Hybrid Retrieval (Dense Vector Search + BM25 Lexical Search) with RRF Fusion.
- Cross-Encoder Re-ranking with Confidence Threshold Gate & Fallback.
- Real-time Retrieval Diagnostics & Score Inspection Expander.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from src.pdf_loader import load_pdf_pages, render_pdf_page_image
from src.vector_store import build_hierarchical_hybrid_pipeline
from src.rag_chain import build_rag_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MANUALS_DIR = BASE_DIR / "data" / "manuals"
CHROMA_DIR = BASE_DIR / "chroma_db"

MANUALS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Page Config & Styles
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Manual RAG — Hybrid & Re-ranked",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0b1d28 0%, #17384a 50%, #204e67 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: #ffffff;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.22);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
        font-weight: 300;
    }

    .badge-pill {
        display: inline-block;
        background: #1e3a4c;
        color: #64d2ff;
        padding: 0.25rem 0.65rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        border: 1px solid rgba(100, 210, 255, 0.2);
    }

    .answer-card {
        background: #f8fafc;
        border-left: 6px solid #204e67;
        border-radius: 10px;
        padding: 1.6rem 2.2rem;
        margin-top: 1rem;
        font-size: 1.02rem;
        line-height: 1.75;
        color: #0f172a;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
    }

    .citation-badge {
        display: inline-block;
        background: linear-gradient(135deg, #204e67, #17384a);
        color: #ffffff;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.3rem 0.4rem 0.3rem 0;
        letter-spacing: 0.2px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1d28 0%, #17384a 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — Manual Manager & Pipeline Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📂 Manual Manager")

    uploaded_file = st.file_uploader(
        "Upload Technical PDF Manual",
        type=["pdf"],
        help="Upload 500-700 page medical manual. Structured tables, headers & diagrams will be indexed.",
    )

    if uploaded_file is not None:
        dest = MANUALS_DIR / uploaded_file.name
        if not dest.exists():
            with open(dest, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Uploaded: {uploaded_file.name}")
        else:
            st.info(f"ℹ️ Loaded existing: {uploaded_file.name}")

    pdf_files = sorted([f.name for f in MANUALS_DIR.glob("*.pdf")])

    if pdf_files:
        selected_manual = st.selectbox(
            "Active Manual",
            pdf_files,
            index=0,
            help="Choose which manual to query.",
        )
    else:
        selected_manual = None
        st.warning("No PDF manuals found in `data/manuals/`. Upload one above to begin.")

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Settings")
    show_diagnostics = st.toggle(
        "Show Diagnostics & Telemetry",
        value=True,
        help="Inspect Dense, BM25, Acronym Expansions, and Cross-Encoder scores in real-time.",
    )
    show_page_previews = st.toggle(
        "Enable Visual Page Previews",
        value=True,
        help="Render high-resolution PDF page diagrams and schematics next to citations.",
    )
    rerank_top_k = st.slider(
        "Context Parent Windows (Top-K)",
        min_value=1,
        max_value=8,
        value=4,
        help="Number of top-scoring parent sections/tables passed to the LLM.",
    )
    confidence_cutoff = st.slider(
        "Confidence Score Cutoff",
        min_value=-8.0,
        max_value=2.0,
        value=-4.5,
        step=0.5,
        help="Minimum Cross-Encoder logit score required to accept retrieved context. Out-of-domain queries below this cutoff trigger graceful fallback.",
    )

    st.markdown("---")
    st.markdown(
        "<small style='opacity:0.75'>Engine: Ollama (nomic-embed-text + llama3.2:3b)<br>Pipeline: Query Expansion · Hybrid Dense+BM25 · Cross-Encoder</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pipeline Caching
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="⚙️ Parsing & Indexing manual with Layout-Aware Hybrid Pipeline...")
def get_hybrid_pipeline(manual_name: str):
    """Load or construct the hybrid pipeline (Chroma + BM25 + Parent Map)."""
    manual_chroma_dir = str(CHROMA_DIR / manual_name.replace(".pdf", ""))
    manual_path = MANUALS_DIR / manual_name

    pages = list(load_pdf_pages(manual_path))
    if not pages:
        raise ValueError(f"Could not extract pages from '{manual_name}'.")

    vector_store, hybrid_retriever = build_hierarchical_hybrid_pipeline(
        page_documents=pages,
        persist_dir=manual_chroma_dir,
    )
    return vector_store, hybrid_retriever


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🏥 Medical Manual RAG Assistant</h1>
        <p>
            <span class="badge-pill">Layout-Aware Ingestion</span>
            <span class="badge-pill">Acronym & Code Expansion</span>
            <span class="badge-pill">Dense + BM25 Hybrid</span>
            <span class="badge-pill">Cross-Encoder Re-ranker</span>
            <span class="badge-pill">Visual Page Previews</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if selected_manual is None:
    st.info("👋 Upload a PDF manual in the sidebar or copy files into `data/manuals/` to start querying.")
    st.stop()

# Initialize Pipeline
try:
    vector_store, hybrid_retriever = get_hybrid_pipeline(selected_manual)
    hybrid_retriever.final_top_k = rerank_top_k
    hybrid_retriever.confidence_threshold = confidence_cutoff
    rag_chain, _ = build_rag_chain(hybrid_retriever)
except Exception as e:
    st.error(f"⚠️ Error initializing pipeline for **{selected_manual}**: {e}")
    st.stop()

st.success(f"✅ Ready — active manual: **{selected_manual}** (Structured Hybrid Index Loaded)")

# Query Input
query = st.text_input(
    "🔎 Search manual (error codes, pinouts, troubleshooting tables, replacement steps, acronyms):",
    placeholder="e.g. What causes E205 on the PSU? What is the fuse rating for model X?",
)

col_btn, _ = st.columns([1, 5])
with col_btn:
    search_clicked = st.button("Search Manual", type="primary", use_container_width=True)

if search_clicked and query.strip():
    manual_path = MANUALS_DIR / selected_manual

    with st.spinner("🔍 Running Query Expansion, Dense + BM25 Search & Re-ranking..."):
        diagnostics = hybrid_retriever.retrieve_with_diagnostics(query)
        final_parents = diagnostics["final_documents"]
        passed_conf = diagnostics["passed_confidence"]
        top_score = diagnostics["top_confidence_score"]

    with st.spinner("🤖 Generating strictly grounded response..."):
        answer = rag_chain.invoke(query)

    # Confidence Banner
    if passed_conf:
        st.markdown(f"🎯 **Retrieval Confidence:** `High` (Top Score: `{top_score:+.2f}` ≥ `{confidence_cutoff:.1f}` cutoff)")
    else:
        st.warning(f"⚠️ **Retrieval Confidence:** `Low / Below Cutoff` (Top Score: `{top_score:+.2f}` < `{confidence_cutoff:.1f}`). Fast fallback engaged.")

    # 1. Answer Card
    st.markdown("### 💡 Grounded Answer")
    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

    # 2. Citations & Visual Page Previews
    if final_parents:
        st.markdown("### 📑 Verified Sources & Breadcrumbs")
        seen_pages = []
        for doc in final_parents:
            src = doc.metadata.get("source", selected_manual)
            page = doc.metadata.get("page", "?")
            breadcrumb = doc.metadata.get("breadcrumb", f"Page {page}")
            key = (src, page, breadcrumb)
            if key not in [k[:3] for k in seen_pages]:
                seen_pages.append((src, page, breadcrumb, doc))
                st.markdown(
                    f'<span class="citation-badge">📄 {src} — Page {page} | 🏷️ {breadcrumb}</span>',
                    unsafe_allow_html=True,
                )

        # Visual PDF Page Preview section
        if show_page_previews and manual_path.exists():
            st.markdown("### 🖼️ Visual Manual Page & Diagram Previews")
            with st.expander("🔍 Click to view original PDF pages, schematics & diagrams", expanded=False):
                cols = st.columns(min(len(seen_pages), 3) if seen_pages else 1)
                for idx, (src, page_num, bcrumb, p_doc) in enumerate(seen_pages):
                    if isinstance(page_num, int):
                        col = cols[idx % len(cols)]
                        with col:
                            st.markdown(f"**Page {page_num}** ({src})")
                            st.caption(bcrumb)
                            img_bytes = render_pdf_page_image(manual_path, page_num, dpi=140)
                            if img_bytes:
                                st.image(img_bytes, caption=f"Page {page_num} — {bcrumb}", use_container_width=True)
                            else:
                                st.info(f"Could not render image for page {page_num}.")

    # 3. Retrieval Diagnostics & Inspection Panel
    if show_diagnostics:
        with st.expander("🔬 Retrieval & Re-ranking Diagnostics Inspector", expanded=False):
            t_expand, t1, t2, t3, t4 = st.tabs([
                "🔤 Query Expansion",
                "🎯 Final Reranked Context",
                "⚡ BM25 Lexical Hits",
                "🌐 Dense Vector Hits",
                "📊 Fused Telemetry",
            ])

            with t_expand:
                st.markdown("#### Domain Adaptation & Query Enrichment")
                st.write(f"**Raw User Query:** `{query}`")
                st.write(f"**Enriched Search Query:** `{diagnostics['expanded_query']}`")
                if diagnostics["expanded_terms"]:
                    st.markdown("**Added Medical / Technical Synonyms:**")
                    for term in diagnostics["expanded_terms"]:
                        st.markdown(f"- `{term}`")
                if diagnostics["detected_error_codes"]:
                    st.markdown("**Normalized Error Code Variations:**")
                    st.write(diagnostics["detected_error_codes"])

            with t1:
                st.markdown("#### Top Re-ranked Parent Context Windows sent to LLM")
                if diagnostics["reranked_results"]:
                    for i, r in enumerate(diagnostics["reranked_results"], 1):
                        doc = r["doc"]
                        score = r["score"]
                        meta = doc.metadata
                        badge = "📊 [TABLE]" if meta.get("is_table") else "📝 [SECTION]"
                        st.markdown(f"**#{i} {badge} Re-ranking Score: `{score:+.4f}`** | Page {meta.get('page')}")
                        st.caption(f"Breadcrumb: `{meta.get('breadcrumb')}`")
                        st.markdown(doc.page_content)
                        st.divider()
                else:
                    st.info("No documents passed the confidence score threshold.")

            with t2:
                st.markdown("#### BM25 Lexical Keyword Matches (Exact Tokens)")
                if diagnostics["sparse_results"]:
                    for i, res in enumerate(diagnostics["sparse_results"][:5], 1):
                        st.markdown(f"**Hit #{i} [BM25 Score: `{res['score']:.4f}`]** — Page {res['doc'].metadata.get('page')}")
                        st.code(res['doc'].page_content[:400], language="markdown")
                else:
                    st.info("No BM25 keyword matches for this query.")

            with t3:
                st.markdown("#### Dense Vector Matches (Chroma Cosine Distance)")
                if diagnostics["dense_results"]:
                    for i, res in enumerate(diagnostics["dense_results"][:5], 1):
                        st.markdown(f"**Hit #{i} [Distance: `{res['score']:.4f}`]** — Page {res['doc'].metadata.get('page')}")
                        st.code(res['doc'].page_content[:400], language="markdown")
                else:
                    st.info("No dense matches.")

            with t4:
                st.markdown("#### Fusion Metrics & Gate")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Dense Candidates", len(diagnostics["dense_results"]))
                c2.metric("BM25 Keyword Hits", len(diagnostics["sparse_results"]))
                c3.metric("Fused Candidates", diagnostics["fused_child_count"])
                c4.metric("Passed Cutoff", "YES" if passed_conf else "NO")
                st.write(f"**Resolved to `{diagnostics['candidate_parent_count']}` unique Parent Context Windows** before Cross-Encoder scoring.")

elif search_clicked:
    st.warning("Please enter a question before searching.")
