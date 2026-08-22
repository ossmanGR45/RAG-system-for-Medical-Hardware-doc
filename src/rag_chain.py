"""
rag_chain.py — RAG Generation Chain with Strict Grounding & Confidence Fallback.

Key Features:
1. Strict System Prompt: Zero hallucinations, explicit refusal if context lacks info, exact source & page citations.
2. Confidence Gate & Fast Fallback: If no retrieved documents pass the confidence score threshold,
   returns an immediate fallback message without wasting LLM compute.
3. Context Formatter: Renders hierarchical breadcrumbs, section titles, and intact Markdown tables.
"""

from __future__ import annotations

from typing import Any
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

FALLBACK_MESSAGE = (
    "Information not available in the provided manual.\n\n"
    "*(The search query did not match any documented error codes, maintenance steps, "
    "or specifications in the active technical manual with sufficient confidence.)*"
)

# ---------------------------------------------------------------------------
# Strict Grounding Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a precise technical assistant for medical equipment field engineers.

RULES — follow every rule without exception:
1. Answer the user's question ONLY using the provided manual context below.
2. If the context does not contain enough information to answer, respond
   EXACTLY with: "Information not available in the provided manual."
3. NEVER fabricate, guess, or infer information that is not explicitly
   present in the context.
4. If the question asks about a table, error code, specification, or procedure,
   reproduce the exact values, pinouts, steps, or parameters as given in the context.
5. NEVER provide unverified safety procedures or improvised troubleshooting steps.
6. At the end of every answer, cite the exact source(s) in this format:
   **Source:** <filename>, Page <page_number> (Section: <breadcrumb/section>)
7. Keep answers clear, technical, and directly focused on the engineer's query.

CONTEXT:
{context}
"""

USER_PROMPT = "{question}"


def format_context_docs(docs: list[Document]) -> str:
    """Format retrieved Parent Documents into a structured context string."""
    if not docs:
        return ""

    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Manual")
        page = doc.metadata.get("page", "?")
        breadcrumb = doc.metadata.get("breadcrumb", f"[Doc: {source} | Page: {page}]")

        header = f"=== SECTION {i}: {breadcrumb} (Page {page}) ==="
        content = doc.page_content.strip()
        parts.append(f"{header}\n{content}")

    return "\n\n" + "\n\n".join(parts) + "\n\n"


def build_rag_chain(retriever: Any, model_name: str = "llama3.2:3b"):
    """Construct and return a LangChain runnable RAG chain using the provided retriever."""
    llm = ChatOllama(
        model=model_name,
        temperature=0.0,
        base_url="http://localhost:11434",
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )

    # Core generation chain
    llm_chain = prompt | llm | StrOutputParser()

    def _execute_rag(inputs: dict | str) -> str:
        q = inputs["question"] if isinstance(inputs, dict) else str(inputs)
        
        # 1. Retrieve candidates
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(q)
        else:
            docs = retriever.get_relevant_documents(q)

        # 2. Confidence Gate: If no candidate passed threshold, return fast fallback
        if not docs:
            return FALLBACK_MESSAGE

        # 3. Format context and invoke LLM
        formatted_ctx = format_context_docs(docs)
        return llm_chain.invoke({"context": formatted_ctx, "question": q})

    rag_chain = RunnableLambda(_execute_rag)
    return rag_chain, retriever
