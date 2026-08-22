# Medical Equipment Manual RAG System (MVP)

A lightweight, memory-efficient local Retrieval-Augmented Generation (RAG) web application for medical equipment field engineers. Ingest heavy PDF technical manuals (500–700 pages) and query them with natural language — all answers are strictly grounded with exact page citations.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10 or higher |
| **Ollama** | Download & install from [https://ollama.com](https://ollama.com) |

### Pull Required Ollama Models

After installing Ollama, open a terminal and run:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

---

## Quick Start

### 1. Clone / Navigate to the Project

```bash
cd medical_rag_mvp
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Place Manuals (optional)

Copy your PDF manuals into the `data/manuals/` directory, or upload them later through the web UI.

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Project Structure

```text
medical_rag_mvp/
├── data/
│   └── manuals/               # Input PDF files (500-700 pages)
├── chroma_db/                 # Persistent vector storage (auto-created)
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py          # Streaming page-by-page PDF ingestion
│   ├── vector_store.py        # Text chunking, embedding & vector DB
│   └── rag_chain.py           # QA retrieval chain with strict grounding
├── app.py                     # Streamlit web interface
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Usage

1. **Upload a Manual** — Use the sidebar to upload a PDF manual (or place it in `data/manuals/`).
2. **Select a Manual** — Pick the active manual from the sidebar dropdown.
3. **Ask a Question** — Type a query about error codes, maintenance procedures, or part numbers.
4. **Get Grounded Answers** — The system responds strictly from the manual content, displaying the exact source file and page number(s).

---

## Key Design Principles

- **Memory Efficiency**: PDFs are parsed page-by-page to prevent memory spikes.
- **Strict Grounding**: The LLM is explicitly instructed to answer only from the provided context.
- **Exact Citations**: Every answer includes the source filename and page number(s).
- **Offline Operation**: Runs entirely locally via Ollama — no cloud API keys needed.
