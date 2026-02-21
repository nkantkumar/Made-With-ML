# Made-With-ML

**`store/`** – Project-level folder for PDFs and other assets (e.g. use `python examples/pdf_rag_qa.py store "question"` to query all PDFs in it).

## Setup & run (PDF transcriber)

Your shell’s `python` may point to a different interpreter than the one where you installed dependencies. Use one of these:

**Option A – Project virtual environment (recommended)**

```bash
python3 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ideate
```

**Option B – Same Python as install**

If you use pyenv and installed with pyenv’s Python 3.10:

```bash
pyenv local 3.10.0   # or use this project’s Python
pip install -r requirements.txt
python -m ideate
```

Set `GEMINI_API_KEY` and optionally `PDF_PATH` in `.env` (copy from `.env.example`).

## Ask your PDF: RAG question-answering

The **`examples/pdf_rag_qa.py`** script lets you ask questions over one or more PDFs using retrieval-augmented generation (RAG):

1. **Transcribe** PDF(s) with Gemini.
2. **Chunk** the text and **embed** chunks with Gemini (e.g. `gemini-embedding-001`, RETRIEVAL_DOCUMENT).
3. On each **query**: embed the query (RETRIEVAL_QUERY), retrieve top-k chunks by cosine similarity, then **generate** an answer with Gemini using only those chunks.

**Single file or folder:**

```bash
# Single PDF
python examples/pdf_rag_qa.py path/to.pdf "What are the main conclusions?"

# All PDFs in a folder (answers can cite which document)
python examples/pdf_rag_qa.py path/to/folder "Summarize the main ideas across these documents."
# Set RECURSIVE_PDF=1 in .env to include subfolders
```

Reusable functions: `build_rag_from_pdf()`, `build_rag_from_folder()`, `ask_pdf()`, and `build_rag_from_text()`.

## Local Llama 3 reader (Ollama)

Use **Llama 3** locally for PDF text extraction and Q&A (no Gemini API):

1. Install [Ollama](https://ollama.com) and run a model: `ollama run llama3.2`
2. `pip install -r requirements.txt` (adds `ollama`, `pypdf`)
3. Use the same interface as the Gemini reader:

```python
from ideate import LlamaPDFReader, LlamaReaderConfig, query_pdf_llama

# Class-based (same API as GeminiPDFTranscriber)
config = LlamaReaderConfig(model="llama3.2", base_url="http://localhost:11434")
reader = LlamaPDFReader(config=config)
text = reader.transcribe("doc.pdf")           # pypdf extraction
answer = reader.query_pdf("doc.pdf", "What are the key points?")

# One-liner
answer = query_pdf_llama("doc.pdf", "Summarize this.")
```

**Transcription** is done with **pypdf** (local text extraction). **Q&A** uses **Ollama** (Llama 3). Set `OLLAMA_HOST` in `.env` if Ollama is not on `localhost:11434`.

## Postgres MCP (transactions + local Llama)

The **`postgres_mcp/`** folder provides PostgreSQL transactions and **local Llama** (Ollama) to turn natural language into SQL and run it.

- **Transactions:** `transaction()`, `execute()`, `execute_many()` with commit/rollback.
- **NL → SQL:** `ask_and_run("How many users?")` uses Llama to generate SQL, then runs it in a transaction.

See **`postgres_mcp/README.md`** for setup (DATABASE_URL or POSTGRES_* env) and examples.
