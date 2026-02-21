"""
Ask your PDF: RAG question-answering (retrieval-augmented generation).

Answer questions over one or more PDFs by transcribing them, chunking and embedding
the text, then retrieving relevant chunks for each question and generating answers with Gemini.

Supports:
  - Single file:  path/to.pdf
  - Folder:       path/to/folder  (all .pdf files in that directory; set RECURSIVE_PDF=1 for subfolders)

Flow:
  1. Transcribe PDF(s) to text (or use plain text).
  2. Split text into chunks and embed with Gemini (RETRIEVAL_DOCUMENT).
  3. On each query: embed query (RETRIEVAL_QUERY), find top-k chunks by similarity,
     then generate an answer with Gemini using only those chunks as context.

Run from project root:
  python examples/pdf_rag_qa.py path/to.pdf "Your question here"
  python examples/pdf_rag_qa.py path/to/folder "Your question here"

Or set PDF_PATH (file or folder) and optional QUERY_PROMPT in .env.
"""

import os
import sys
from pathlib import Path

# Add project root so "ideate" can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ideate import transcribe_pdf


def _is_quota_error(exc: BaseException) -> bool:
    """True if the exception is a Gemini API rate-limit / quota error."""
    err = str(exc)
    return "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower()


def _handle_api_error(exc: BaseException) -> None:
    """Print a short message for quota/rate-limit errors and exit."""
    if _is_quota_error(exc):
        print("Error: Gemini API rate limit or quota exceeded.", file=sys.stderr)
        print("Wait a minute and retry, or check https://ai.google.dev/gemini-api/docs/rate-limits", file=sys.stderr)
        sys.exit(1)
    raise exc


# --- Chunking ---
def chunk_text(
    text: str,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """Split text into overlapping chunks (by character count)."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # Prefer breaking at paragraph or sentence
        if end < len(text):
            for sep in ("\n\n", "\n", ". "):
                last = chunk.rfind(sep)
                if last > chunk_size // 2:
                    chunk = chunk[: last + len(sep)]
                    end = start + len(chunk)
                    break
        chunks.append(chunk.strip())
        start = end - overlap if overlap < chunk_size else end
    return [c for c in chunks if c]


# --- Embeddings (Gemini) ---
# Gemini API (Google AI Studio / api_key): gemini-embedding-001. Vertex: text-embedding-004.
# Set EMBED_MODEL in .env to override.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
GENERATE_MODEL = "gemini-2.5-flash"


def _get_client():
    from google import genai
    key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env")
    return genai.Client(api_key=key)


def embed_documents(client, texts: list[str]):
    """Embed document chunks. Uses RETRIEVAL_DOCUMENT if the model supports it."""
    from google.genai import types
    if not texts:
        return []
    config = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    try:
        out = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts,
            config=config,
        )
    except Exception:
        # Some models (e.g. gemini-embedding-001) may not support task_type
        out = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    if not out.embeddings:
        return []
    return [e.values for e in out.embeddings if e.values]


def embed_query(client, query: str) -> list[float]:
    """Embed a single query. Uses RETRIEVAL_QUERY if the model supports it."""
    from google.genai import types
    config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    try:
        out = client.models.embed_content(
            model=EMBED_MODEL,
            contents=query,
            config=config,
        )
    except Exception:
        out = client.models.embed_content(model=EMBED_MODEL, contents=query)
    if not out.embeddings or not out.embeddings[0].values:
        return []
    return out.embeddings[0].values


# --- Retrieval ---
def cosine_similarity(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve(
    query_embedding: list[float],
    chunk_embeddings: list[list[float]],
    chunk_texts: list[str],
    top_k: int = 5,
) -> list[str]:
    """Return top-k chunk texts by similarity to the query embedding."""
    if not query_embedding or not chunk_embeddings:
        return []
    scored = [
        (cosine_similarity(query_embedding, emb), text)
        for emb, text in zip(chunk_embeddings, chunk_texts)
    ]
    scored.sort(key=lambda x: -x[0])
    return [text for _, text in scored[:top_k]]


# --- Generate answer from retrieved context ---
def generate_answer(client, context_chunks: list[str], query: str) -> str:
    """Generate an answer using only the provided context chunks."""
    from google.genai import types
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "Use only the following context to answer the question. "
        "If the context does not contain enough information, say so. "
        "When context is labeled with 'From: <filename>', you may cite that source in your answer.\n\n"
        "Context:\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    response = client.models.generate_content(
        model=GENERATE_MODEL,
        contents=types.Part.from_text(text=prompt),
        config=types.GenerateContentConfig(temperature=0.2),
    )
    if not response.text:
        return ""
    return response.text.strip()


# --- RAG pipeline ---
def build_rag_from_text(
    text: str,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
):
    """Build in-memory RAG index from document text. Returns (chunk_texts, chunk_embeddings, client)."""
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return [], [], _get_client()
    client = _get_client()
    embeddings = embed_documents(client, chunks)
    return chunks, embeddings, client


def build_rag_from_pdf(pdf_path: str | Path, **kwargs):
    """Transcribe PDF and build RAG index. Returns (chunk_texts, chunk_embeddings, client)."""
    text = transcribe_pdf(pdf_path)
    return build_rag_from_text(text, **kwargs)


def list_pdfs_in_folder(
    folder_path: str | Path,
    *,
    recursive: bool = False,
) -> list[Path]:
    """Return paths to all .pdf files in a folder. Optionally include subfolders."""
    folder = Path(folder_path)
    if not folder.is_dir():
        return []
    if recursive:
        return sorted(folder.rglob("*.pdf"))
    return sorted(folder.glob("*.pdf"))


def build_rag_from_folder(
    folder_path: str | Path,
    *,
    recursive: bool = False,
    chunk_size: int = 512,
    overlap: int = 64,
    tag_sources: bool = True,
):
    """
    Transcribe all PDFs in a folder and build one RAG index over them.

    Returns (chunk_texts, chunk_embeddings, client). When tag_sources is True,
    each chunk is prefixed with "From: <filename>\\n\\n" so answers can cite which document.
    """
    folder = Path(folder_path)
    pdf_paths = list_pdfs_in_folder(folder, recursive=recursive)
    if not pdf_paths:
        return [], [], _get_client()

    all_chunks: list[str] = []
    for pdf_path in pdf_paths:
        text = transcribe_pdf(pdf_path)
        if not text.strip():
            continue
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        name = pdf_path.name
        for c in chunks:
            if tag_sources:
                all_chunks.append(f"From: {name}\n\n{c}")
            else:
                all_chunks.append(c)

    if not all_chunks:
        return [], [], _get_client()
    client = _get_client()
    embeddings = embed_documents(client, all_chunks)
    return all_chunks, embeddings, client


def ask_pdf(
    query: str,
    chunk_texts: list[str],
    chunk_embeddings: list[list[float]],
    client,
    top_k: int = 5,
) -> str:
    """Run a single RAG query: retrieve relevant chunks, then generate answer."""
    query_emb = embed_query(client, query)
    if not query_emb:
        return ""
    relevant = retrieve(query_emb, chunk_embeddings, chunk_texts, top_k=top_k)
    if not relevant:
        return ""
    return generate_answer(client, relevant, query)


def main():
    path = (
        sys.argv[1]
        if len(sys.argv) >= 2
        else os.environ.get("PDF_PATH")
    )
    query = " ".join(sys.argv[2:]).strip() if len(sys.argv) >= 3 else os.environ.get("QUERY_PROMPT", "").strip()

    if not path:
        print("Usage: python examples/pdf_rag_qa.py <path_to.pdf | path_to_folder> [query]")
        print("  Single PDF:")
        print("    python examples/pdf_rag_qa.py doc.pdf 'What are the main conclusions?'")
        print("  All PDFs in a folder (set RECURSIVE_PDF=1 in .env to include subfolders):")
        print("    python examples/pdf_rag_qa.py ./papers 'Summarize the main ideas across these documents.'")
        sys.exit(1)

    path = path.strip()
    if " " in path and not query:
        pdf_part = next((p for p in path.split() if p.endswith(".pdf")), None)
        if pdf_part:
            path = pdf_part

    path = Path(path)
    recursive = os.environ.get("RECURSIVE_PDF", "").strip().lower() in ("1", "true", "yes")

    try:
        if path.is_dir():
            print(f"Indexing PDFs in folder: {path}", file=sys.stderr)
            pdf_paths = list_pdfs_in_folder(path, recursive=recursive)
            if not pdf_paths:
                print("No .pdf files found in that folder.", file=sys.stderr)
                sys.exit(1)
            print(f"Found {len(pdf_paths)} PDF(s): {[p.name for p in pdf_paths]}", file=sys.stderr)
            chunk_texts, chunk_embeddings, client = build_rag_from_folder(
                path, recursive=recursive, tag_sources=True
            )
        else:
            print("Transcribing PDF...", file=sys.stderr)
            chunk_texts, chunk_embeddings, client = build_rag_from_pdf(path)
    except Exception as e:
        _handle_api_error(e)

    print(f"Chunks: {len(chunk_texts)}", file=sys.stderr)

    if not chunk_texts:
        print("No content to index.", file=sys.stderr)
        sys.exit(1)

    if query:
        print(f"Query: {query}", file=sys.stderr)
        try:
            answer = ask_pdf(query, chunk_texts, chunk_embeddings, client)
            print(answer)
        except Exception as e:
            _handle_api_error(e)
    else:
        print("No query provided. Set QUERY_PROMPT in .env or pass as second argument.", file=sys.stderr)


if __name__ == "__main__":
    main()
