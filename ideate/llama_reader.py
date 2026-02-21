"""
Local PDF reader using Llama 3 via Ollama.

Same high-level interface as the Gemini reader: transcribe PDF to text,
then query that text with a local Llama model. PDF text is extracted with
pypdf (no cloud API); generation and Q&A use Ollama (e.g. llama3.2).

Requires Ollama running locally with a model pulled, e.g.:
  ollama serve
  ollama run llama3.2
"""

import os
from pathlib import Path

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class LlamaReaderConfig(BaseModel):
    """Configuration for the local Llama PDF reader."""

    model: str = Field(
        default="llama3.2",
        description="Ollama model name (e.g. llama3.2, llama3.1, llama3.2:1b)",
    )
    base_url: str | None = Field(
        default=None,
        description="Ollama API base URL (default: OLLAMA_HOST env or http://localhost:11434)",
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


def _get_ollama_client(base_url: str | None = None):
    """Return an Ollama client. Lazy import to keep ollama optional at import time."""
    try:
        from ollama import Client
    except ImportError as e:
        raise ImportError(
            "Install the Ollama client: pip install ollama. "
            "Then run Ollama locally (e.g. ollama serve && ollama run llama3.2)."
        ) from e
    url = (
        base_url
        or os.environ.get("OLLAMA_HOST")
        or "http://localhost:11434"
    )
    if not url.startswith("http"):
        url = f"http://{url}"
    return Client(host=url)


def _extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract text from a PDF file using pypdf (no LLM)."""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("Install pypdf: pip install pypdf") from e
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


class LlamaPDFReader:
    """
    Local PDF reader using Llama 3 (Ollama).

    Mirrors the Gemini reader interface:
      - transcribe(pdf_path) -> extract text from PDF (pypdf)
      - query_transcript(transcript, query_prompt) -> answer using Llama
      - query_pdf(pdf_path, query_prompt) -> transcribe then query
    """

    def __init__(self, config: LlamaReaderConfig | None = None):
        self.config = config or LlamaReaderConfig()
        self._client = None

    def _client_once(self):
        if self._client is None:
            self._client = _get_ollama_client(self.config.base_url)
        return self._client

    def transcribe(self, pdf_path: str | Path) -> str:
        """
        Extract text from a PDF file (using pypdf, no LLM).

        For layout-aware or image-heavy PDFs, consider using the Gemini
        transcriber instead; this is a fast local extraction.
        """
        return _extract_text_from_pdf(pdf_path)

    def transcribe_bytes(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes (writes to a temp file, then extracts)."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            path = f.name
        try:
            return _extract_text_from_pdf(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def query_transcript(self, transcript: str, query_prompt: str) -> str:
        """
        Answer a question about the given text using the local Llama model.
        """
        if not transcript.strip():
            return ""
        client = self._client_once()
        context = (
            "Use the following document text to answer the question. "
            "Answer only based on this content.\n\n"
            "--- Document ---\n"
            f"{transcript}\n"
            "--- End document ---\n\n"
            f"Question: {query_prompt}"
        )
        # Truncate if too long for context window (e.g. 8k tokens ~ 32k chars)
        max_chars = 28_000
        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n[Document truncated...]"
        response = client.chat(
            model=self.config.model,
            messages=[{"role": "user", "content": context}],
            options={"temperature": self.config.temperature},
        )
        content = response.get("message", {}).get("content", "")
        return (content or "").strip()

    def query_pdf(self, pdf_path: str | Path, query_prompt: str) -> str:
        """Extract text from the PDF, then answer the question using Llama."""
        transcript = self.transcribe(pdf_path)
        return self.query_transcript(transcript, query_prompt)

    def _list_pdfs(self, folder_path: str | Path, recursive: bool = False) -> list[Path]:
        folder = Path(folder_path)
        if not folder.is_dir():
            return []
        if recursive:
            return sorted(folder.rglob("*.pdf"))
        return sorted(folder.glob("*.pdf"))

    def query_folder(
        self,
        folder_path: str | Path,
        query_prompt: str,
        *,
        recursive: bool = False,
        tag_sources: bool = True,
    ) -> str:
        """Extract text from all PDFs in a folder, then answer the question using Llama."""
        folder = Path(folder_path)
        pdf_paths = self._list_pdfs(folder, recursive=recursive)
        if not pdf_paths:
            return ""
        parts = []
        for pdf_path in pdf_paths:
            try:
                text = self.transcribe(pdf_path)
            except Exception:
                continue
            if not text.strip():
                continue
            if tag_sources:
                parts.append(f"From: {pdf_path.name}\n\n{text}")
            else:
                parts.append(text)
        if not parts:
            return ""
        combined = "\n\n---\n\n".join(parts)
        return self.query_transcript(combined, query_prompt)


def transcribe_pdf_llama(pdf_path: str | Path, *, model: str = "llama3.2", base_url: str | None = None) -> str:
    """Convenience: extract text from PDF (pypdf). No Llama call for transcription."""
    config = LlamaReaderConfig(model=model, base_url=base_url)
    reader = LlamaPDFReader(config=config)
    return reader.transcribe(pdf_path)


def query_pdf_llama(
    pdf_path: str | Path,
    query_prompt: str,
    *,
    model: str = "llama3.2",
    base_url: str | None = None,
) -> str:
    """Convenience: extract PDF text and ask Llama one question."""
    config = LlamaReaderConfig(model=model, base_url=base_url)
    reader = LlamaPDFReader(config=config)
    return reader.query_pdf(pdf_path, query_prompt)


def main() -> None:
    """CLI: path (file or folder) and optional query. Prints transcription or answer."""
    import sys
    path = (
        sys.argv[1]
        if len(sys.argv) >= 2
        else os.environ.get("PDF_PATH")
    )
    query = " ".join(sys.argv[2:]).strip() if len(sys.argv) >= 3 else os.environ.get("QUERY_PROMPT", "").strip()
    if not path:
        print("Usage: python ideate/llama_reader.py <path_to.pdf | path_to_folder> [query]", file=sys.stderr)
        print("  Single file: python ideate/llama_reader.py doc.pdf 'What is this about?'", file=sys.stderr)
        print("  Folder:      python ideate/llama_reader.py store 'What do these documents say?'", file=sys.stderr)
        sys.exit(1)
    path = Path(path.strip())
    recursive = os.environ.get("RECURSIVE_PDF", "").strip().lower() in ("1", "true", "yes")
    config = LlamaReaderConfig(
        model=os.environ.get("LLAMA_MODEL", "llama3.2"),
        base_url=os.environ.get("OLLAMA_HOST") or None,
    )
    reader = LlamaPDFReader(config=config)
    if path.is_dir():
        if not query:
            print("Usage: pass a question as second argument when path is a folder.", file=sys.stderr)
            sys.exit(1)
        pdfs = reader._list_pdfs(path, recursive=recursive)
        if not pdfs:
            print("No PDF files found in that folder.", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(pdfs)} PDF(s), querying with Llama...", file=sys.stderr)
        result = reader.query_folder(path, query, recursive=recursive, tag_sources=True)
    else:
        if query:
            result = reader.query_pdf(path, query)
        else:
            result = reader.transcribe(path)
    if result:
        print(result)
    else:
        print("(No output)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
