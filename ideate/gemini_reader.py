"""Gemini LLM reader for transcribing PDF files."""

import os
from pathlib import Path

from pydantic import BaseModel, Field

# Load GEMINI_API_KEY / GOOGLE_API_KEY from .env when python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Lazy import so google.genai is only required when this module is used
_client = None


def _get_client(api_key: str | None = None):
    global _client
    if _client is None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError(
                "Install the Gemini SDK: pip install -r requirements.txt"
            ) from e
        key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not key:
            raise ValueError(
                "Set GEMINI_API_KEY or GOOGLE_API_KEY in a .env file (copy .env.example to .env), "
                "in the environment, or pass api_key to GeminiPDFTranscriber."
            )
        _client = genai.Client(api_key=key)
    return _client


class TranscriberConfig(BaseModel):
    """Configuration for the Gemini PDF transcriber."""

    model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model name (e.g. gemini-2.0-flash, gemini-1.5-pro)",
    )
    api_key: str | None = Field(default=None, description="Override API key from env")
    prompt: str = Field(
        default=(
            "Transcribe this PDF document. Output the full text content, "
            "preserving structure (headings, paragraphs, lists) where possible. "
            "If there are tables, reproduce them in text form. "
            "If there are images or diagrams, describe them briefly. "
            "Return only the transcribed text, no extra commentary."
        ),
        description="Instruction sent to the model for transcription.",
    )


class GeminiPDFTranscriber:
    """Transcribe PDF files using the Gemini LLM."""

    def __init__(self, config: TranscriberConfig | None = None):
        self.config = config or TranscriberConfig()

    def transcribe(self, pdf_path: str | Path) -> str:
        """
        Transcribe a PDF file to text using Gemini.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Transcribed text from the PDF.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If no API key is configured.
        """
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")

        pdf_bytes = path.read_bytes()
        return self.transcribe_bytes(pdf_bytes)

    def transcribe_bytes(self, pdf_bytes: bytes) -> str:
        """
        Transcribe PDF content from raw bytes using Gemini.

        Args:
            pdf_bytes: Raw PDF file content.

        Returns:
            Transcribed text from the PDF.
        """
        from google.genai import types

        client = _get_client(self.config.api_key)

        pdf_part = types.Part(
            inline_data=types.Blob(
                mime_type="application/pdf",
                data=pdf_bytes,
            )
        )
        prompt_part = types.Part.from_text(text=self.config.prompt)

        response = client.models.generate_content(
            model=self.config.model,
            contents=[pdf_part, prompt_part],
            config=types.GenerateContentConfig(
                temperature=0.2,
            ),
        )

        if not response.text:
            return ""
        return response.text.strip()

    def query_transcript(self, transcript: str, query_prompt: str) -> str:
        """
        Answer a question about already-transcribed text using Gemini.

        Args:
            transcript: Full transcribed text (e.g. from transcribe()).
            query_prompt: Your question or instruction (e.g. "Summarize the key points.").

        Returns:
            The model's answer based on the transcript.
        """
        from google.genai import types

        if not transcript.strip():
            return ""

        client = _get_client(self.config.api_key)
        context = (
            "Use the following transcribed document text to answer the question. "
            "Answer only based on this content.\n\n"
            "--- Document ---\n"
            f"{transcript}\n"
            "--- End document ---\n\n"
            f"Question: {query_prompt}"
        )
        response = client.models.generate_content(
            model=self.config.model,
            contents=types.Part.from_text(text=context),
            config=types.GenerateContentConfig(temperature=0.2),
        )
        if not response.text:
            return ""
        return response.text.strip()

    def query_pdf(self, pdf_path: str | Path, query_prompt: str) -> str:
        """
        Transcribe a PDF and then answer a question about its content.

        Args:
            pdf_path: Path to the PDF file.
            query_prompt: Question or instruction about the document.

        Returns:
            The model's answer.
        """
        transcript = self.transcribe(pdf_path)
        return self.query_transcript(transcript, query_prompt)


def transcribe_pdf(
    pdf_path: str | Path,
    *,
    model: str = "gemini-2.5-flash",
    api_key: str | None = None,
    prompt: str | None = None,
) -> str:
    """
    Convenience function to transcribe a PDF with Gemini.

    Args:
        pdf_path: Path to the PDF file.
        model: Gemini model name.
        api_key: Optional API key (otherwise uses GEMINI_API_KEY / GOOGLE_API_KEY).
        prompt: Optional custom transcription prompt.

    Returns:
        Transcribed text from the PDF.
    """
    config_kw: dict = {"model": model, "api_key": api_key}
    if prompt is not None:
        config_kw["prompt"] = prompt
    config = TranscriberConfig(**config_kw)
    transcriber = GeminiPDFTranscriber(config=config)
    return transcriber.transcribe(pdf_path)


def query_transcript(
    transcript: str,
    query_prompt: str,
    *,
    model: str = "gemini-2.5-flash",
    api_key: str | None = None,
) -> str:
    """
    Ask a question about transcribed text. Uses Gemini to answer based on the text.

    Args:
        transcript: Full transcribed document text.
        query_prompt: Your question (e.g. "What are the main conclusions?").
        model: Gemini model name.
        api_key: Optional API key override.

    Returns:
        The model's answer.
    """
    config = TranscriberConfig(model=model, api_key=api_key)
    transcriber = GeminiPDFTranscriber(config=config)
    return transcriber.query_transcript(transcript, query_prompt)


def query_pdf(
    pdf_path: str | Path,
    query_prompt: str,
    *,
    model: str = "gemini-2.5-flash",
    api_key: str | None = None,
    transcribe_prompt: str | None = None,
) -> str:
    """
    Transcribe a PDF and answer a question about its content in one go.

    Args:
        pdf_path: Path to the PDF file.
        query_prompt: Question or instruction (e.g. "Summarize in 3 bullet points.").
        model: Gemini model name.
        api_key: Optional API key override.
        transcribe_prompt: Optional custom transcription prompt (default transcribes fully).

    Returns:
        The model's answer.
    """
    config_kw: dict = {"model": model, "api_key": api_key}
    if transcribe_prompt is not None:
        config_kw["prompt"] = transcribe_prompt
    config = TranscriberConfig(**config_kw)
    transcriber = GeminiPDFTranscriber(config=config)
    return transcriber.query_pdf(pdf_path, query_prompt)


def main() -> None:
    """CLI entry point. Path from sys.argv or PDF_PATH env. Easy to debug by running this file."""
    import sys

    path = (
        sys.argv[1]
        if len(sys.argv) >= 2
        else os.environ.get("PDF_PATH")
    )
    # Optional query: if 3+ args, argv[2:] is the prompt (e.g. "Summarize the key points")
    query = " ".join(sys.argv[2:]).strip() if len(sys.argv) >= 3 else None

    if not path:
        print("Usage: python -m ideate [path_to.pdf] [query]")
        print("  Transcribe only:  python -m ideate doc.pdf")
        print("  Query content:    python -m ideate doc.pdf 'What are the key points?'")
        print("Or set PDF_PATH (and optionally QUERY_PROMPT) in .env")
        sys.exit(1)

    path = path.strip()
    # If PyCharm/IDE passed "script.py  file.pdf" as one arg, use the .pdf part
    if " " in path and not query:
        pdf_part = next((p for p in path.split() if p.endswith(".pdf")), None)
        if pdf_part is not None:
            path = pdf_part

    if not query and os.environ.get("QUERY_PROMPT"):
        query = os.environ.get("QUERY_PROMPT", "").strip()

    try:
        if query:
            print(query_pdf(path, query))
        else:
            print(transcribe_pdf(path))
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
            print("Error: Gemini API rate limit or quota exceeded.", file=sys.stderr)
            print(
                "Wait a few minutes or check https://ai.google.dev/gemini-api/docs/rate-limits",
                file=sys.stderr,
            )
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
