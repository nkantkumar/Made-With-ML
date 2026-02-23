"""Ollama-based embedding function for ChromaDB (local LLM only)."""

import os
from typing import Any

from whatsapp_mcp.config import LlamaConfig


def _get_ollama_client(base_url: str | None = None):
    from ollama import Client
    url = base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
    if not url.startswith("http"):
        url = f"http://{url}"
    return Client(host=url)


class OllamaChromaEmbeddingFunction:
    """
    ChromaDB-compatible embedding function using Ollama (local only).
    Use with nomic-embed-text: ollama pull nomic-embed-text
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str | None = None,
        config: LlamaConfig | None = None,
    ):
        cfg = config or LlamaConfig.from_env()
        self.model = os.environ.get("OLLAMA_EMBED_MODEL", model)
        self._base_url = base_url or cfg.get_base_url()
        self._client = None

    def _client_once(self):
        if self._client is None:
            self._client = _get_ollama_client(self._base_url)
        return self._client

    def __call__(self, input: Any) -> list[list[float]]:
        """
        Embed a list of text strings. ChromaDB passes documents as list of str.
        """
        if not input:
            return []
        texts = list(input) if isinstance(input, (list, tuple)) else [str(input)]
        client = self._client_once()
        # Ollama embed() accepts str or Sequence[str]
        try:
            out = client.embed(model=self.model, input=texts)
        except Exception as e:
            raise RuntimeError(
                f"Ollama embedding failed. Run: ollama pull {self.model}"
            ) from e
        # Response: EmbedResponse with .embeddings = Sequence[Sequence[float]]
        embeddings = getattr(out, "embeddings", None) or getattr(out, "embedding", None)
        if embeddings is None:
            raise RuntimeError("Ollama embed response missing embeddings")
        # If single input, Ollama might return one embedding
        if isinstance(embeddings, (list, tuple)) and len(embeddings) > 0:
            if isinstance(embeddings[0], (int, float)):
                return [list(embeddings)]
            return [list(e) for e in embeddings]
        return []
