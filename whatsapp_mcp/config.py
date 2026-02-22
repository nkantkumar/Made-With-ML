"""Configuration for WhatsApp MCP (Llama/Ollama only)."""

import os

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


class LlamaConfig(BaseModel):
    """Ollama/Llama settings for chat and vision."""

    model: str = Field(default="llama3.2", description="Ollama model for text (e.g. llama3.2)")
    vision_model: str = Field(
        default="llama3.2-vision",
        description="Ollama model for images (e.g. llama3.2-vision)",
    )
    base_url: str | None = Field(default=None, description="Ollama API URL")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    @classmethod
    def from_env(cls) -> "LlamaConfig":
        base = _env("OLLAMA_HOST") or _env("LLAMA_BASE_URL") or None
        return cls(
            model=_env("LLAMA_MODEL", "llama3.2"),
            vision_model=_env("LLAMA_VISION_MODEL", "llama3.2-vision"),
            base_url=base if base else None,
            temperature=float(_env("LLAMA_TEMPERATURE", "0.2")),
        )

    def get_base_url(self) -> str:
        url = self.base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        return url if url.startswith("http") else f"http://{url}"
