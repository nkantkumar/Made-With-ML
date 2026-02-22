"""Query WhatsApp export using local Llama only (text + vision for images)."""

from pathlib import Path

from whatsapp_mcp.config import LlamaConfig
from whatsapp_mcp.parser import MediaFile, Message, load_export

_client = None


def _get_client(config: LlamaConfig | None = None):
    global _client
    if _client is None:
        try:
            from ollama import Client
        except ImportError as e:
            raise ImportError(
                "Install ollama: pip install ollama. Run: ollama serve && ollama run llama3.2"
            ) from e
        cfg = config or LlamaConfig.from_env()
        url = cfg.get_base_url()
        _client = (Client(host=url), cfg)
    return _client


def _messages_to_context(messages: list[Message], max_chars: int = 28000) -> str:
    parts = []
    for m in messages:
        line = f"[{m['date']}] {m['author']}: {m['text']}"
        if m.get("media_file"):
            line += f" [media: {m['media_file']}]"
        parts.append(line)
    text = "\n".join(parts)
    return text[:max_chars] + ("\n...[truncated]" if len(text) > max_chars else "")


def query_chat(
    question: str,
    messages: list[Message],
    *,
    config: LlamaConfig | None = None,
) -> str:
    """Answer a question over the chat text using Llama."""
    if not messages:
        return "No messages to query."
    client, cfg = _get_client(config)
    context = _messages_to_context(messages)
    prompt = (
        "Use only the following WhatsApp chat transcript to answer the question. "
        "If the answer is not in the chat, say so.\n\n"
        "--- Chat ---\n"
        f"{context}\n"
        "--- End chat ---\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    response = client.chat(
        model=cfg.model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": cfg.temperature},
    )
    return (response.get("message", {}).get("content") or "").strip()


def query_image(
    image_path: str | Path,
    question: str = "What is in this image? Describe briefly.",
    *,
    config: LlamaConfig | None = None,
) -> str:
    """Ask Llama vision about an image (photo). Uses llama3.2-vision by default."""
    path = Path(image_path)
    if not path.is_file():
        return f"File not found: {path}"
    client, cfg = _get_client(config)
    response = client.chat(
        model=cfg.vision_model,
        messages=[
            {
                "role": "user",
                "content": question,
                "images": [str(path.resolve())],
            }
        ],
        options={"temperature": cfg.temperature},
    )
    return (response.get("message", {}).get("content") or "").strip()


def describe_photos(
    media: list[MediaFile],
    *,
    config: LlamaConfig | None = None,
    max_photos: int = 50,
) -> list[tuple[str, str]]:
    """Describe each photo with Llama vision. Returns list of (filename, description)."""
    client, cfg = _get_client(config)
    photos = [m for m in media if m["type"] == "photo"][:max_photos]
    result = []
    for m in photos:
        try:
            desc = query_image(m["path"], "Describe this image in one or two sentences.", config=config)
            result.append((m["filename"], desc))
        except Exception as e:
            result.append((m["filename"], f"(error: {e})"))
    return result


def query_all(
    export_path: str | Path,
    question: str,
    *,
    config: LlamaConfig | None = None,
    include_photo_descriptions: bool = True,
    max_photos_to_describe: int = 20,
) -> str:
    """
    Load export, optionally describe photos with Llama vision, then answer the question
    over combined text (chat + photo descriptions + audio/video metadata).
    Llama only.
    """
    messages, media, _ = load_export(export_path)
    if not messages and not media:
        return "No chat or media found in the export."

    parts = []
    if messages:
        parts.append("WhatsApp chat:\n" + _messages_to_context(messages))

    if include_photo_descriptions and media:
        photos = [m for m in media if m["type"] == "photo"][:max_photos_to_describe]
        if photos:
            parts.append("\nPhoto descriptions (from Llama vision):")
            for m in photos:
                try:
                    desc = query_image(m["path"], "Describe in one sentence.", config=config)
                    parts.append(f"  - {m['filename']}: {desc}")
                except Exception as e:
                    parts.append(f"  - {m['filename']}: (skip: {e})")

    audio_video = [m for m in media if m["type"] in ("audio", "video")]
    if audio_video:
        parts.append("\nAudio/Video files (metadata only; content not transcribed):")
        for m in audio_video:
            parts.append(f"  - [{m['type']}] {m['filename']}")

    combined = "\n".join(parts)
    if len(combined) > 28000:
        combined = combined[:28000] + "\n...[truncated]"

    client, cfg = _get_client(config)
    prompt = (
        "Use only the following WhatsApp export content (chat + photo descriptions + media list) to answer the question.\n\n"
        "--- Content ---\n"
        f"{combined}\n"
        "--- End ---\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    response = client.chat(
        model=cfg.model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": cfg.temperature},
    )
    return (response.get("message", {}).get("content") or "").strip()
