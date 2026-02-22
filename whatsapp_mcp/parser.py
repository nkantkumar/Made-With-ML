"""Parse WhatsApp chat export (ZIP or folder with chat.txt and optional Media)."""

import re
import zipfile
from pathlib import Path
from typing import TypedDict


class Message(TypedDict):
    date: str
    author: str
    text: str
    media_file: str | None


class MediaFile(TypedDict):
    path: str
    filename: str
    type: str  # "photo" | "audio" | "video"


# Common WhatsApp export patterns (Android / iOS style)
# e.g. "12/04/2024, 3:25 PM - Name: message" or "[12/04/2024, 3:25:33 PM] Name: message"
MSG_PATTERN = re.compile(
    r"^\[?(?P<date>[^\]]+?)\]?\s*[-–]\s*(?P<author>[^:]+):\s*(?P<text>.*)$",
    re.MULTILINE,
)
# Media reference in message: <attached: filename> or similar
MEDIA_REF = re.compile(r"<[^>]*attached[^>]*:\s*([^>]+)>", re.IGNORECASE)


def _find_chat_file(folder: Path) -> Path | None:
    for name in ("_chat.txt", "chat.txt", "WhatsApp Chat.txt"):
        p = folder / name
        if p.is_file():
            return p
    for f in folder.iterdir():
        if f.suffix.lower() == ".txt" and "chat" in f.name.lower():
            return f
    return None


def _parse_chat_line(line: str) -> Message | None:
    line = line.strip()
    if not line or line.startswith("Messages to this chat") or "created this group" in line.lower():
        return None
    m = MSG_PATTERN.match(line)
    if not m:
        return None
    date, author, text = m.group("date", "author", "text")
    author = author.strip()
    media_ref = MEDIA_REF.search(text)
    media_file = media_ref.group(1).strip() if media_ref else None
    return {"date": date.strip(), "author": author, "text": text.strip(), "media_file": media_file}


def parse_chat(chat_path: str | Path) -> list[Message]:
    """Parse a WhatsApp chat .txt file. Returns list of messages."""
    path = Path(chat_path)
    if not path.is_file():
        return []
    messages = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        msg = _parse_chat_line(line)
        if msg:
            messages.append(msg)
    return messages


def list_media(media_folder: str | Path) -> list[MediaFile]:
    """List media files and classify by extension. Returns list of {path, filename, type}."""
    folder = Path(media_folder)
    if not folder.is_dir():
        return []
    photo_ext = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    audio_ext = {".ogg", ".mp3", ".m4a", ".opus", ".wav"}
    video_ext = {".mp4", ".webm", ".mov", ".avi", ".3gp"}
    result = []
    for f in folder.rglob("*"):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in photo_ext:
            kind = "photo"
        elif ext in audio_ext:
            kind = "audio"
        elif ext in video_ext:
            kind = "video"
        else:
            continue
        result.append({"path": str(f.resolve()), "filename": f.name, "type": kind})
    return result


def load_export(export_path: str | Path) -> tuple[list[Message], list[MediaFile], Path]:
    """
    Load WhatsApp export from a ZIP or folder.
    Returns (messages, media_files, chat_folder_path).
    If export_path is a ZIP, extracts to a temp dir or sibling folder.
    """
    path = Path(export_path).resolve()
    chat_folder = path
    if path.suffix.lower() == ".zip":
        import tempfile
        dest = Path(tempfile.mkdtemp(prefix="wa_export_"))
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(dest)
        # Find chat file: might be at root or inside a single folder
        chat_file = _find_chat_file(dest)
        if chat_file:
            chat_folder = chat_file.parent
        else:
            for sub in dest.iterdir():
                if sub.is_dir():
                    chat_file = _find_chat_file(sub)
                    if chat_file:
                        chat_folder = chat_file.parent
                        break
    chat_file = _find_chat_file(chat_folder)
    if not chat_file:
        return [], [], chat_folder
    messages = parse_chat(chat_file)
    media_folder = chat_folder / "Media" if (chat_folder / "Media").is_dir() else chat_folder
    media = list_media(media_folder)
    return messages, media, chat_folder
