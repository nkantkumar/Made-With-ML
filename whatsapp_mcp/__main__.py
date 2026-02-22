"""CLI: query a WhatsApp export with a natural language question. Llama only."""
import sys
from pathlib import Path

from whatsapp_mcp import query_all


def _resolve_export_path(path: Path) -> Path:
    """If path is a folder (e.g. store/), find a .zip inside and use it."""
    if not path.exists():
        return path
    if path.is_file():
        return path
    zips = list(path.glob("*.zip"))
    if len(zips) == 1:
        return zips[0]
    if len(zips) > 1:
        return zips[0]  # use first zip
    return path  # folder with chat.txt etc.


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage: python -m whatsapp_mcp <path_to_export.zip | folder (e.g. store)> \"Your question\"",
            file=sys.stderr,
        )
        print("  Export: ZIP file, or folder containing chat.txt (or a .zip inside, e.g. store/)", file=sys.stderr)
        print("  Example: python -m whatsapp_mcp store \"What did we say about the meeting?\"", file=sys.stderr)
        sys.exit(1)
    export_path = Path(sys.argv[1])
    question = " ".join(sys.argv[2:]).strip()
    if not question:
        print("Provide a question in quotes.", file=sys.stderr)
        sys.exit(1)
    if not export_path.exists():
        print(f"Not found: {export_path}", file=sys.stderr)
        sys.exit(1)
    export_path = _resolve_export_path(export_path)
    if export_path.is_file() and export_path.suffix.lower() == ".zip":
        print(f"Using export: {export_path}", file=sys.stderr)
    print("Loading export and querying with Llama...", file=sys.stderr)
    answer = query_all(export_path, question, include_photo_descriptions=True, max_photos_to_describe=15)
    print(answer)


if __name__ == "__main__":
    main()
