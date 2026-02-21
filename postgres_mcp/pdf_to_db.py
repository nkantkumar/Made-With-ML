"""
Read a PDF and add an entry into a Postgres table.

Extracts text from the PDF (pypdf, or optional Gemini via ideate), then inserts
(source_path, content, created_at) into a table. Creates the table if missing.
"""

import re
from pathlib import Path
from typing import Literal

from postgres_mcp.config import PostgresConfig
from postgres_mcp.db import execute, transaction

DEFAULT_TABLE = "pdf_entries"
_VALID_TABLE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _check_table_name(name: str) -> None:
    if not _VALID_TABLE.match(name):
        raise ValueError(f"Invalid table name: {name!r}")


def _extract_text_pypdf(pdf_path: str | Path) -> str:
    """Extract text from PDF using pypdf (local, no API)."""
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


def _extract_text_ideate(pdf_path: str | Path) -> str:
    """Extract/transcribe PDF using ideate (Gemini). Richer for complex PDFs."""
    try:
        from ideate import transcribe_pdf
    except ImportError as e:
        raise ImportError("Install ideate (and set GEMINI_API_KEY) to use transcribe_pdf") from e
    return transcribe_pdf(pdf_path)


def create_pdf_entries_table(
    table_name: str = DEFAULT_TABLE,
    config: PostgresConfig | None = None,
) -> None:
    """
    Create the table used by add_pdf_entry if it does not exist.

    Schema: id (serial), source_path (text), content (text), created_at (timestamptz).
    """
    _check_table_name(table_name)
    sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        source_path TEXT NOT NULL,
        content TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    )
    """
    execute(sql, config=config, fetch=False)


def add_pdf_entry(
    pdf_path: str | Path,
    *,
    table_name: str = DEFAULT_TABLE,
    config: PostgresConfig | None = None,
    extract_with: Literal["pypdf", "gemini"] = "pypdf",
    create_table: bool = True,
) -> int:
    """
    Read the PDF, extract text, and insert one row into the table. Returns the new row id.

    Args:
        pdf_path: Path to the PDF file.
        table_name: Table to insert into (default: pdf_entries).
        config: Postgres config; uses env if not set.
        extract_with: "pypdf" (local, fast) or "gemini" (ideate, better for complex PDFs).
        create_table: If True, create the table if it does not exist.

    Returns:
        Inserted row id (id column).
    """
    _check_table_name(table_name)
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    cfg = config or PostgresConfig.from_env()

    if extract_with == "gemini":
        content = _extract_text_ideate(path)
    else:
        content = _extract_text_pypdf(path)

    source_path = str(path.resolve())

    if create_table:
        create_pdf_entries_table(table_name=table_name, config=cfg)

    with transaction(cfg) as cur:
        cur.execute(
            f"INSERT INTO {table_name} (source_path, content) VALUES (%s, %s) RETURNING id",
            (source_path, content),
        )
        row = cur.fetchone()
        return row[0] if row else 0


def add_pdf_entries_from_folder(
    folder_path: str | Path,
    *,
    table_name: str = DEFAULT_TABLE,
    config: PostgresConfig | None = None,
    extract_with: Literal["pypdf", "gemini"] = "pypdf",
    recursive: bool = False,
) -> list[int]:
    """
    Find all PDFs in a folder, extract text from each, and insert one row per PDF.
    Returns list of inserted row ids.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    pdfs = sorted(folder.rglob("*.pdf") if recursive else folder.glob("*.pdf"))
    ids = []
    for pdf_path in pdfs:
        try:
            id_ = add_pdf_entry(
                pdf_path,
                table_name=table_name,
                config=config,
                extract_with=extract_with,
                create_table=(len(ids) == 0),
            )
            ids.append(id_)
        except Exception:
            raise
    return ids


def main() -> None:
    """CLI: read a PDF (or folder) and add entry/entries to pdf_entries table."""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m postgres_mcp <path_to.pdf | path_to_folder> [--gemini] [--recursive]", file=sys.stderr)
        print("  Single PDF:  python -m postgres_mcp store/doc.pdf", file=sys.stderr)
        print("  Folder:      python -m postgres_mcp store [--recursive]", file=sys.stderr)
        print("  Use Gemini:  add --gemini to use ideate transcribe (set GEMINI_API_KEY)", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1])
    extract_with = "gemini" if "--gemini" in sys.argv else "pypdf"
    recursive = "--recursive" in sys.argv
    if path.is_file():
        pid = add_pdf_entry(path, extract_with=extract_with)
        print(f"Inserted PDF entry id={pid}")
    elif path.is_dir():
        ids = add_pdf_entries_from_folder(path, extract_with=extract_with, recursive=recursive)
        print(f"Inserted {len(ids)} PDF entries: ids={ids}")
    else:
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
