"""
Postgres MCP: PostgreSQL transactions with local Llama (Ollama) for natural-language SQL.

Use transaction() or execute() for direct SQL. Use ask_and_run() to ask in plain
language and have Llama generate SQL, then run it in a transaction.
Use add_pdf_entry() to read a PDF and insert its text into a table.
"""

from postgres_mcp.config import LlamaConfig, PostgresConfig
from postgres_mcp.db import execute, execute_many, get_connection, transaction
from postgres_mcp.llama_sql import (
    ask_and_run,
    get_schema_hint,
    sql_from_natural_language,
)
from postgres_mcp.pdf_to_db import (
    add_pdf_entry,
    add_pdf_entries_from_folder,
    create_pdf_entries_table,
)

__all__ = [
    "PostgresConfig",
    "LlamaConfig",
    "get_connection",
    "transaction",
    "execute",
    "execute_many",
    "get_schema_hint",
    "sql_from_natural_language",
    "ask_and_run",
    "create_pdf_entries_table",
    "add_pdf_entry",
    "add_pdf_entries_from_folder",
]
