#!/usr/bin/env python3
"""
Run this from project root to verify Postgres MCP setup:
  python postgres_mcp/check.py
  python -m postgres_mcp.check
"""

import sys
from pathlib import Path

# Ensure project root is on path
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))


def main() -> None:
    print("Postgres MCP check (run from project root)\n")

    # 1. DB connection
    try:
        from postgres_mcp import execute, get_schema_hint
        rows = execute("SELECT 1 AS ok", fetch=True)
        print("1. DB connection: OK", rows)
    except Exception as e:
        print("1. DB connection: FAIL", e)
        print("   Set DATABASE_URL or POSTGRES_* in .env")
        sys.exit(1)

    # 2. Schema hint (optional)
    try:
        hint = get_schema_hint()
        if hint:
            print("2. Schema hint: OK (public tables/columns loaded)")
        else:
            print("2. Schema hint: OK (no public tables or empty)")
    except Exception as e:
        print("2. Schema hint: FAIL", e)

    # 3. Llama (Ollama) for SQL generation
    try:
        from postgres_mcp import sql_from_natural_language
        sql = sql_from_natural_language("Return only this SQL: SELECT 1 AS test;")
        print("3. Llama (Ollama): OK")
        print("   Sample generated SQL:", (sql or "")[:80] + ("..." if len(sql or "") > 80 else ""))
    except Exception as e:
        print("3. Llama (Ollama): FAIL", e)
        print("   Run: ollama serve && ollama run llama3.2")

    print("\nDone. Use ask_and_run('your question') to run natural-language SQL.")


if __name__ == "__main__":
    main()
