"""Use local Llama (Ollama) to generate SQL and run it in a transaction."""

from postgres_mcp.config import LlamaConfig, PostgresConfig
from postgres_mcp.db import execute, transaction


def _get_ollama_client(base_url: str):
    from ollama import Client
    url = base_url if base_url.startswith("http") else f"http://{base_url}"
    return Client(host=url)


def get_schema_hint(config: PostgresConfig | None = None) -> str:
    """Fetch table and column names from the database to give Llama context."""
    cfg = config or PostgresConfig.from_env()
    sql = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
    ORDER BY table_name, ordinal_position
    """
    try:
        rows = execute(sql, config=cfg, fetch=True)
    except Exception:
        return ""
    if not rows:
        return ""
    lines = []
    prev = None
    for table, col, dtype in rows:
        if table != prev:
            lines.append(f"\n{table} ({dtype}):")
            prev = table
        lines.append(f"  - {col}")
    return "Tables and columns:" + "\n".join(lines).strip() if lines else ""


def sql_from_natural_language(
    question: str,
    *,
    schema_hint: str = "",
    llama_config: LlamaConfig | None = None,
) -> str:
    """
    Ask local Llama to generate a single SQL statement for the given question.
    Optionally pass schema_hint (e.g. from get_schema_hint()) for better SQL.
    """
    cfg = llama_config or LlamaConfig.from_env()
    client = _get_ollama_client(cfg.get_base_url())
    prompt = (
        "You are a PostgreSQL expert. Reply with ONLY a single valid SQL statement, no explanation.\n"
    )
    if schema_hint:
        prompt += f"\n{schema_hint}\n\n"
    prompt += f"Question: {question}\n\nSQL:"
    response = client.generate(
        model=cfg.model,
        prompt=prompt,
        options={"temperature": cfg.temperature},
    )
    sql = (response.get("response") or "").strip()
    # Trim markdown code block if present
    if sql.startswith("```"):
        lines = sql.split("\n")
        out = []
        for line in lines:
            if line.strip() == "```" or line.strip().startswith("```sql"):
                continue
            out.append(line)
        sql = "\n".join(out)
    return sql.strip()


def ask_and_run(
    question: str,
    *,
    postgres_config: PostgresConfig | None = None,
    llama_config: LlamaConfig | None = None,
    include_schema: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Use Llama to generate SQL from a natural language question, then run it in a transaction.

    Returns:
        {"sql": str, "rows": list | None, "error": str | None}
        - For SELECT: rows is the result set.
        - For INSERT/UPDATE/DELETE: rows is None.
        - On exception: error is set, sql/rows may be present.
    """
    pg_cfg = postgres_config or PostgresConfig.from_env()
    llm_cfg = llama_config or LlamaConfig.from_env()
    schema = get_schema_hint(pg_cfg) if include_schema else ""
    try:
        sql = sql_from_natural_language(question, schema_hint=schema, llama_config=llm_cfg)
    except Exception as e:
        return {"sql": "", "rows": None, "error": str(e)}

    if dry_run:
        return {"sql": sql, "rows": None, "error": None}

    upper = sql.upper().strip()
    fetch = upper.startswith("SELECT") or upper.startswith("WITH")
    try:
        rows = execute(sql, config=pg_cfg, fetch=fetch)
        return {"sql": sql, "rows": rows, "error": None}
    except Exception as e:
        return {"sql": sql, "rows": None, "error": str(e)}
