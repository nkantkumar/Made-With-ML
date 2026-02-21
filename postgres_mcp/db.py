"""PostgreSQL connection and transaction support."""

from contextlib import contextmanager
from typing import Any, Generator

from postgres_mcp.config import PostgresConfig


def get_connection(config: PostgresConfig | None = None):
    """Open a new connection. Caller must close or use transaction() instead."""
    import psycopg2
    cfg = config or PostgresConfig.from_env()
    return psycopg2.connect(cfg.get_dsn())


@contextmanager
def transaction(config: PostgresConfig | None = None) -> Generator[Any, None, None]:
    """
    Context manager for a single database transaction.
    Commits on success, rolls back on exception. Closes the connection on exit.
    """
    cfg = config or PostgresConfig.from_env()
    import psycopg2
    conn = psycopg2.connect(cfg.get_dsn())
    try:
        with conn:
            with conn.cursor() as cur:
                yield cur
    finally:
        conn.close()


def execute(
    sql: str,
    params: tuple | dict | None = None,
    *,
    config: PostgresConfig | None = None,
    fetch: bool = True,
) -> list[tuple] | None:
    """
    Run a single statement in a transaction. Commits on success, rolls back on error.

    Args:
        sql: SQL statement (use %s placeholders if params given).
        params: Optional query parameters (tuple or dict).
        config: Optional Postgres config; uses env if not set.
        fetch: If True (default), return fetched rows for SELECT; else None.

    Returns:
        List of rows for SELECT when fetch=True, else None.
    """
    with transaction(config) as cur:
        cur.execute(sql, params)
        if fetch:
            return cur.fetchall()
    return None


def execute_many(
    sql: str,
    params_list: list[tuple] | list[dict],
    *,
    config: PostgresConfig | None = None,
) -> None:
    """Execute the same statement with multiple parameter sets in one transaction."""
    with transaction(config) as cur:
        cur.executemany(sql, params_list)
