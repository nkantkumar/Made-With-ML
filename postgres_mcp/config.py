"""Configuration for Postgres MCP (database + local Llama)."""

import os

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


class PostgresConfig(BaseModel):
    """PostgreSQL connection settings. Use env POSTGRES_* or DATABASE_URL."""

    database_url: str = Field(
        default="",
        description="PostgreSQL connection URL (e.g. postgresql://user:pass@localhost:5432/dbname)",
    )
    host: str = Field(default="localhost", description="DB host (used if database_url not set)")
    port: int = Field(default=5432, description="DB port")
    dbname: str = Field(default="postgres", description="Database name")
    user: str = Field(default="postgres", description="DB user")
    password: str = Field(default="", description="DB password")

    @classmethod
    def from_env(cls) -> "PostgresConfig":
        return cls(
            database_url=_env("DATABASE_URL"),
            host=_env("POSTGRES_HOST", "localhost"),
            port=int(_env("POSTGRES_PORT", "5432")),
            dbname=_env("POSTGRES_DBNAME", "postgres"),
            user=_env("POSTGRES_USER", "postgres"),
            password=_env("POSTGRES_PASSWORD"),
        )

    def get_dsn(self) -> str:
        if self.database_url:
            return self.database_url
        auth = f"{self.user}:{self.password}" if self.password else self.user
        return f"postgresql://{auth}@{self.host}:{self.port}/{self.dbname}"


class LlamaConfig(BaseModel):
    """Local Llama (Ollama) settings for SQL generation."""

    model: str = Field(default="llama3.2", description="Ollama model name")
    base_url: str | None = Field(default=None, description="Ollama API URL")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Lower = more deterministic SQL")

    @classmethod
    def from_env(cls) -> "LlamaConfig":
        base = _env("OLLAMA_HOST") or _env("LLAMA_BASE_URL") or None
        return cls(
            model=_env("LLAMA_MODEL", "llama3.2"),
            base_url=base if base else None,
            temperature=float(_env("LLAMA_TEMPERATURE", "0.1")),
        )

    def get_base_url(self) -> str:
        url = self.base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        return url if url.startswith("http") else f"http://{url}"
