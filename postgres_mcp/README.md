# Postgres MCP

PostgreSQL transaction support with **local Llama** (Ollama) for natural-language SQL.

## How to run and check

**1. Start services**

```bash
# PostgreSQL (if local: start your Postgres server, or use a cloud DB URL)
# Ollama (for Llama SQL generation)
ollama serve
ollama run llama3.2
```

**2. Set env**

From project root, copy and edit `.env`:

```bash
cp .env.example .env
# Edit .env: set DATABASE_URL or POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD, etc.
```

**3. Install and run the check script**

```bash
cd /path/to/Made-With-ML
pip install -r requirements.txt
python postgres_mcp/check.py
```

This checks: (1) DB connection, (2) schema hint, (3) Llama/Ollama. Fix any reported step before using `ask_and_run`.

**4. Check Llama + DB together (natural language → SQL)**

```bash
# Ensure you have at least one table (e.g. public.users). Then:
python -c "
from postgres_mcp import ask_and_run
r = ask_and_run('How many rows are in the users table?', dry_run=True)
print('Generated SQL:', r['sql'])
print('Error:', r['error'])
# Remove dry_run=True to actually run the SQL
r = ask_and_run('How many rows are in the users table?')
print('Rows:', r['rows'])
"
```

**5. Run from project root**

Always run Python from the **project root** (where `postgres_mcp/` and `.env` live) so imports and env work:

```bash
cd /path/to/Made-With-ML
python -c "from postgres_mcp import execute; print(execute('SELECT 1', fetch=True))"
```

---

## Setup (reference)

1. **PostgreSQL** running (local or remote).
2. **Ollama** with a model (e.g. `ollama run llama3.2`).
3. **Env** (copy to `.env`):

```bash
# One of:
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Or:
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DBNAME=postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=yourpassword

# Llama (optional; defaults shown)
OLLAMA_HOST=http://localhost:11434
LLAMA_MODEL=llama3.2
```

4. **Install**: `pip install -r requirements.txt` (adds `psycopg2-binary`, `ollama`).

## Direct transactions

```python
from postgres_mcp import transaction, execute, PostgresConfig

# Single statement (auto commit/rollback)
rows = execute("SELECT * FROM users WHERE id = %s", (1,), fetch=True)

# Multiple statements in one transaction
with transaction() as cur:
    cur.execute("INSERT INTO logs (msg) VALUES (%s)", ("hello",))
    cur.execute("UPDATE counters SET n = n + 1 WHERE name = %s", ("logs",))
```

## Natural language → SQL (Llama) → run in transaction

```python
from postgres_mcp import ask_and_run

# Llama generates SQL, then we run it in a transaction
result = ask_and_run("How many users are there?")
print(result["sql"])   # e.g. SELECT COUNT(*) FROM users;
print(result["rows"])  # [(42,)]
print(result["error"]) # None or error message

# Dry run: only generate SQL, do not execute
result = ask_and_run("List all active users", dry_run=True)
print(result["sql"])
```

## Read PDF and add entry to table

Extract text from a PDF and insert one row into a table (`pdf_entries` by default). Table schema: `id`, `source_path`, `content`, `created_at`. The table is created automatically if it does not exist.

**CLI (from project root):**

```bash
# Single PDF (extract with pypdf, local)
python -m postgres_mcp path/to/doc.pdf

# Single PDF using Gemini for transcription (set GEMINI_API_KEY)
python -m postgres_mcp path/to/doc.pdf --gemini

# All PDFs in a folder
python -m postgres_mcp store
python -m postgres_mcp store --recursive
```

**In code:**

```python
from postgres_mcp import add_pdf_entry, add_pdf_entries_from_folder, create_pdf_entries_table

# One PDF → one row (default: pypdf extraction)
id_ = add_pdf_entry("store/doc.pdf")
print("Inserted id:", id_)

# Use Gemini for richer transcription
id_ = add_pdf_entry("store/doc.pdf", extract_with="gemini")

# Folder of PDFs
ids = add_pdf_entries_from_folder("store", recursive=True)
print("Inserted ids:", ids)

# Create table only (optional)
create_pdf_entries_table(table_name="pdf_entries")
```

Then query with Llama: `ask_and_run("Summarize the content of the last PDF we added")` after adding a row, or query the table directly.

**Check your inserted entries (e.g. id=3):**

```bash
# Option 1: psql (use your DB name/user)
psql -U postgres -d your_dbname -c "SELECT id, source_path, left(content, 80) AS content_preview, created_at FROM pdf_entries ORDER BY id DESC LIMIT 5;"
# Single row by id
psql -U postgres -d your_dbname -c "SELECT * FROM pdf_entries WHERE id = 3;"
```

```bash
# Option 2: Python one-liner (from project root; uses .env)
python -c "
from postgres_mcp import execute
rows = execute('SELECT id, source_path, left(content, 100) AS preview, created_at FROM pdf_entries WHERE id = %s', (3,), fetch=True)
print(rows)
# Or list latest 5
rows = execute('SELECT id, source_path, created_at FROM pdf_entries ORDER BY id DESC LIMIT 5', fetch=True)
for r in rows: print(r)
"
```

---

## API

- **`transaction(config=None)`** – Context manager: yields a cursor; commit on success, rollback on exception.
- **`execute(sql, params=None, config=None, fetch=True)`** – Run one statement in a transaction; returns rows if `fetch=True`.
- **`execute_many(sql, params_list, config=None)`** – Same statement, many parameter sets, one transaction.
- **`ask_and_run(question, ..., include_schema=True, dry_run=False)`** – Natural language → Llama → SQL → execute in transaction. Returns `{"sql", "rows", "error"}`.
- **`sql_from_natural_language(question, schema_hint="", llama_config=None)`** – Only generate SQL (no execution).
- **`get_schema_hint(config=None)`** – Fetch public tables/columns for Llama context.
- **`create_pdf_entries_table(table_name="pdf_entries", config=None)`** – Create the PDF entries table if not exists.
- **`add_pdf_entry(pdf_path, table_name="pdf_entries", extract_with="pypdf"|"gemini", create_table=True)`** – Read PDF, insert one row; returns inserted id.
- **`add_pdf_entries_from_folder(folder_path, table_name="pdf_entries", extract_with="pypdf"|"gemini", recursive=False)`** – Insert one row per PDF in folder; returns list of ids.
