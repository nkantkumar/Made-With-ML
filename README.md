# Made-With-ML

## Setup & run (PDF transcriber)

Your shell’s `python` may point to a different interpreter than the one where you installed dependencies. Use one of these:

**Option A – Project virtual environment (recommended)**

```bash
python3 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ideate
```

**Option B – Same Python as install**

If you use pyenv and installed with pyenv’s Python 3.10:

```bash
pyenv local 3.10.0   # or use this project’s Python
pip install -r requirements.txt
python -m ideate
```

Set `GEMINI_API_KEY` and optionally `PDF_PATH` in `.env` (copy from `.env.example`).
