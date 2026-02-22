# Store

Project-level folder for assets and inputs.

- **PDFs for RAG:** Put PDF files here and run:
  ```bash
  python examples/pdf_rag_qa.py store "Your question"
  ```
- **Single PDF:** Set `PDF_PATH=store/yourfile.pdf` in `.env` or pass the path to the transcriber / RAG script.
- **WhatsApp export:** Put your chat export `.zip` here, then run:
  ```bash
  python -m whatsapp_mcp store "Your question"
  ```
  (The CLI uses the first `.zip` in `store/` if you pass the folder.)

You can add subfolders; use `RECURSIVE_PDF=1` in `.env` to index PDFs in subfolders.
