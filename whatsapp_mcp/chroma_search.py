"""
ChromaDB-backed semantic search over WhatsApp messages. Local LLM only (Ollama embeddings).
"""

import hashlib
from pathlib import Path
from typing import Any

from whatsapp_mcp.config import LlamaConfig
from whatsapp_mcp.embeddings import OllamaChromaEmbeddingFunction
from whatsapp_mcp.parser import MediaFile, Message, load_export


def _collection_name(export_path: Path) -> str:
    path_str = str(export_path.resolve())
    h = hashlib.sha256(path_str.encode()).hexdigest()[:16]
    return f"whatsapp_{h}"


def _message_to_doc(m: Message) -> str:
    line = f"[{m['date']}] {m['author']}: {m['text']}"
    if m.get("media_file"):
        line += f" [media: {m['media_file']}]"
    return line


def get_chroma_client(persist_dir: str | Path | None = None):
    """Return a ChromaDB client. If persist_dir is set, use PersistentClient."""
    try:
        import chromadb
    except Exception as e:
        err = str(e).lower()
        if "pydantic" in err or "config" in err or "3.14" in str(e):
            raise RuntimeError(
                "ChromaDB has known compatibility issues with Python 3.14 (Pydantic v1). "
                "Use Python 3.12 or 3.11 for --semantic, or run without --semantic: "
                "python -m whatsapp_mcp store \"Your question\""
            ) from e
        raise
    if persist_dir:
        return chromadb.PersistentClient(path=str(persist_dir))
    return chromadb.Client()


def build_index(
    export_path: str | Path,
    *,
    persist_dir: str | Path | None = None,
    config: LlamaConfig | None = None,
    include_photo_descriptions: bool = True,
    max_photos: int = 30,
) -> str:
    """
    Load WhatsApp export, embed messages (and optional photo descriptions), store in ChromaDB.
    Returns the collection name. Uses Ollama nomic-embed-text (run: ollama pull nomic-embed-text).
    """
    messages, media, _ = load_export(export_path)
    if not messages:
        return ""

    path = Path(export_path).resolve()
    coll_name = _collection_name(path)
    client = get_chroma_client(persist_dir)
    embed_fn = OllamaChromaEmbeddingFunction(config=config)

    # Build documents: one per message (+ optional photo description docs)
    ids = []
    docs = []
    metadatas = []

    for i, m in enumerate(messages):
        ids.append(f"msg_{i}")
        docs.append(_message_to_doc(m))
        metadatas.append({"date": m["date"], "author": m["author"], "kind": "message"})

    if include_photo_descriptions and media:
        from whatsapp_mcp.query import query_image
        photos = [x for x in media if x["type"] == "photo"][:max_photos]
        for j, p in enumerate(photos):
            try:
                desc = query_image(p["path"], "Describe in one sentence.", config=config)
                ids.append(f"photo_{j}")
                docs.append(f"Photo {p['filename']}: {desc}")
                metadatas.append({"filename": p["filename"], "kind": "photo"})
            except Exception:
                continue

    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=coll_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    batch_size = 50
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metadatas[start:end],
        )
    return coll_name


def semantic_query(
    export_path: str | Path,
    question: str,
    *,
    n_results: int = 10,
    persist_dir: str | Path | None = None,
    config: LlamaConfig | None = None,
    rebuild: bool = False,
) -> str:
    """
    Answer a question using ChromaDB semantic search (local Ollama embeddings) + Llama.
    Builds the index if missing (or if rebuild=True), retrieves top-k relevant messages,
    then asks Llama to answer from that context.
    """
    path = Path(export_path).resolve()
    if not path.exists():
        return f"Path not found: {path}"
    coll_name = _collection_name(path)
    client = get_chroma_client(persist_dir)
    embed_fn = OllamaChromaEmbeddingFunction(config=config)

    if rebuild:
        build_index(export_path, persist_dir=persist_dir, config=config, include_photo_descriptions=True, max_photos=30)
    try:
        collection = client.get_collection(name=coll_name, embedding_function=embed_fn)
    except Exception:
        build_index(export_path, persist_dir=persist_dir, config=config)
        collection = client.get_collection(name=coll_name, embedding_function=embed_fn)

    results = collection.query(query_texts=[question], n_results=n_results, include=["documents", "metadatas"])
    docs = results.get("documents") or []
    if not docs or not docs[0]:
        return "No relevant messages found. Try rebuilding the index or rephrasing."
    context = "\n\n".join(docs[0])
    if len(context) > 28000:
        context = context[:28000] + "\n...[truncated]"

    from whatsapp_mcp.query import _get_client
    llm_client, cfg = _get_client(config)
    prompt = (
        "Use only the following WhatsApp chat excerpts to answer the question.\n\n"
        "--- Excerpts ---\n"
        f"{context}\n"
        "--- End ---\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    response = llm_client.chat(
        model=cfg.model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": cfg.temperature},
    )
    return (response.get("message", {}).get("content") or "").strip()


def semantic_query_with_index(
    collection_name: str,
    question: str,
    *,
    n_results: int = 10,
    persist_dir: str | Path | None = None,
    config: LlamaConfig | None = None,
) -> str:
    """Query an existing collection by name. Use after build_index."""
    client = get_chroma_client(persist_dir)
    embed_fn = OllamaChromaEmbeddingFunction(config=config)
    collection = client.get_collection(name=collection_name, embedding_function=embed_fn)
    results = collection.query(query_texts=[question], n_results=n_results, include=["documents"])
    docs = results.get("documents") or []
    if not docs or not docs[0]:
        return "No results."
    context = "\n\n".join(docs[0])
    if len(context) > 28000:
        context = context[:28000] + "\n...[truncated]"
    from whatsapp_mcp.query import _get_client
    llm_client, cfg = _get_client(config)
    prompt = (
        "Use only the following chat excerpts to answer the question.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    response = llm_client.chat(
        model=cfg.model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": cfg.temperature},
    )
    return (response.get("message", {}).get("content") or "").strip()
