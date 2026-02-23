"""
WhatsApp MCP: query WhatsApp export (text, photo, audio, video) using local Llama only.

Export: ZIP or folder with chat.txt and optional Media/. Photos are queried via
Llama vision; audio/video are listed by metadata (no transcription).
ChromaDB semantic search (Ollama nomic-embed-text) is optional for better retrieval.
"""

from whatsapp_mcp.config import LlamaConfig
from whatsapp_mcp.parser import load_export, list_media, parse_chat
from whatsapp_mcp.query import query_all, query_chat, query_image, describe_photos
from whatsapp_mcp.chroma_search import build_index, semantic_query, get_chroma_client

__all__ = [
    "LlamaConfig",
    "load_export",
    "parse_chat",
    "list_media",
    "query_chat",
    "query_image",
    "query_all",
    "describe_photos",
    "build_index",
    "semantic_query",
    "get_chroma_client",
]
