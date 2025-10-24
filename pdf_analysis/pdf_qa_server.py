# pdf_qa_server.py
from __future__ import annotations
import os, json
from mcp.server.fastmcp import FastMCP
import retrieval_core as core

# Safety for transformers
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

mcp = FastMCP("PDF-QA-RAG", version="1.0.0", description="Index PDFs and retrieve per-document context for QA")

@mcp.tool()
def index_pdfs(paths: list[str], chunk_chars: int = 1200, overlap: int = 150) -> str:
    """Index one or more PDFs or directories of PDFs."""
    return json.dumps(core.index_pdfs(paths, chunk_chars=chunk_chars, overlap=overlap), ensure_ascii=False)

@mcp.tool()
def retrieve_context(question: str, top_k: int = 24, files: list[str] | None = None, max_ctx_chars: int = 9000) -> str:
    """
    Retrieve per-document context (front-matter + snippets) WITHOUT calling an LLM.
    Returns JSON with sections and a ready-to-drop 'context_md'.
    """
    return json.dumps(core.retrieve_context(question, top_k=top_k, files=files, max_ctx_chars=max_ctx_chars), ensure_ascii=False)

@mcp.tool()
def list_indexed_files() -> str:
    """Return basenames of all indexed PDFs."""
    return json.dumps({"files": core.list_indexed_files()}, ensure_ascii=False)

@mcp.tool()
def clear_index() -> str:
    """Delete FAISS index + metadata."""
    return json.dumps(core.clear_index(), ensure_ascii=False)

if __name__ == "__main__":
    mcp.run()
