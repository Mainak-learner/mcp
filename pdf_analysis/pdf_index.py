
import os
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks

class PdfIndex:
    """
    Simple on-disk vector index for a collection of PDFs.
    Saves: index.faiss, meta.json, embeddings.npy
    """
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.index = None
        self.meta: List[Dict[str, Any]] = []
        self.emb_path = self.base_dir / "embeddings.npy"
        self.faiss_path = self.base_dir / "index.faiss"
        self.meta_path = self.base_dir / "meta.json"

    def _load_model(self):
        if self.model is None:
            # lightweight, widely available model
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def build_from_pdfs(self, pdf_paths: List[str]):
        texts = []
        metas = []
        for path in pdf_paths:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"PDF not found: {path}")
            reader = PdfReader(str(p))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                for chunk in _chunk_text(text):
                    metas.append({"source": str(p), "page": i+1, "chunk": chunk[:200]})
                    texts.append(chunk)
        if not texts:
            raise ValueError("No text extracted from the provided PDFs")

        self._load_model()
        embeds = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        dim = embeds.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeds)
        self.index = index
        self.meta = metas
        np.save(self.emb_path, embeds)
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f, indent=2)
        faiss.write_index(self.index, str(self.faiss_path))

    def load(self):
        if self.faiss_path.exists():
            self.index = faiss.read_index(str(self.faiss_path))
        if self.meta_path.exists():
            self.meta = json.load(open(self.meta_path))

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        if self.index is None:
            self.load()
        self._load_model()
        q = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        D, I = self.index.search(q, k)
        results = []
        for idx, score in zip(I[0].tolist(), D[0].tolist()):
            if idx == -1:
                continue
            results.append((idx, float(score)))
        return results

    def get_chunks(self, indices: List[int]) -> List[Dict[str, Any]]:
        out = []
        for i in indices:
            if 0 <= i < len(self.meta):
                meta = self.meta[i]
                out.append(meta)
        return out
