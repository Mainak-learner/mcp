from __future__ import annotations
import os, re, json, pathlib, typing as T
from pypdf import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss  # faiss-cpu
from rank_bm25 import BM25Okapi

# --------- Config ---------
DATA_DIR = pathlib.Path(os.environ.get("PDF_RAG_DATA_DIR", "~/.mcp/pdf_rag")).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = DATA_DIR / "index.faiss"
META_PATH  = DATA_DIR / "meta.json"
EMBEDDER_NAME = os.environ.get("PDF_RAG_EMBEDDER", "BAAI/bge-small-en-v1.5")  # strong 384-d

# --------- Globals ---------
_model: SentenceTransformer | None = None
_index: faiss.IndexFlatIP | None = None
_meta: list[dict] = []

# --------- Utils ---------
def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _tok(s: str):
    return re.findall(r"[a-zA-Z0-9]+", s.lower())

def _normalize_scores(d: dict[int, float]) -> dict[int, float]:
    if not d: return {}
    vals = list(d.values()); lo, hi = min(vals), max(vals)
    if hi <= lo: return {k: 0.0 for k in d}
    rng = hi - lo
    return {k: (v - lo) / rng for k, v in d.items()}

def _load_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDER_NAME)
    return _model

def _load_or_init_index(dim: int | None = None):
    global _index, _meta
    if _index is None:
        if INDEX_PATH.exists() and META_PATH.exists():
            _index = faiss.read_index(str(INDEX_PATH))
            _meta = json.loads(META_PATH.read_text())
        else:
            _index = faiss.IndexFlatIP(dim or 384)
            _meta = []

def _persist_index():
    if _index is not None:
        faiss.write_index(_index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(_meta, ensure_ascii=False, indent=2))

def _first_page_text(pdf_path: str, max_chars=2000) -> str:
    try:
        r = PdfReader(pdf_path)
        t = (r.pages[0].extract_text() or "")
        return t[:max_chars]
    except Exception:
        return ""

# --------- Indexing ---------
def extract_text_from_pdf(path: pathlib.Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)

def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 150) -> list[str]:
    text = _normalize_ws(text)
    chunks, start, n = [], 0, len(text)
    step = max(1, chunk_chars - overlap)
    while start < n:
        end = min(n, start + chunk_chars)
        chunks.append(text[start:end]); start += step
    return chunks

def index_pdfs(paths: list[str], chunk_chars: int = 1200, overlap: int = 150) -> dict:
    if not paths:
        return {"ok": False, "message": "No paths provided."}
    emb = _load_embedder()
    _load_or_init_index(emb.get_sentence_embedding_dimension())
    added_chunks = 0; added_files = 0

    for p in paths:
        pp = pathlib.Path(p).expanduser().resolve()
        if not pp.exists(): continue
        pdfs = sorted(pp.rglob("*.pdf")) if pp.is_dir() else [pp]
        for pdf in pdfs:
            try:
                text = extract_text_from_pdf(pdf)
            except Exception:
                continue
            chunks = chunk_text(text, chunk_chars, overlap)
            if not chunks: continue
            embs = emb.encode(chunks, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
            global _index, _meta
            if _index is None:
                _index = faiss.IndexFlatIP(embs.shape[1])
            _index.add(embs)
            start_pos = 0
            for c in chunks:
                end_pos = start_pos + len(c)
                _meta.append({"text": c, "file": str(pdf), "start": start_pos, "end": end_pos})
                start_pos = end_pos - overlap
            added_chunks += len(chunks); added_files += 1

    _persist_index()
    return {"ok": True, "message": f"Indexed {added_chunks} chunks from {added_files} PDF(s). Store: {DATA_DIR}"}

def list_indexed_files() -> list[str]:
    _load_or_init_index()
    return sorted({os.path.basename(m["file"]) for m in _meta})

def clear_index() -> dict:
    global _index, _meta
    _index = None; _meta = []
    if INDEX_PATH.exists(): INDEX_PATH.unlink()
    if META_PATH.exists():  META_PATH.unlink()
    return {"ok": True, "message": "Cleared index."}

# --------- Retrieval (returns per-document context) ---------
def retrieve_context(
    question: str,
    top_k: int = 24,
    files: list[str] | None = None,
    max_ctx_chars: int = 9000,
    use_bm25: bool = True,
    w_dense: float = 0.6,
    w_bm25: float = 0.4,
) -> dict:
    if not question.strip():
        return {"ok": False, "message": "Empty question."}
    _load_or_init_index()
    if _index is None or len(_meta) == 0:
        return {"ok": False, "message": "Index empty. Run index_pdfs first."}

    # Target files (basenames) – default: all indexed
    targets = list({os.path.basename(m["file"]) for m in _meta}) if not files \
        else list({os.path.basename(f) for f in files})

    # Dense candidates
    emb = _load_embedder()
    q_emb = emb.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    dense_k = min(60, len(_meta))
    D, I = _index.search(q_emb, dense_k)
    dense = [(idx, _meta[idx], float(score)) for score, idx in zip(D[0].tolist(), I[0].tolist())]

    # BM25 candidates + fusion
    cand = dense
    if use_bm25:
        pool = []
        for idx, m in enumerate(_meta):
            if files and not any(f.lower() in m["file"].lower() for f in files):
                continue
            pool.append((idx, m, _tok(m["text"])))
        if pool:
            bm25 = BM25Okapi([t for _,_,t in pool])
            scores = bm25.get_scores(_tok(question))
            order = list(reversed(sorted(range(len(scores)), key=lambda i: scores[i])))[:min(60, len(scores))]
            bm25_list = [(pool[i][0], pool[i][1], float(scores[i])) for i in order]
            dmap = {i: s for i,_,s in dense}
            bmap = {i: s for i,_,s in bm25_list}
            dn = _normalize_scores(dmap); bn = _normalize_scores(bmap)
            keys = set(dn) | set(bn)
            fused = [(i, _meta[i], w_dense*dn.get(i,0.0) + w_bm25*bn.get(i,0.0)) for i in keys]
            fused.sort(key=lambda t: t[2], reverse=True)
            cand = fused

    # Ensure per-file coverage (round-robin)
    buckets = {t: [] for t in targets}; rest = []
    for idx, m, s in cand:
        bn = os.path.basename(m["file"])
        (buckets[bn] if bn in buckets else rest).append((idx, m, s))

    selected, seen = [], set()
    for t in targets:
        if buckets[t]:
            i,m,s = buckets[t].pop(0); selected.append(m); seen.add(i)
    while len(selected) < top_k and any(buckets.values()):
        for t in targets:
            if len(selected) >= top_k: break
            if buckets[t]:
                i,m,s = buckets[t].pop(0)
                if i not in seen: selected.append(m); seen.add(i)
    tail = []
    for t in targets: tail.extend(buckets[t])
    tail.extend(rest)
    for i,m,s in tail:
        if len(selected) >= top_k: break
        if i not in seen: selected.append(m); seen.add(i)

    # Build per-doc sections with optional front-matter, under budget
    per_doc = {t: {"filename": t, "front_matter": "", "snippets": []} for t in targets}
    total = 0
    for t in targets:
        path = next((mm["file"] for mm in _meta if os.path.basename(mm["file"]) == t), None)
        if path:
            fm = _first_page_text(path)
            if fm.strip():
                block = f"### {t}\n[front-matter]\n{fm}"
                if total + len(block) <= max_ctx_chars:
                    per_doc[t]["front_matter"] = fm
                    total += len(block)

    for m in selected:
        t = os.path.basename(m["file"])
        text = m["text"]; tag = f"[{t} {m['start']}–{m['end']}]"
        block_len = len(tag) + 1 + len(text)
        if total + block_len > max_ctx_chars: break
        per_doc[t]["snippets"].append({"start": m["start"], "end": m["end"], "text": text})
        total += block_len

    # Assemble markdown
    sections_md = []
    for t in targets:
        sec = per_doc.get(t)
        if not sec: continue
        blocks = [f"### {t}"]
        if sec["front_matter"]:
            blocks.append("[front-matter]\n" + sec["front_matter"])
        for sn in sec["snippets"]:
            blocks.append(f"[{t} {sn['start']}–{sn['end']}]\n{sn['text']}")
        if len(blocks) > 1:
            sections_md.append("\n".join(blocks))

    return {
        "ok": True,
        "target_files": targets,
        "sections": [per_doc[t] for t in targets],
        "context_md": "-----\n".join(sections_md),
        "counts": {"candidates": len(cand), "selected": sum(len(d['snippets']) for d in per_doc.values())}
    }
