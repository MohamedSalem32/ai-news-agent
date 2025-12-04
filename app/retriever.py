'''# app/retriever.py
from __future__ import annotations
import json, os, re
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional, Tuple

import numpy as np

TODAY_FIXED = date(2020, 12, 31)  # deterministic reference date for recency helpers

# ----------------------- Public helpers & data model -----------------------

import re
_WS = re.compile("\\s+")

def _normalize_text(s: str) -> str:
    return _WS.sub(" ", (s or "")).strip()


def st_is_available() -> bool:
    """
    Return True if sentence-transformers can be imported.
    """
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False

# --- dataclass (keep summary_ref) ---
from dataclasses import dataclass

@dataclass
class Doc:
    id: str
    title: str
    text: str
    url: str = ""
    source: str = ""
    date: str = ""
    language: str = "en"
    summary_ref: str = ""   # keep this

# --- loader (explicit construction) ---
def load_corpus(data_dir: str) -> List[Doc]:
    p = os.path.join(data_dir, "corpus_for_demo.json")
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)

    docs: List[Doc] = []
    for r in raw:
        docs.append(Doc(
            id=r.get("id", ""),
            title=r.get("title", ""),
            text=_normalize_text(r.get("text", "")),
            url=r.get("url", ""),
            source=r.get("source", ""),
            date=r.get("date", ""),
            language=r.get("language", "en"),
            summary_ref=r.get("summary_ref", ""),
        ))
    return docs


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def parse_date(d: str) -> Optional[date]:
    d = (d or "").strip()
    if not d:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(d, fmt).date()
        except Exception:
            continue
    return None

def days_old(d: Optional[date]) -> Optional[int]:
    if not d:
        return None
    delta = TODAY_FIXED - d
    return max(delta.days, 0)

# -------------------------- Sentence-Transformers --------------------------

class _BaseRetriever:
    def query(self, q: str, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

class OptionalSentenceTransformerRetriever(_BaseRetriever):
    """
    Sentence-Transformers retriever (required). No TF-IDF fallback is provided.
    Normalizes embeddings and uses dot-product similarity.
    """
    def __init__(self, docs: List[Doc], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(
                "Sentence-Transformers is required but not installed. "
                "Please run: pip install sentence-transformers torch"
            ) from e

        self.docs = docs
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        texts = [f"{d.title}. {d.text}" for d in docs]
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        # encode(..., normalize_embeddings=True) returns L2-normalized vectors already
        self.doc_emb = emb.astype(np.float32, copy=False)

    def query(self, q: str, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        qv = self.model.encode([q], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        sims = (qv @ self.doc_emb.T).ravel()  # dot product == cosine due to normalization
        idx = np.argsort(-sims)[:top_k]
        return idx, sims[idx]

# ------------------------ Builder (no more fallbacks) -----------------------

def build_best_retriever(docs: List[Doc], prefer_st: bool = True) -> _BaseRetriever:
    """
    Kept for compatibility with callers, but now always returns the
    Sentence-Transformers retriever (no TF-IDF branch).
    """
    return OptionalSentenceTransformerRetriever(docs)
'''

# app/retriever.py
from __future__ import annotations
import json, os, re
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional, Tuple

import numpy as np

# ----------------------- Public helpers & data model -----------------------
TODAY_FIXED = date.today()

import re
_WS = re.compile("\\s+")

def _normalize_text(s: str) -> str:
    # Use re.sub for consistent normalization as used later in the file
    return re.sub(r"\s+", " ", (s or "")).strip()


def st_is_available() -> bool:
    """
    Return True if sentence-transformers can be imported.
    """
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False

# --- dataclass (keep summary_ref) ---
from dataclasses import dataclass

@dataclass
class Doc:
    id: str
    title: str
    text: str
    url: str = ""
    source: str = ""
    date: str = ""
    language: str = "en"
    summary_ref: str = ""   # keep this

# -------------------- Date Parsing and Recency Helpers --------------------

def parse_date(d: str) -> Optional[date]:
    """
    Parses a date string into a datetime.date object. More tolerant of formats.
    """
    s = (d or "").strip()
    if not s: return None

    # quick ISO-8601 path (e.g., 2024-05-15)
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        try: return datetime.strptime(s[:10], "%Y-%m-%d").date()
        except Exception: pass

    # Try common formats
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%SZ", "%Y.%m.%d"):
        try: return datetime.strptime(s, fmt).date()
        except Exception: continue

    return None

def days_old(d: Optional[date]) -> Optional[int]:
    """
    Calculates the number of days between today and the given date.
    Returns 0 for future dates.
    """
    if not d: return None
    return max((date.today() - d).days, 0)

# --- loader (explicit construction) ---
def load_corpus(data_dir: str) -> List[Doc]:
    """
    Loads the corpus from a file, prioritizing CORPUS_PATH env var, 
    then live_corpus.json, then corpus_for_demo.json.
    """
    env_path = os.getenv("CORPUS_PATH", "").strip()
    candidates = []
    if env_path and os.path.isfile(env_path):
        candidates.append(env_path)
    candidates.append(os.path.join(data_dir, "live_corpus.json"))
    candidates.append(os.path.join(data_dir, "corpus_for_demo.json"))

    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError(
            "No corpus file found. Checked: CORPUS_PATH environment variable, "
            f"{os.path.join(data_dir, 'live_corpus.json')}, and "
            f"{os.path.join(data_dir, 'corpus_for_demo.json')}."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    docs: List[Doc] = []
    for r in raw:
        docs.append(Doc(
            id=r.get("id", ""),
            title=r.get("title", ""),
            text=_normalize_text(r.get("text", "")),
            url=r.get("url", ""),
            source=r.get("source", ""),
            date=r.get("date", ""),
            language=r.get("language", "en"),
            summary_ref=r.get("summary_ref", ""),
        ))
    return docs


# -------------------------- Sentence-Transformers --------------------------

class _BaseRetriever:
    def query(self, q: str, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

class OptionalSentenceTransformerRetriever(_BaseRetriever):
    """
    Sentence-Transformers retriever (required). No TF-IDF fallback is provided.
    Normalizes embeddings and uses dot-product similarity.
    """
    def __init__(self, docs: List[Doc], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(
                "Sentence-Transformers is required but not installed. "
                "Please run: pip install sentence-transformers torch"
            ) from e

        self.docs = docs
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        texts = [f"{d.title}. {d.text}" for d in docs]
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        # encode(..., normalize_embeddings=True) returns L2-normalized vectors already
        self.doc_emb = emb.astype(np.float32, copy=False)

    def query(self, q: str, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        qv = self.model.encode([q], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        sims = (qv @ self.doc_emb.T).ravel()  # dot product == cosine due to normalization
        idx = np.argsort(-sims)[:top_k]
        return idx, sims[idx]

# ------------------------ Builder (no more fallbacks) -----------------------

def build_best_retriever(docs: List[Doc], prefer_st: Optional[bool] = True) -> _BaseRetriever:
    """
    Kept for compatibility with callers, but now always returns the
    Sentence-Transformers retriever (no TF-IDF branch).
    """
    return OptionalSentenceTransformerRetriever(docs)