# app/tools.py
'''
from __future__ import annotations
from typing import List, Dict, Any

from .retriever import load_corpus, st_is_available, OptionalSentenceTransformerRetriever
from .scoring import load_credibility, recency_score, credibility_score, focus_score
from .summarize import summarize_40w

DEFAULT_TOPK = 20

class FastReaderCore:
    def __init__(self, data_dir: str, use_st: bool | str = "auto", st_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        This version *forces* Sentence-Transformers. Any attempt to run without
        sentence-transformers installed will raise a clear error.
        Arguments like `use_st` are kept only for call-site compatibility and ignored.
        """
        if not st_is_available():
            raise RuntimeError(
                "Sentence-Transformers is required for retrieval. "
                "Install with: pip install sentence-transformers torch"
            )
        self.data_dir = data_dir
        self.docs = load_corpus(data_dir)
        self.priors = load_credibility(data_dir)  # used by focus()
        self.retriever = OptionalSentenceTransformerRetriever(self.docs, model_name=st_model)
        self.backend = "sentence-transformers"

    def _retrieve(self, query: str, top_k: int = DEFAULT_TOPK):
        return self.retriever.query(query, top_k=top_k)

    def focus(self, topic: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Return top_n re-ranked by focus score with components.
        Fields: title, source, date, url, relevance, recency, credibility, score, backend
        """
        idx, rel = self._retrieve(topic, top_k=DEFAULT_TOPK)
        items = []
        for i, r in zip(idx, rel):
            d = self.docs[i]
            rec = recency_score(d.date)
            cred = credibility_score(d.source, self.priors)
            sc = focus_score(float(r), float(rec), float(cred))
            items.append({
                "title": d.title,
                "source": d.source,
                "date": d.date,
                "url": d.url,
                "relevance": float(r),
                "recency": float(rec),
                "credibility": float(cred),
                "score": float(sc),
                "backend": self.backend,
            })
        items.sort(key=lambda x: -x["score"])
        return items[:top_n]

    def digest(self, topic: str, n: int = 5) -> List[Dict[str, Any]]:
        """
        Return n short bullets (<=40 words) from best-matching articles.
        Fields: title, source, date (if any), url, bullet
        """
        ranked = self.focus(topic, top_n=max(n, 2*n))
        out, seen = [], set()
        for it in ranked:
            key = (it["title"], it["source"])
            if key in seen:
                continue
            seen.add(key)
            # find the doc again (same dataset, so match by title+source is fine)
            doc = next((d for d in self.docs if d.title == it["title"] and d.source == it["source"]), None)
            if not doc:
                continue
            bullet = summarize_40w(doc.text)
            out.append({
                "title": it["title"],
                "source": it["source"],
                "date": it["date"],
                "url": it["url"],
                "bullet": bullet
            })
            if len(out) >= n:
                break
        return out

# -------- LangChain tool wrappers (return JSON-serializable) --------
def build_tools(data_dir: str, use_st: bool | str = "auto", st_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Kept signature-compatible with previous code, but now always builds ST-based tools.
    """
    core = FastReaderCore(data_dir, use_st=True, st_model=st_model)

    # Try modern import path first, then legacy.
    try:
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
    except Exception:
        try:
            from langchain.tools import StructuredTool  # legacy path
            from pydantic import BaseModel, Field
        except Exception:
            # Return plain callables; agent will run in fallback CLI
            return {"focus": core.focus, "digest": core.digest, "_core": core, "_lc_tools": None}

    class FocusInput(BaseModel):
        topic: str = Field(..., description="Search topic or query string.")
        top_n: int = Field(5, description="Number of results.")

    class DigestInput(BaseModel):
        topic: str = Field(..., description="Search topic or query string.")
        n: int = Field(5, description="Number of bullets to return.")

    focus_tool = StructuredTool.from_function(
        name="focus",
        description="Find and re-rank news for a topic using relevance, recency, and credibility.",
        func=core.focus,
        args_schema=FocusInput,
        return_direct=False,
    )

    digest_tool = StructuredTool.from_function(
        name="digest",
        description="Return <=40-word bullets for top matching news articles on a topic.",
        func=core.digest,
        args_schema=DigestInput,
        return_direct=False,
    )

    return {
        "focus": core.focus,
        "digest": core.digest,
        "_core": core,
        "_lc_tools": [focus_tool, digest_tool],
    }
'''

# app/tools.py
from __future__ import annotations
from typing import List, Dict, Any

from .retriever import load_corpus, st_is_available, OptionalSentenceTransformerRetriever
from .scoring import load_credibility, recency_score, credibility_score, focus_score
from .summarize import summarize_40w  # This import is already correct and in place

DEFAULT_TOPK = 20

class FastReaderCore:
    def __init__(self, data_dir: str, use_st: bool | str = "auto", st_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        This version *forces* Sentence-Transformers. Any attempt to run without
        sentence-transformers installed will raise a clear error.
        Arguments like `use_st` are kept only for call-site compatibility and ignored.
        """
        if not st_is_available():
            raise RuntimeError(
                "Sentence-Transformers is required for retrieval. "
                "Install with: pip install sentence-transformers torch"
            )
        self.data_dir = data_dir
        self.docs = load_corpus(data_dir)
        self.priors = load_credibility(data_dir)  # used by focus()
        self.retriever = OptionalSentenceTransformerRetriever(self.docs, model_name=st_model)
        self.backend = "sentence-transformers"

    def _retrieve(self, query: str, top_k: int = DEFAULT_TOPK):
        return self.retriever.query(query, top_k=top_k)

    def focus(self, topic: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Return top_n re-ranked by focus score with components.
        Fields: title, source, date, url, relevance, recency, credibility, score, backend
        """
        idx, rel = self._retrieve(topic, top_k=DEFAULT_TOPK)
        items = []
        for i, r in zip(idx, rel):
            d = self.docs[i]
            rec = recency_score(d.date)
            cred = credibility_score(d.source, self.priors)
            sc = focus_score(float(r), float(rec), float(cred))
            items.append({
                "title": d.title,
                "source": d.source,
                "date": d.date,
                "url": d.url,
                "relevance": float(r),
                "recency": float(rec),
                "credibility": float(cred),
                "score": float(sc),
                "backend": self.backend,
                # Include the raw text and language to make summarization easier in digest()
                "text": d.text,
                "language": d.language,
            })
        items.sort(key=lambda x: -x["score"])
        return items[:top_n]

    def digest(self, topic: str, n: int = 5) -> List[Dict[str, Any]]:
        """
        Returns n short summaries (<=40 words) from best-matching articles.
        
        It calls focus() for the ranked articles and then summarizes them.
        Fields: title, source, date (if any), url, summary_40w
        """
        # Call the existing focus tool to get the top n ranked articles
        # NOTE: focus() must be updated to include 'text' and 'language' fields (see above change)
        top = self.focus(topic, top_n=n)
        out = []
        
        for r in top:
            # Get language and text from the result dictionary (included by updated focus)
            lang = (r.get("language") or "en").lower()
            text = r.get("text", "")
            
            # Summarize the text using the imported summarize_40w function
            summary = summarize_40w(text, lang=lang)
            
            # Create a new dictionary record, copying existing fields
            rec = {
                "title": r.get("title"),
                "source": r.get("source"),
                "date": r.get("date"),
                "url": r.get("url"),
                "summary_40w": summary,
            }
            out.append(rec)
            
        return out

# -------- LangChain tool wrappers (return JSON-serializable) --------
def build_tools(data_dir: str, use_st: bool | str = "auto", st_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Kept signature-compatible with previous code, but now always builds ST-based tools.
    """
    core = FastReaderCore(data_dir, use_st=True, st_model=st_model)

    # Try modern import path first, then legacy.
    try:
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
    except Exception:
        try:
            from langchain.tools import StructuredTool  # legacy path
            from pydantic import BaseModel, Field
        except Exception:
            # Return plain callables; agent will run in fallback CLI
            return {"focus": core.focus, "digest": core.digest, "_core": core, "_lc_tools": None}

    class FocusInput(BaseModel):
        topic: str = Field(..., description="Search topic or query string.")
        top_n: int = Field(5, description="Number of results.")

    class DigestInput(BaseModel):
        topic: str = Field(..., description="Search topic or query string.")
        n: int = Field(5, description="Number of bullets to return.")

    focus_tool = StructuredTool.from_function(
        name="focus",
        description="Find and re-rank news for a topic using relevance, recency, and credibility.",
        func=core.focus,
        args_schema=FocusInput,
        return_direct=False,
    )

    digest_tool = StructuredTool.from_function(
        name="digest",
        description="Return <=40-word bullets for top matching news articles on a topic.",
        func=core.digest,
        args_schema=DigestInput,
        return_direct=False,
    )

    return {
        "focus": core.focus,
        "digest": core.digest,
        "_core": core,
        "_lc_tools": [focus_tool, digest_tool],
    }