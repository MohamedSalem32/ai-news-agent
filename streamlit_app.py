# streamlit_app.py
import os, json, math, re
from datetime import date, datetime
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Config ----------
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "data")
# default weights (α relevance, β recency, γ credibility)
ALPHA, BETA, GAMMA = 0.60, 0.30, 0.10

@dataclass
class Doc:
    id: str
    title: str
    text: str
    url: str
    source: str
    date: str
    language: str

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def parse_date(d: str) -> Optional[date]:
    if not d: return None
    for fmt in ("%Y-%m-%d","%d/%m/%Y","%Y/%m/%d","%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(d, fmt).date()
        except Exception:
            continue
    return None

def days_old(d: Optional[date], today: Optional[date] = None) -> Optional[int]:
    if not d: return None
    today = today or date.today()
    return max((today - d).days, 0)

def recency_score(date_str: str, today: Optional[date] = None) -> float:
    d = parse_date(date_str)
    if d is None:
        return math.exp(-365/4)  # treat missing as old
    do = days_old(d, today) or 0
    return math.exp(-do/4.0)

@st.cache_data(show_spinner=False)
def load_priors(data_dir: str) -> Dict[str, float]:
    p = os.path.join(data_dir, "credibility_prior.json")
    if not os.path.exists(p):
        return {"_default": 0.5}
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    prior = obj.get("prior", {})
    prior["_default"] = float(obj.get("default", 0.5))
    return prior

def credibility_score(source: str, priors: Dict[str, float]) -> float:
    return float(priors.get(source, priors.get("_default", 0.5)))

# Accept either corpus_for_demo.json OR raw_articles.json (your existing file)
@st.cache_data(show_spinner=False)
def load_docs(data_dir: str) -> List[Doc]:
    """
    Loads either data/corpus_for_demo.json or data/raw_articles.json
    and maps flexible keys to the Doc schema, ignoring extras like 'summary_ref'.
    """
    corpus_path = os.path.join(data_dir, "corpus_for_demo.json")
    raw_path    = os.path.join(data_dir, "raw_articles.json")

    def coerce(record, i):
        return Doc(
            id=str(record.get("id", i)),
            title=(record.get("title") or "").strip(),
            text=_norm(
                record.get("text")
                or record.get("content")
                or record.get("summary")
                or record.get("highlights")
                or ""
            ),
            url=record.get("url") or "",
            source=(record.get("source") or record.get("domain") or "").strip(),
            date=(record.get("date") or record.get("published") or record.get("publish_date") or "").strip(),
            language=(record.get("language") or record.get("lang") or "en").strip(),
        )

    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [coerce(r, i) for i, r in enumerate(raw)]

    with open(raw_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [coerce(r, i) for i, r in enumerate(raw)]


# ---------- Retriever (TF-IDF) ----------
@st.cache_resource(show_spinner=False)
def build_tfidf(docs: List[Doc]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [f"{d.title}. {d.text}" for d in docs]
    vec = TfidfVectorizer(
        lowercase=True, ngram_range=(1,2),
        max_features=40000, min_df=1, max_df=0.98,
        strip_accents="unicode", dtype=np.float32,
        token_pattern=r"(?u)\b\w\w+\b"
    )
    X = vec.fit_transform(texts)
    return vec, X

def query_tfidf(vec, X, docs: List[Doc], q: str, top_k: int = 30):
    Xq = vec.transform([q])
    sims = (Xq @ X.T).A.ravel()
    idx = np.argsort(-sims)[:top_k]
    return idx, sims[idx], Xq

def term_contributions(vec, Xq, Xi, top_n=8):
    try:
        q = Xq.toarray()[0]
        d = Xi.toarray()[0]
    except Exception:
        q = Xq.A[0]; d = Xi.A[0]
    contrib = q * d
    nz = np.nonzero(contrib)[0]
    pairs = [(vec.get_feature_names_out()[i], float(contrib[i])) for i in nz]
    pairs.sort(key=lambda x: -x[1])
    return pairs[:top_n]

# ---------- Summarization ----------
# Light extractive (CPU-safe). You can swap for your LLM later.
def summarize_40w(text: str) -> str:
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    S_SPLIT = re.compile(r"(?<=[\.!?…])\s+")
    sents = [s.strip() for s in S_SPLIT.split(text or "") if s.strip()]
    if not sents:
        return (text or "")[:280]
    try:
        X = TfidfVectorizer(ngram_range=(1,2), max_features=5000).fit_transform(sents)
        scores = X.max(axis=1).A.ravel()
        idx = np.argsort(-scores)[:2]
        chosen = [sents[i] for i in idx]
    except Exception:
        chosen = sents[:2]
    summary = " ".join(chosen)
    words = summary.split()
    return " ".join(words[:40]) if len(words) > 40 else summary

def focus_score(relevance: float, recency: float, credibility: float) -> float:
    return ALPHA*float(relevance) + BETA*float(recency) + GAMMA*float(credibility)

# ================== UI ==================
st.set_page_config(page_title="Fast Reading + XAI", layout="wide")
st.title("Fast Reading News — XAI Demo (Local)")

with st.sidebar:
    st.markdown("### Settings")
    data_dir = st.text_input("Data folder", value=DEFAULT_DATA_DIR)
    top_n = st.slider("Top results", 3, 20, 5, 1)
    show_digest = st.checkbox("Show digest (≤40 words)", value=True)
    st.markdown("---")
    st.markdown("**Scoring weights**")
    ALPHA_s = st.slider("Relevance α", 0.0, 1.0, ALPHA, 0.05)
    BETA_s  = st.slider("Recency β",   0.0, 1.0, BETA, 0.05)
    GAMMA_s = st.slider("Credibility γ",0.0, 1.0, GAMMA,0.05)
    total = ALPHA_s + BETA_s + GAMMA_s
    if abs(total-1.0) > 1e-6:
        st.warning("α+β+γ != 1 → normalized.")
        ALPHA_s, BETA_s, GAMMA_s = ALPHA_s/total, BETA_s/total, GAMMA_s/total

# override session weights
ALPHA, BETA, GAMMA = ALPHA_s, BETA_s, GAMMA_s

docs = load_docs(data_dir)
priors = load_priors(data_dir)
vec, X = build_tfidf(docs)

query = st.text_input("Search topic", value="AI regulation")
if not query.strip():
    st.stop()

idx, rels, Xq = query_tfidf(vec, X, docs, query, top_k=max(top_n*2, 30))

rows = []
for i, r in zip(idx, rels):
    d = docs[i]
    rec = recency_score(d.date)
    cred = credibility_score(d.source, priors)
    sc = focus_score(r, rec, cred)
    rows.append((i, d, r, rec, cred, sc))

rows.sort(key=lambda t: -t[5])
rows = rows[:top_n]

colL, colR = st.columns([2,1], gap="large")

with colL:
    st.subheader("Ranked results")
    for rank, (i, d, r, rec, cred, sc) in enumerate(rows, start=1):
        with st.container(border=True):
            st.markdown(f"**{rank}. {d.title}**")
            meta = f"{d.source} — {d.date or 'n/a'}"
            if d.url: meta += f" — [link]({d.url})"
            st.caption(meta)

            bdf = pd.DataFrame({
                "component": ["relevance","recency","credibility"],
                "value": [r, rec, cred],
                "weight": [ALPHA, BETA, GAMMA],
                "contribution": [ALPHA*r, BETA*rec, GAMMA*cred],
            })
            st.dataframe(bdf, use_container_width=True, hide_index=True)
            st.markdown(f"**Final score:** {sc:.4f}")

            Xi = X[i]
            contribs = term_contributions(vec, Xq, Xi, top_n=8)
            if contribs:
                cdf = pd.DataFrame(contribs, columns=["term","contribution"])
                st.bar_chart(cdf.set_index("term"))

            if show_digest:
                with st.expander("Digest (≤40 words)"):
                    st.write(summarize_40w(d.text))

            with st.expander("Preview (first 600 chars)"):
                st.write(d.text[:600] + ("…" if len(d.text) > 600 else ""))

with colR:
    st.subheader("Why this ranking?")
    st.markdown("""
- **Score** = α·relevance + β·recency + γ·credibility  
- **Relevance** = TF-IDF cosine similarity between your query and article.  
- **Recency** = exp(−days_old/4): fresher → higher.  
- **Credibility** = prior per source from `credibility_prior.json` (default 0.5).  
- **Term attributions** = top shared TF-IDF terms driving similarity.
    """)
