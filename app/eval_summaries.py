import json, os, re, argparse, math
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

# ---- Optional deps: install if you can ----
# pip install rouge-score scikit-learn pandas numpy
try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except Exception:
    HAS_ROUGE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SK = True
except Exception:
    HAS_SK = False

# ---------- Utils ----------
WS = re.compile(r"\w+[\w'-]*", re.UNICODE)

def tokenize_words(text):
    return [t.lower() for t in WS.findall(text or "")]

def bigrams(tokens):
    return list(zip(tokens, tokens[1:]))

def word_count(text): return len(tokenize_words(text))

def cap_40_words(text):
    toks = tokenize_words(text)
    if len(toks) <= 40:
        return text
    # rebuild with the first 40 tokens preserving rough spacing
    # (good enough for length constraint auditing)
    return " ".join(toks[:40])

def sentences(text):
    # very light splitter to avoid heavy deps
    return re.split(r"(?<=[\.\!\?])\s+", text.strip())

# ---------- Try to import your project summarizer ----------
def load_project_summarizer():
    try:
        import sys
        sys.path.append("/mnt/data")  # adjust if needed
        import summarize  # expects summarize.summarize_40w(text, lang=None)
        if hasattr(summarize, "summarize_40w"):
            def _summ(text, lang=None):
                out = summarize.summarize_40w(text, lang=lang)
                return cap_40_words(out)
            return _summ
    except Exception:
        pass
    return None

# ---------- Fallback extractive 40-word summarizer ----------
def tfidf_extractive_40(text, lang=None, max_words=40):
    sents = [s.strip() for s in sentences(text) if s.strip()]
    if not sents:
        return ""
    if not HAS_SK:
        # no sklearn: take first sentence(s) until 40 words
        out = []
        total = 0
        for s in sents:
            wc = word_count(s)
            if total + wc <= max_words:
                out.append(s)
                total += wc
            else:
                break
        return cap_40_words(" ".join(out) if out else sents[0])

    # With sklearn: rank sentences by cosine to doc centroid
    docs = sents + [text]
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), lowercase=True)
    X = vec.fit_transform(docs)
    doc_vec = X[-1]
    sent_vecs = X[:-1]
    sims = cosine_similarity(sent_vecs, doc_vec).ravel()
    order = np.argsort(-sims)
    out, total = [], 0
    for idx in order:
        s = sents[idx]
        wc = word_count(s)
        if total + wc <= max_words or not out:
            out.append(s)
            total += wc
        if total >= max_words: break
    return cap_40_words(" ".join(out))

# choose summarizer
PROJECT_SUMM = load_project_summarizer()
def summarize_40(text, lang=None):
    if PROJECT_SUMM:
        return PROJECT_SUMM(text, lang=lang)
    return tfidf_extractive_40(text, lang=lang, max_words=40)

# ---------- Metrics ----------
def rouge_scores(pred, ref):
    if not HAS_ROUGE:
        return {"rouge1": np.nan, "rouge2": np.nan, "rougeL": np.nan}
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeLsum"], use_stemmer=True)
    s = scorer.score(ref or "", pred or "")
    return {
        "rouge1": s["rouge1"].fmeasure,
        "rouge2": s["rouge2"].fmeasure,
        "rougeL": s["rougeLsum"].fmeasure,
    }

def tfidf_cosine(src, summ):
    if not HAS_SK: return np.nan
    vec = TfidfVectorizer(max_features=8000, ngram_range=(1,2), lowercase=True)
    X = vec.fit_transform([src or "", summ or ""])
    return cosine_similarity(X[0], X[1])[0,0]

def coverage_ratio(src, summ):
    src_toks = set(tokenize_words(src))
    summ_toks = tokenize_words(summ)
    if not summ_toks: return 0.0
    covered = sum(1 for t in summ_toks if t in src_toks)
    return covered / len(summ_toks)

def novel_bigram_ratio(src, summ):
    src_bi = set(bigrams(tokenize_words(src)))
    summ_bi = list(bigrams(tokenize_words(summ)))
    if not summ_bi: return 0.0
    nov = sum(1 for bg in summ_bi if bg not in src_bi)
    return nov / len(summ_bi)

# ---------- Data loading ----------
def load_json_any(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # supports list OR dict with "items"
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    return data

def build_eval_set(corpus_path, eval_pairs_path):
    items = []

    # 1) corpus items that already have summary_ref
    corpus = load_json_any(corpus_path)
    for d in corpus:
        if d.get("summary_ref"):
            items.append({
                "id": d.get("id"),
                "language": d.get("language"),
                "title": d.get("title"),
                "source": d.get("source"),
                "text": d.get("text"),
                "summary_ref": d.get("summary_ref"),
                "from": "corpus"
            })

    # 2) eval_pairs (guaranteed summary_ref)
    evals = load_json_any(eval_pairs_path)
    for e in evals:
        items.append({
            "id": e.get("id"),
            "language": e.get("language"),
            "title": e.get("title"),
            "source": e.get("source"),
            "text": e.get("text"),
            "summary_ref": e.get("summary_ref"),
            "from": "eval_pairs"
        })

    # de-dup by id+source if present
    seen = set()
    dedup = []
    for it in items:
        key = (it.get("id"), it.get("source"))
        if key in seen: continue
        seen.add(key)
        dedup.append(it)
    return dedup

# ---------- Main ----------
def main(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    eval_items = build_eval_set(args.corpus, args.eval)

    rows = []
    for it in eval_items:
        lang = it.get("language")
        src = it.get("text") or ""
        ref = it.get("summary_ref") or ""
        pred = summarize_40(src, lang=lang)
        pred = cap_40_words(pred)  # enforce constraint

        r = rouge_scores(pred, ref)
        wc = word_count(pred)
        row = {
            "id": it.get("id"),
            "from": it.get("from"),
            "language": lang,
            "title": it.get("title"),
            "source": it.get("source"),
            "wc_pred": wc,
            "len_ok_<=40": wc <= 40,
            "rouge1_f": r["rouge1"],
            "rouge2_f": r["rouge2"],
            "rougeL_f": r["rougeL"],
            "tfidf_cosine": tfidf_cosine(src, pred),
            "coverage_ratio": coverage_ratio(src, pred),
            "novel_bigram_ratio": novel_bigram_ratio(src, pred),
            "pred_summary": pred,
            "ref_summary": ref
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8")

    def agg_report(sub, name):
        if sub.empty: return
        print(f"\n== {name} ({len(sub)} items) ==")
        for m in ["rouge1_f","rouge2_f","rougeL_f","tfidf_cosine","coverage_ratio","novel_bigram_ratio"]:
            print(f"{m}: {np.nanmean(sub[m].values):.4f}")
        print(f"len_ok rate (<=40): {sub['len_ok_<=40'].mean():.3f}")
        print(f"avg words: {sub['wc_pred'].mean():.2f}")

    agg_report(df, "ALL")
    for lg, sub in df.groupby("language"):
        agg_report(sub, f"LANG={lg}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="corpus_for_demo.json")
    p.add_argument("--eval", default="eval_pairs.json")
    p.add_argument("--out", default="summary_eval.csv")
    args = p.parse_args()
    main(args)
