# Fast Reading (Digest & Focus)

> **Context:** school team project (Advanced Deep Learning, ESPRIT). Each team member built one news agent; this repository is my agent, **AI-News-Agent-V**. The team repository is [Aethelios/ai-news-agent](https://github.com/Aethelios/ai-news-agent).

CPU-friendly local news assistant with TF-IDF retrieval (default), cosine similarity, deterministic recency, and source credibility blending. Summaries try `sshleifer/distilbart-cnn-12-6` **if available**; otherwise a tiny extractive TF-IDF fallback (2 sentences, ≤ 40 words).

## Project Layout

```
AI-news-agent/
├─ app/
│  ├─ __init__.py
│  ├─ retriever.py          # TF-IDF (and optional SentenceTransformers) + cosine
│  ├─ scoring.py            # recency, credibility, αβγ blend, focus_score()
│  ├─ summarize.py          # abstractive → extractive fallback
│  ├─ tools.py              # tools: focus(topic, top_n), digest(topic, n)
│  ├─ agent.py              # ReAct-style agent (fallback CLI if LC not installed)
│  └─ api.py                # FastAPI endpoints (/focus, /digest)
├─ data/
│  ├─ corpus_for_demo.json
│  ├─ credibility_prior.json
│  └─ eval_pairs.json
├─ demo.py                  # small demo + acceptance + latency + optional ROUGE
└─ notebook-fast-reading-digest-focus.ipynb  # (optional)
```

> Place your three JSON files in `./data/` as specified.

## Install (CPU-only, minimal)

> **Windows PowerShell** example; on Linux/macOS replace the activation line accordingly.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install scikit-learn numpy fastapi uvicorn
# Optional (LangChain agent UX):
pip install "langchain>=0.2" "langchain-core>=0.2" "langchain-openai>=0.1"
# Optional abstractive summarization (may download a small model):
pip install transformers
# Optional semantic retriever:
pip install sentence-transformers
# Optional eval:
pip install rouge-score pandas tabulate
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install scikit-learn numpy fastapi uvicorn
# optional extras same as above
```

> If you prefer a single file, add a `requirements.txt` and run:  
> `pip install -r requirements.txt`

## Run the demo (prints acceptance results)

```powershell
python demo.py
```

You should see:

- `focus("AI in healthcare", top_n=3)` → items with fields  
  `title, source, date, url, relevance, recency, credibility, score`
- `digest("économie française", n=6)` → bullets (≤ 40 words)
- A small latency table
- Optional ROUGE sanity (skips if `rouge_score` not installed)

## Use the simple CLI agent (fallback mode)

```powershell
python -m app.agent
```

Then type:

```
> focus AI in healthcare 3
> digest économie française 6
> quit
```

If LangChain and an LLM are configured, it builds a ReAct agent; otherwise it runs the fallback CLI that calls `focus`/`digest` directly.

## Start the REST API (optional)

```powershell
uvicorn app.api:app --reload
```

Example requests:

```powershell
# Focus
curl -X POST http://127.0.0.1:8000/focus -H "Content-Type: application/json" ^
     -d "{\"query\":\"AI in healthcare\",\"top_n\":3}"

# Digest
curl -X POST http://127.0.0.1:8000/digest -H "Content-Type: application/json" ^
     -d "{\"query\":\"économie française\",\"n\":6}"
```

(Linux/macOS remove the PowerShell caret `^` line continuations.)

## How scoring works

- **Relevance:** cosine similarity of query vs TF-IDF doc vectors (title + text).  
- **Recency:** `exp(-days_old/4)` with fixed `today = 2020-12-31` (deterministic).  
- **Credibility:** lookup in `credibility_prior.json`, else default = `0.5`.  
- **Focus score:** `0.60*relevance + 0.30*recency + 0.10*credibility`.

## Notes & tips

- CPU-only by default. `transformers` and `sentence-transformers` are optional.
- If `transformers` isn’t installed, summarization falls back automatically.
- If you get import-path issues, ensure `app/__init__.py` exists.
- Large JSONs are read into memory once; for huge corpora consider chunking.
- Determinism: aside from floating-point ties, behavior is deterministic.

## Quick sanity checklist

- [ ] `app/__init__.py` exists  
- [ ] `data/` contains the three JSON files  
- [ ] `pip install scikit-learn numpy` succeeded  
- [ ] `python demo.py` runs without errors
