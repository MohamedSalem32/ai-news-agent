# app/fetch_news.py

import os, json, hashlib, urllib.parse, datetime as dt, sqlite3
from typing import List, Dict, Optional
import requests, feedparser, trafilatura
from dateutil.parser import parse as parse_dt
from langdetect import detect, DetectorFactory

# Use deterministic language detection
DetectorFactory.seed = 0

# --- Import your DistilBART wrapper for 40-word summaries ---
# If summarize.py is not in the 'app' package, adjust this import.
from .summarize import summarize_40w

# ============================================================
# 1) Generic Helpers
# ============================================================

def now_utc() -> dt.datetime:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

def to_iso(d) -> Optional[str]:
    """Convert various date representations to UTC ISO string."""
    if not d:
        return None
    if isinstance(d, str):
        try:
            d = parse_dt(d)
        except Exception:
            return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc).isoformat()

def clean_url(u: str) -> str:
    """Strip UTM parameters and fragments from URLs."""
    if not u:
        return u
    p = urllib.parse.urlsplit(u)
    q = urllib.parse.parse_qsl(p.query)
    q = [(k, v) for (k, v) in q if not k.lower().startswith("utm_")]
    return urllib.parse.urlunsplit(
        (p.scheme, p.netloc, p.path, urllib.parse.urlencode(q), "")
    )

def sha(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def extract_text(url: str, html: Optional[str] = None) -> str:
    """Extract main article text from a URL using trafilatura."""
    try:
        downloaded = html or trafilatura.fetch_url(url)
        return trafilatura.extract(downloaded, include_comments=False) or ""
    except Exception:
        return ""

def detect_lang(title: str, text: str) -> Optional[str]:
    """Detect language from text (or title if text is too short)."""
    txt = (text or "")[:1000] or (title or "")
    if not txt.strip():
        return None
    try:
        return detect(txt)
    except Exception:
        return None

# ============================================================
# 2) News Providers
# ============================================================

DEFAULT_RSS = [
    # general/world
    "https://www.reuters.com/world/rss",
    "https://www.bbc.co.uk/news/world/rss",
    "https://www.theguardian.com/world/rss",
    # sports
    "https://www.espn.com/espn/rss/news",
]

def fetch_rss(urls: List[str], limit: int = 50) -> List[Dict]:
    out: List[Dict] = []
    for u in urls:
        f = feedparser.parse(u)
        for e in f.entries[:limit]:
            url = clean_url(e.get("link"))
            title = e.get("title")
            d = e.get("published") or e.get("updated") or e.get("created")
            iso = to_iso(d)
            out.append({
                "title": title,
                "url": url,
                "source": urllib.parse.urlsplit(url).netloc if url else "",
                "date": iso,
                "text": "",  # fill later
            })
    return out

def fetch_newsapi(query: str, since_hours: int = 24,
                  page_size: int = 50, language: str = "en") -> List[Dict]:
    key = os.getenv("NEWSAPI_KEY")
    if not key:
        return []
    frm = (now_utc() - dt.timedelta(hours=since_hours)).date().isoformat()
    params = {
        "q": query,
        "from": frm,
        "language": language,
        "pageSize": min(page_size, 100),
        "sortBy": "publishedAt",
        "apiKey": key,
    }
    r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    out = []
    for a in data.get("articles", []):
        out.append({
            "title": a.get("title"),
            "url": clean_url(a.get("url")),
            "source": (a.get("source") or {}).get("name"),
            "date": to_iso(a.get("publishedAt")),
            "text": "",
        })
    return out

def fetch_gdelt(query: str, since_hours: int = 24, max_records: int = 50) -> List[Dict]:
    # guard: very short tokens cause errors—expand 'ai' -> 'artificial intelligence'
    q = query.strip()
    if len(q) < 3:
        q = f'"{q}" OR artificial intelligence'
    start = (now_utc() - dt.timedelta(hours=since_hours)).strftime("%Y%m%d%H%M%S")
    params = {
        "query": q,
        "mode": "ArtList",
        "maxrecords": str(max_records),
        "format": "JSON",
        "startdatetime": start,
        "sort": "DateDesc",
    }
    r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=20)
    try:
        data = r.json()
    except Exception:
        return []
    arts = data.get("articles", [])
    out = []
    for a in arts:
        out.append({
            "title": a.get("title"),
            "url": clean_url(a.get("url")),
            "source": a.get("sourceDomain"),
            "date": to_iso(a.get("seendate") or a.get("date")),
            "text": "",
        })
    return out

# ============================================================
# 3) Topic Queries & Normalization
# ============================================================

TOPIC_QUERIES = {
    "ai":       '("artificial intelligence" OR AI OR "machine learning")',
    "politics": '(election OR parliament OR government OR president)',
    "africa":   '(Africa OR African)',
    "world":    '(world OR international)',
    "tunisia":  '(Tunisia OR Tunis OR Tunisian)',
    "football": '(football OR soccer OR FIFA OR UEFA OR CAF)',
}

def normalize_item(it: Dict, topic: str, fetched_at_iso: str) -> Dict:
    """
    Normalize a raw provider item into a canonical article dict that we store in DB.
    """
    url = clean_url(it.get("url") or "")
    domain = urllib.parse.urlsplit(url).netloc if url else (it.get("source") or "")
    title = (it.get("title") or "").strip()
    text = (it.get("text") or "").strip()
    date_iso = to_iso(it.get("date"))
    lang = detect_lang(title, text) or it.get("language") or it.get("lang")
    return {
        "id": sha(f"{domain}|{title}|{date_iso or ''}"),
        "topic": topic.lower(),
        "lang": (lang or "").lower(),
        "title": title,
        "text": text,
        "date": date_iso,
        "source": domain,
        "url": url,
        "fetched_at": fetched_at_iso,
    }

def fetch_all_for_topic(topic: str, languages: List[str], since_hours: int, limit: int,
                        providers=("rss", "newsapi", "gdelt")) -> List[Dict]:
    """
    Fetch for a single topic. NewsAPI loops per language; RSS/GDELT are post-filtered by language.
    """
    q = TOPIC_QUERIES.get(topic.lower(), topic)
    items: List[Dict] = []

    if "rss" in providers:
        items += fetch_rss(DEFAULT_RSS, limit // 2)

    if "newsapi" in providers:
        for lg in languages:
            items += fetch_newsapi(q, since_hours,
                                   page_size=min(100, limit),
                                   language=lg)

    if "gdelt" in providers:
        items += fetch_gdelt(q, since_hours, max_records=min(250, limit))

    # extract text, detect language, keep only requested languages, dedupe per (url,title)
    seen = set()
    out: List[Dict] = []
    keep_langs = set(languages)
    for it in items:
        if not it.get("text") and it.get("url"):
            it["text"] = extract_text(it["url"])
        lang = (detect_lang(it.get("title", ""), it.get("text", "")) or "").lower()
        it["language"] = lang
        if lang not in keep_langs:
            continue
        key = sha((it.get("url") or "") + "||" + (it.get("title") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)

    out.sort(key=lambda x: x.get("date") or "", reverse=True)
    return out

def fetch_multi_topics(topics: List[str], languages: List[str], since_hours: int, limit: int,
                       providers=("rss", "newsapi", "gdelt")) -> List[Dict]:
    """
    Fetch for multiple topics and normalize. Also dedupe across topics.
    """
    fetched_at_iso = now_utc().isoformat()
    bag: List[Dict] = []

    for t in topics:
        raw = fetch_all_for_topic(t, languages, since_hours, limit, providers=providers)
        for it in raw:
            bag.append(normalize_item(it, topic=t, fetched_at_iso=fetched_at_iso))

    # cross-topic dedupe (source + title)
    seen = set()
    final: List[Dict] = []
    for a in bag:
        k = sha((a["source"] or "") + "||" + (a["title"] or ""))
        if k in seen:
            continue
        seen.add(k)
        final.append(a)

    final.sort(key=lambda x: x.get("date") or "", reverse=True)
    return final

# ============================================================
# 4) Database Helpers (SQLite)  — NO DUPLICATES ACROSS RUNS
# ============================================================

DB_PATH = os.path.join("data", "news.db")

def get_db_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open a connection and ensure schema exists."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id           TEXT PRIMARY KEY,
            url          TEXT UNIQUE,
            title        TEXT,
            text         TEXT,
            summary_40w  TEXT,
            source       TEXT,
            language     TEXT,
            topic        TEXT,
            published_at TEXT,
            fetched_at   TEXT
        )
    """)
    return conn

def article_exists(conn: sqlite3.Connection, url: str) -> bool:
    """Check if an article with this URL is already stored."""
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM articles WHERE url = ? LIMIT 1", (url,))
    return cur.fetchone() is not None

def insert_articles(conn: sqlite3.Connection, articles: List[Dict]) -> int:
    """
    Insert normalized articles into the DB, skipping those whose URL already exists.
    Precomputes a 40-word summary for each newly inserted article.
    Returns the number of newly inserted rows.
    """
    cur = conn.cursor()
    new_count = 0

    for a in articles:
        url = a.get("url") or ""
        if not url:
            continue

        # Skip if this URL is already in DB (no double storage, no double summarization)
        if article_exists(conn, url):
            continue

        title = a.get("title") or ""
        text = a.get("text") or ""
        lang = (a.get("lang") or a.get("language") or "en").lower()
        topic = a.get("topic") or ""
        pub_date = a.get("date") or ""
        fetched_at = a.get("fetched_at") or ""
        source = a.get("source") or ""
        article_id = a.get("id") or sha(f"{source}|{title}|{pub_date}")

        # Precompute summary once, using your DistilBART-based helper
        summary_40w = summarize_40w(text, lang=lang) if text else ""

        try:
            cur.execute(
                """
                INSERT INTO articles
                    (id, url, title, text, summary_40w, source,
                     language, topic, published_at, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (article_id, url, title, text, summary_40w,
                 source, lang, topic, pub_date, fetched_at)
            )
            new_count += 1
        except sqlite3.IntegrityError:
            # URL or ID already exists (race condition or rare case) -> ignore
            continue

    conn.commit()
    return new_count

# ============================================================
# 5) Merge JSON Corpus for the Existing Agent (optional bridge)
# ============================================================

def make_agent_corpus(demo_path: str, fresh_path: str, out_path: str):
    """
    Merge fixed demo corpus + freshly fetched news into one JSON for the Agent:
    fields: id,title,text,url,source,date,language (plus topic,fetched_at kept as extras)
    This keeps compatibility with your existing retriever that reads live_corpus.json.
    """
    def load_json(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    demo = load_json(demo_path)
    fresh = load_json(fresh_path)

    fresh_agent = []
    for a in fresh:
        fresh_agent.append({
            "id": a.get("id"),
            "title": a.get("title"),
            "text": a.get("text"),
            "url": a.get("url"),
            "source": a.get("source"),
            "date": a.get("date"),
            "language": a.get("lang"),
            "topic": a.get("topic"),
            "fetched_at": a.get("fetched_at"),
        })

    merged = list(demo) + fresh_agent
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

# ============================================================
# 6) CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse, pathlib

    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default="ai,politics,africa,world,tunisia,football")
    ap.add_argument("--languages", default="en,fr")
    ap.add_argument("--since-hours", type=int, default=24)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--providers", default="rss,gdelt,newsapi")
    ap.add_argument("--out", default="data/raw_articles.json")
    ap.add_argument("--demo-corpus", default="data/corpus_for_demo.json")
    ap.add_argument(
        "--agent-out",
        default="data/live_corpus.json",
        help="Merged demo + fresh, for the Agent to read",
    )
    ap.add_argument(
        "--db-path",
        default=DB_PATH,
        help="Path to SQLite DB where articles + summaries are stored.",
    )
    args = ap.parse_args()

    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    languages = [l.strip().lower() for l in args.languages.split(",") if l.strip()]
    providers = tuple(p.strip() for p in args.providers.split(",") if p.strip())

    # 1) Fetch fresh normalized articles
    arts = fetch_multi_topics(
        topics=topics,
        languages=languages,
        since_hours=args.since_hours,
        limit=args.limit,
        providers=providers,
    )

    # 2) Still write JSON dump (for inspection / backup)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(arts, f, ensure_ascii=False, indent=2)
    print(f"[fetch_news] Saved {len(arts)} normalized articles to {args.out}")

    # 3) Insert into SQLite DB (no duplicates across runs, 40-word summaries precomputed)
    conn = get_db_conn(args.db_path)
    new_count = insert_articles(conn, arts)
    conn.close()
    print(f"[db] Inserted {new_count} new articles into {args.db_path} (existing URLs skipped).")

    # 4) Keep generating live_corpus.json for your existing agent (bridge step)
    make_agent_corpus(args.demo_corpus, args.out, args.agent_out)
    print(f"[agent_corpus] Wrote merged corpus to {args.agent_out}")
