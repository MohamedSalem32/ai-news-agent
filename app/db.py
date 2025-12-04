# app/db.py
import os, sqlite3
from typing import Dict, List
from .summarize import summarize_40w  # reuse your DistilBART wrapper

DB_PATH = os.path.join("data", "news.db")

def get_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id          TEXT PRIMARY KEY,
            url         TEXT UNIQUE,
            title       TEXT,
            text        TEXT,
            summary_40w TEXT,
            source      TEXT,
            language    TEXT,
            topic       TEXT,
            published_at TEXT,
            fetched_at  TEXT
        )
    """)
    return conn

def insert_articles(conn: sqlite3.Connection, articles: List[Dict]) -> int:
    """
    Insert articles, skipping those whose URL is already in the DB.
    Returns the count of newly inserted rows.
    """
    cur = conn.cursor()
    new_count = 0

    for a in articles:
        url = a.get("url") or ""
        title = a.get("title") or ""
        text = a.get("text") or ""
        lang = (a.get("lang") or a.get("language") or "en").lower()
        topic = a.get("topic") or ""
        pub_date = a.get("date") or ""
        fetched_at = a.get("fetched_at") or ""

        # precompute summary once
        summary = summarize_40w(text, lang=lang) if text else ""

        try:
            cur.execute("""
                INSERT OR IGNORE INTO articles
                    (id, url, title, text, summary_40w, source, language, topic, published_at, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                a.get("id"),
                url,
                title,
                text,
                summary,
                a.get("source") or "",
                lang,
                topic,
                pub_date,
                fetched_at,
            ))
            if cur.rowcount > 0:
                new_count += 1
        except sqlite3.Error as e:
            print("[DB] insert error:", e, "for URL:", url)

    conn.commit()
    return new_count
