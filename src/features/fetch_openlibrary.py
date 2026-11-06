"""
Fetch books + popularity outcomes from Open Library (free, no key).

Design
------
1. DISCOVER authors: genre searches -> author_keys, most frequently seen first.
2. For each author, fetch their WHOLE catalogue (one query per author_key). A "book" is an
   Open Library *work* (all editions already merged), so there are no reprint duplicates.
3. Keep only usable books: English, primary author, has first_publish_year, enough reading-log
   activity, not an obvious non-novel (study guide, box set, ...).
4. Keep authors with MIN_BOOKS_PER_AUTHOR..MAX_BOOKS_PER_AUTHOR usable books: the model needs
   author HISTORY, and a few prolific authors must not dominate the data.
5. Stop when TARGET_ROWS *and* TARGET_AUTHORS are reached. Every author's raw response is cached in raw_openlibrary/,
   so re-running is free and resumable.

Columns
-------
PRE-launch (knowable before release):  work_key, title, author_key, author_name,
                                       first_publish_year, number_of_pages_median, subjects (genre tags)
OUTCOMES (after release; never model inputs): readinglog_count, want_to_read_count,
                                       already_read_count, currently_reading_count,
                                       edition_count, ratings_count, ratings_average

Open Library asks API users to identify themselves: set OPENLIBRARY_CONTACT in .env
(an email or URL). It is only sent in the User-Agent header.
"""
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SEARCH_URL = "https://openlibrary.org/search.json"
FIELDS = ("key,title,author_name,author_key,first_publish_year,number_of_pages_median,edition_count,"
          "ratings_average,ratings_count,want_to_read_count,already_read_count,"
          "currently_reading_count,readinglog_count,subject")

GENRES = ["thriller", "mystery", "romance", "fantasy", "science_fiction", "horror",
          "historical_fiction", "crime", "literary_fiction", "suspense"]
DISCOVERY_PAGES = 3            # x100 books per genre
PAGE_SIZE = 100
MAX_PAGES_PER_AUTHOR = 5       # up to 500 works per author is plenty
MIN_READINGLOG = 5             # below this a book has no usable popularity signal
MIN_BOOKS_PER_AUTHOR = 5       # need history: >= 3-4 earlier books for most rows
MAX_BOOKS_PER_AUTHOR = 60      # skip series factories / huge back catalogues (Stine, Patterson...)
TARGET_ROWS = 800
TARGET_AUTHORS = 60            # stop only when BOTH targets are met: many authors, not just many rows
CACHE_DIR = Path("raw_openlibrary_v2")          # v2: responses now include `subject`
OLD_DISCOVERY = Path("raw_openlibrary") / "_discovery.json"   # reused so the same authors are kept
PAUSE = 1.0                    # seconds between requests: be polite

_NON_NOVEL_RE = re.compile(
    r"\b(study guide|sparknotes|cliffs?\s?notes|summary|analysis|workbook|diary|journal|notebook|"
    r"planner|calendar|colou?ring|box(?:ed)? set|collection|collected|omnibus|bundle|complete works|"
    r"quiz|trivia|companion|essays|screenplay|teacher|lesson plan|graphic novel|anthology|"
    r"selected (?:stories|works)|short stories|vol\.|volume \d)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable without network)
# --------------------------------------------------------------------------- #
def title_key(title: str) -> str:
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", str(title).lower().replace("&", " and "))
    t = re.split(r"[:–—]| - ", t)[0]
    t = re.sub(r"[^a-z0-9 ]", " ", t).strip()
    t = re.sub(r"^(the|a|an) ", "", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_doc(doc: dict, fetched_at: str):
    """One search result -> a flat row, or None when it is not a usable book."""
    keys = doc.get("author_key") or []
    names = doc.get("author_name") or []
    title = doc.get("title")
    year = doc.get("first_publish_year")
    if not title or not keys or not year or not (1800 <= int(year) <= datetime.now().year):
        return None
    if _NON_NOVEL_RE.search(title):
        return None
    names = doc.get("author_name") or []
    if names and title_key(title) == title_key(names[0]):   # a record titled with the author's own name
        return None
    if int(doc.get("readinglog_count") or 0) < MIN_READINGLOG:
        return None
    return {
        "work_key": doc.get("key"),
        "title": title,
        "author_key": keys[0],
        "author_name": names[0] if names else None,
        "n_authors": len(keys),
        "first_publish_year": int(year),
        "number_of_pages_median": doc.get("number_of_pages_median"),
        "edition_count": doc.get("edition_count"),
        "ratings_average": doc.get("ratings_average"),
        "ratings_count": doc.get("ratings_count"),
        "want_to_read_count": doc.get("want_to_read_count"),
        "already_read_count": doc.get("already_read_count"),
        "currently_reading_count": doc.get("currently_reading_count"),
        "readinglog_count": int(doc.get("readinglog_count") or 0),
        "subjects": " | ".join(str(t) for t in (doc.get("subject") or [])[:60]),
        "fetched_at": fetched_at,
    }


def build_author_books(author_key: str, docs: list, fetched_at: str) -> pd.DataFrame:
    """Usable books by this PRIMARY author; the same title appearing twice keeps the busier work."""
    rows = []
    for d in docs:
        row = parse_doc(d, fetched_at)
        if row and row["author_key"] == author_key and row["n_authors"] == 1:  # solo-authored only
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["_k"] = df["title"].map(title_key)
    df = (df.sort_values("readinglog_count", ascending=False)
            .drop_duplicates("_k").drop(columns="_k")
            .sort_values("first_publish_year").reset_index(drop=True))
    return df


def rank_authors(discovered_docs: list) -> list:
    """Author keys, most frequently seen first (frequent = prolific in the genre searches)."""
    counts = Counter(d["author_key"][0] for d in discovered_docs if d.get("author_key"))
    return [k for k, _ in counts.most_common()]


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def make_session() -> requests.Session:
    s = requests.Session()
    contact = os.environ.get("OPENLIBRARY_CONTACT", "unset-contact")
    s.headers["User-Agent"] = f"novel-success-ml-project ({contact})"
    retry = Retry(total=5, backoff_factor=2.0, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",), respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _get(session, params):
    r = session.get(SEARCH_URL, params=params, timeout=60)
    r.raise_for_status()
    time.sleep(PAUSE)
    return r.json()


def discover_authors(session, genres=GENRES, pages=DISCOVERY_PAGES) -> list:
    cache = CACHE_DIR / "_discovery.json"
    if cache.exists():
        return json.loads(cache.read_text())
    if OLD_DISCOVERY.exists():   # same author list as the first fetch, so results stay comparable
        CACHE_DIR.mkdir(exist_ok=True)
        cache.write_text(OLD_DISCOVERY.read_text())
        return json.loads(cache.read_text())
    docs = []
    for g in genres:
        for p in range(1, pages + 1):
            data = _get(session, {"q": f"subject:{g}", "language": "eng", "limit": PAGE_SIZE,
                                  "page": p, "fields": "key,author_key"})
            batch = data.get("docs", [])
            docs += batch
            if len(batch) < PAGE_SIZE:
                break
        print(f"discovery: {g} -> {len(docs)} books so far")
    authors = rank_authors(docs)
    CACHE_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps(authors))
    return authors


def fetch_author_docs(session, author_key: str) -> list:
    cache = CACHE_DIR / f"{author_key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    docs = []
    for p in range(1, MAX_PAGES_PER_AUTHOR + 1):
        data = _get(session, {"q": f"author_key:{author_key}", "language": "eng",
                              "limit": PAGE_SIZE, "page": p, "fields": FIELDS})
        batch = data.get("docs", [])
        docs += batch
        if len(batch) < PAGE_SIZE:
            break
    CACHE_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps(docs))  # cached only after a COMPLETE fetch
    return docs


def fetch_books(out_path="openlibrary_books.csv", target_rows=TARGET_ROWS,
                target_authors=TARGET_AUTHORS, session=None) -> pd.DataFrame:
    try:  # pick up OPENLIBRARY_CONTACT from .env when python-dotenv is installed
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv(), override=False)
    except ImportError:
        pass
    session = session or make_session()
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    frames, total, failures = [], 0, []
    for i, author_key in enumerate(discover_authors(session), 1):
        if total >= target_rows and len(frames) >= target_authors:
            break
        try:
            books = build_author_books(author_key, fetch_author_docs(session, author_key), fetched_at)
        except requests.RequestException as exc:
            failures.append((author_key, str(exc)[:120]))
            print(f"[{i}] {author_key}: FAILED ({str(exc)[:80]})")
            continue
        if len(books) > MAX_BOOKS_PER_AUTHOR:
            print(f"[{i}] {author_key}: skipped ({len(books)} books: likely a series factory)")
        elif len(books) >= MIN_BOOKS_PER_AUTHOR:
            frames.append(books)
            total += len(books)
            print(f"[{i}] {author_key} {books['author_name'].iloc[0]}: {len(books)} books (total {total})")
    if not frames:
        raise RuntimeError(f"Nothing usable fetched. Failures: {failures}")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"\nsaved {len(df)} books by {df['author_key'].nunique()} authors -> {out_path} | "
          f"failed authors: {len(failures)}")
    from src.features.genre import genre_from_subjects
    has = df["subjects"].fillna("").str.len() > 0
    print(f"books with subject tags: {has.mean():.0%}")
    print("genre buckets:", df["subjects"].map(genre_from_subjects).value_counts().to_dict())
    return df


if __name__ == "__main__":
    fetch_books()