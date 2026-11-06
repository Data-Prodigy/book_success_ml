"""
Feature pipeline:  openlibrary_books.csv  ->  author-history features  ->  Hopsworks feature group.

    python src/features/feature_pipeline.py              # build + upload
    python src/features/feature_pipeline.py --no-upload  # build + check + save a local CSV only

Every book is stored (not only the trainable ones): a book's history is its author's EARLIER
books, so inference needs the whole catalogue in the store. The training pipeline keeps just the
rows that have at least MIN_PRIOR_BOOKS earlier books.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from author_history import compute_history
from genre import GENRE_COLUMNS, add_genre_columns, fill_unknown_with_author_genre, genre_from_subjects
from config import (
    FEATURE_COLUMNS, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION, MIN_PRIOR_BOOKS,
    OUTCOME_COLUMNS, STORED_COLUMNS,
)

BOOKS_CSV = "openlibrary_books.csv"
LOCAL_COPY = "features_local.csv"


def load_books(path: str = BOOKS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["work_key", "title", "author_key", "author_name", "first_publish_year",
              "number_of_pages_median", "readinglog_count", "want_to_read_count", "edition_count", "subjects"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}. Run fetch_openlibrary.py again (the newer version also fetches subjects).")
    df = df.dropna(subset=["work_key", "author_key", "first_publish_year", "readinglog_count"])
    df = df.drop_duplicates("work_key").reset_index(drop=True)
    print(f"[features] loaded {len(df)} books by {df['author_key'].nunique()} authors")
    return df


def add_book_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["book_id"] = out["work_key"].astype(str).str.replace("/works/", "", regex=False)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["book_published_year"] = out["first_publish_year"].astype(int)
    pages = pd.to_numeric(out["number_of_pages_median"], errors="coerce")
    # pages is a property of the book (not an outcome): fill gaps with the author's median, then the overall median
    pages = pages.fillna(pages.groupby(out["author_key"]).transform("median")).fillna(pages.median())
    out["num_pages"] = pages
    out["genre"] = out["subjects"].map(genre_from_subjects)
    out = fill_unknown_with_author_genre(out)   # sparse tags must not mean "obscure book" (see genre.py)
    hist = compute_history(out, out)
    return add_genre_columns(pd.concat([out, hist], axis=1))


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("want_to_read_count", "edition_count"):   # a missing count from Open Library means "none"
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out[STORED_COLUMNS].reset_index(drop=True)


def check_frame(df: pd.DataFrame) -> None:
    """Cheap, strict checks (replaces the heavy Great Expectations suite)."""
    problems = []
    if df["book_id"].duplicated().any():
        problems.append(f"{int(df['book_id'].duplicated().sum())} duplicate book_id")
    for c in ["book_id", "author_key", "title", "book_published_year", "num_pages", "readinglog_count", "prior_n_books"]:
        if df[c].isna().any():
            problems.append(f"{int(df[c].isna().sum())} missing in {c}")
    if (df["readinglog_count"] < 0).any():
        problems.append("negative readinglog_count")
    if not df["book_published_year"].between(1800, pd.Timestamp.now().year + 1).all():
        problems.append("book_published_year outside 1800..next year")
    trainable = df["prior_n_books"] >= MIN_PRIOR_BOOKS
    if df.loc[trainable, FEATURE_COLUMNS].isna().any().any():
        problems.append("trainable rows have missing feature values")
    if problems:
        raise ValueError("Feature check failed: " + "; ".join(problems))
    print(f"[features] checks passed | rows {len(df)} | trainable (>= {MIN_PRIOR_BOOKS} earlier books): "
          f"{int(trainable.sum())} | authors {df['author_key'].nunique()}")


def cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["prior_n_books", "book_published_year", "readinglog_count", "want_to_read_count", "edition_count"]:
        out[c] = out[c].astype("int64")
    for c in ["prior_mean_log_reach", "prior_median_log_reach", "prior_max_log_reach", "last_book_log_reach",
              "years_since_last_book", "author_career_years", "avg_years_between_books",
              "last3_mean_log_reach", "pages_vs_author_median", "author_genre_share", "num_pages"]:
        out[c] = out[c].astype("float64")
    for c in GENRE_COLUMNS:
        out[c] = out[c].astype("int64")
    for c in ["book_id", "author_key", "author_name", "title", "genre"]:
        out[c] = out[c].astype("string")
    return out


def send_to_feature_store(df: pd.DataFrame) -> None:
    from hopsworks_utils import login
    fs = login().get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="One row per book: author-history features (from EARLIER books only) and "
                    "post-launch reach outcomes. The Hit/Miss label is derived at training time.",
        primary_key=["book_id"],
        online_enabled=True,
        stream=True,   # send through Kafka: works from outside the cluster (direct HDFS writes did not)
    )
    fg.insert(df, wait=True)
    print(f"commits: {fg.commit_details()}")
    mark_ready(len(df))


def mark_ready(n_rows: int) -> None:
    """Tell the next stage (training) how many rows to expect. Only when FEATURES_READY_FILE is set (Docker)."""
    path = os.environ.get("FEATURES_READY_FILE")
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"rows": int(n_rows)}, f)
        print(f"[features] wrote {path} (expecting {n_rows} rows)")


def main(upload: bool = True):
    df = load_books(BOOKS_CSV)
    df = add_book_id(df)
    df = add_features(df)
    df = select_final_columns(df)
    check_frame(df)
    df = cast_dtypes(df)
    df.to_csv(LOCAL_COPY, index=False)       # local backup: a network failure never costs you the data
    print(f"[features] saved a local copy -> {LOCAL_COPY}")
    if upload:
        send_to_feature_store(df)


if __name__ == "__main__":
    main(upload="--no-upload" not in sys.argv)