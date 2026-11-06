"""
Author-history features: "given an author's EARLIER books and how they did, how will their next book do?"

For a target book published in year Y, history = the same author's books with
first_publish_year STRICTLY LESS than Y. Strictly-earlier (not "<=") means:
  * the book never sees itself, and
  * books from the same year (whose order we cannot know) never leak into each other.

The one function below is used by the feature pipeline (training data), the inference
pipeline and the API, so train and serve compute the features identically.
"""
import numpy as np
import pandas as pd

AUTHOR_COL = "author_key"
YEAR_COL = "first_publish_year"
REACH_SOURCE_COL = "readinglog_count"
PAGES_COL = "num_pages"   # optional columns: used when present in BOTH the targets and the history
GENRE_COL = "genre"

HISTORY_FEATURES = [
    "prior_n_books",
    "prior_mean_log_reach",
    "prior_median_log_reach",
    "prior_max_log_reach",
    "last_book_log_reach",
    "years_since_last_book",
    "author_career_years",
    "avg_years_between_books",   # cadence: prolific every year vs. a book every 6 years
    "last3_mean_log_reach",      # momentum: how the author's most recent 3 books did
    "pages_vs_author_median",    # log(pages / author's usual pages): a much longer/shorter book than usual?
    "author_genre_share",        # share of the author's earlier books in THIS book's genre (0 = new territory)
]


def log_reach(counts) -> np.ndarray:
    """Popularity on a log scale: 100 readers vs 1,000 matters as much as 10,000 vs 100,000."""
    return np.log10(1.0 + np.asarray(counts, dtype=float))


def compute_history(targets: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """History features for every row of `targets`, computed from `history` rows only.

    `history` needs author_key, first_publish_year, readinglog_count (it may be the same
    table as `targets`: strictly-earlier years keep a book out of its own history).
    Returns a frame with the same index as `targets`; authors with no earlier book get
    prior_n_books = 0 and NaN for the rest (the training code drops those rows).
    """
    optional = [c for c in (PAGES_COL, GENRE_COL) if c in history.columns]
    hist = history[[AUTHOR_COL, YEAR_COL, REACH_SOURCE_COL] + optional].dropna(
        subset=[AUTHOR_COL, YEAR_COL, REACH_SOURCE_COL]).copy()
    hist["_lr"] = log_reach(hist[REACH_SOURCE_COL])
    by_author = {a: g for a, g in hist.groupby(AUTHOR_COL)}

    nan = [np.nan] * len(targets)
    t_pages = targets[PAGES_COL].to_numpy(dtype=float) if PAGES_COL in targets else nan
    t_genre = targets[GENRE_COL].to_numpy() if GENRE_COL in targets else nan

    rows = []
    for author, year, page, genre in zip(targets[AUTHOR_COL], targets[YEAR_COL], t_pages, t_genre):
        g = by_author.get(author)
        prior = g[g[YEAR_COL] < year] if g is not None and pd.notna(year) else None
        if prior is None or prior.empty:
            rows.append({"prior_n_books": 0, **{c: np.nan for c in HISTORY_FEATURES[1:]}})
            continue
        last_year = prior[YEAR_COL].max()
        pages_vs, genre_share = np.nan, np.nan
        if PAGES_COL in prior and pd.notna(page) and page > 0:
            usual = prior[PAGES_COL].dropna().median()
            if pd.notna(usual) and usual > 0:
                pages_vs = float(np.log(page / usual))
        if GENRE_COL in prior and pd.notna(genre):
            genre_share = float((prior[GENRE_COL] == genre).mean())
        rows.append({
            "prior_n_books": len(prior),
            "prior_mean_log_reach": prior["_lr"].mean(),
            "prior_median_log_reach": prior["_lr"].median(),
            "prior_max_log_reach": prior["_lr"].max(),
            "last_book_log_reach": prior.loc[prior[YEAR_COL] == last_year, "_lr"].mean(),
            "years_since_last_book": float(year - last_year),
            "author_career_years": float(year - prior[YEAR_COL].min()),
            "avg_years_between_books": (float(prior[YEAR_COL].max() - prior[YEAR_COL].min()) / (len(prior) - 1)
                                        if len(prior) >= 2 else np.nan),
            # sorted by (year, reach) so ties inside a year resolve the same way at training and serving time
            "last3_mean_log_reach": prior.sort_values([YEAR_COL, "_lr"]).tail(3)["_lr"].mean(),
            "pages_vs_author_median": pages_vs,
            "author_genre_share": genre_share,
        })
    return pd.DataFrame(rows, index=targets.index)