"""
The two tests that matter most for this project:

1. NO LEAKAGE   - a book's features never use its own outcome, or the outcome of any later book.
2. REAL HOLD-OUT - the books the model is scored on were never trained on, and their predictions do not
                   depend on their own outcomes.
Run:  pytest
"""
import numpy as np
import pandas as pd
import pytest

import src.config
import src.features.feature_pipeline as fp
import src.features.inference_pipeline as ip
import src.features.train_pipeline as tp
from src.features.author_history import compute_history


def test_no_leakage_from_own_or_later_outcomes():
    df = pd.DataFrame({
        "author_key": ["A"] * 5 + ["B"] * 3,
        "first_publish_year": [2000, 2002, 2002, 2005, 2010, 1999, 2001, 2003],
        "readinglog_count": [100, 1000, 10, 10000, 50, 5, 5, 500],
    })
    base = compute_history(df, df)
    changed = df.copy()
    changed.loc[3, "readinglog_count"] = 10**9      # the 2005 book's OWN outcome
    changed.loc[4, "readinglog_count"] = 10**9      # a LATER book's outcome
    after = compute_history(changed, changed)
    # the 2005 book's features must not move, nor may any earlier book's
    pd.testing.assert_frame_equal(base.loc[[0, 1, 2, 3]], after.loc[[0, 1, 2, 3]])
    assert not after.loc[4].equals(base.loc[4])     # only the later book's history changes


def make_catalogue(n_authors=12, per=25, seed=0):
    """Synthetic catalogue run through the REAL feature code. Reach = author level + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_authors):
        level = rng.uniform(1.5, 4.0)
        for i in range(per):
            rows.append({"work_key": f"/works/OL{a}_{i}W", "title": f"T{a}_{i}", "author_key": f"A{a}",
                         "author_name": f"Author {a}", "first_publish_year": 1980 + 2 * i,
                         "number_of_pages_median": 250 + int(rng.integers(0, 200)),
                         "readinglog_count": int(10 ** (level + 0.06 * i + rng.normal(0, 0.5))),
                         "want_to_read_count": 5, "edition_count": 3,
                         "subjects": "Thrillers | Suspense" if (a + i) % 4 else "Mystery fiction | Detective and mystery stories"})
    return fp.select_final_columns(fp.add_features(fp.add_book_id(pd.DataFrame(rows))))


def test_real_holdout_is_never_trained_on_and_scored_without_outcomes():
    stored = make_catalogue()
    data = tp.prepare_data(stored)
    train_part, holdout = tp.split_latest_per_author(data, src.config.HOLDOUT_LATEST_SHARE)
    ids = set(holdout["book_id"].astype(str))
    assert len(holdout) > 0 and not ids & set(train_part["book_id"].astype(str))   # never trained on

    model, scaler, rule = tp.fit_final(train_part, "logreg")
    bundle = dict(model=model, scaler=scaler, label_rule=rule, feature_columns=src.config.FEATURE_COLUMNS,
                  version="1", run_id="r", holdout_book_ids=sorted(ids))
    hist = ip.to_history_table(stored)
    scored = ip.score_holdout(hist, bundle)
    assert set(scored["book_id"].astype(str)) == ids
    assert scored["predicted_hit_miss"].isin(["Hit", "Miss"]).all()

    # A book's own outcome must never feed its prediction: scramble the newest held-out book per author.
    scrambled = hist.copy()
    mask = scrambled["book_id"].astype(str).isin(ids)
    newest = scrambled[mask].sort_values("first_publish_year").groupby("author_key").tail(1).index
    scrambled.loc[newest, "readinglog_count"] = 1
    again = ip.score_holdout(scrambled, bundle)
    pd.testing.assert_series_equal(scored["hit_probability"], again["hit_probability"])

    # and every held-out book is newer than the author's newest training book
    newest_train = train_part.groupby("author_key")["book_published_year"].max()
    held = holdout.set_index("author_key")
    assert (held["book_published_year"].to_numpy() >= newest_train.reindex(held.index).to_numpy()).all()