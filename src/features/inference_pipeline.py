"""
Inference: score books with the model currently in Production.

DEFAULT  python src/features/inference_pipeline.py
    Scores the REAL hold-out: the newest books of authors the model trained on, which the model never
    saw (train_pipeline.py sets them aside and records their ids with the model). Each prediction is
    compared with how the book actually did. This is the "next book by a known author" situation.

NEW BOOKS  python src/features/inference_pipeline.py --file my_books.csv
    One row per book: author_key (Open Library id, e.g. OL19981A) and book_published_year are required;
    num_pages and genre are optional (blank = the author's usual) and so is title. No outcomes needed.

Features are built exactly as in training: the author's earlier books come from the catalogue in the
feature store (strictly earlier years, author_history.compute_history); scaler and model come from the same
MLflow run. Outputs: inference_metrics/inference_predictions.csv and, when outcomes are known,
inference_metrics/inference_metrics.json.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from author_history import compute_history
from config import (
    FEATURE_COLUMNS, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION, INT_TO_LABEL, LABEL_TO_INT,
    METRICS_DIR, MIN_PRIOR_BOOKS,
)
from evaluation import evaluate_binary, lift_at
from genre import add_genre_columns, clean_genre
from labeling import REQUIRED as LABEL_REQUIRED

LOCAL_COPY = "features_local.csv"   # written by feature_pipeline.py


def check_columns(df: pd.DataFrame, needed, what: str) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{what} is missing columns {missing}. It has: {sorted(df.columns)}")


def to_history_table(stored: pd.DataFrame) -> pd.DataFrame:
    """The feature store calls the year `book_published_year`; compute_history expects first_publish_year."""
    check_columns(stored, ["author_key", "book_published_year", "readinglog_count"], "Stored catalogue")
    return stored.rename(columns={"book_published_year": "first_publish_year"})


def load_history() -> pd.DataFrame:
    """Whole stored catalogue; Hopsworks first, the local copy only if Hopsworks is unreachable."""
    try:
        from hopsworks_utils import login
        fs = login().get_feature_store()
        stored = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION).read()
        print(f"[inference] history: {len(stored)} books from the feature store")
    except Exception as exc:  # noqa: BLE001
        if not os.path.exists(LOCAL_COPY):
            raise
        print(f"[inference] WARNING: feature store unreachable ({type(exc).__name__}); "
              f"using the local copy {LOCAL_COPY}, which may be stale")
        stored = pd.read_csv(LOCAL_COPY)
    return to_history_table(stored)


def read_new_books(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "first_publish_year" in df.columns and "book_published_year" not in df.columns:
        df = df.rename(columns={"first_publish_year": "book_published_year"})
    check_columns(df, ["author_key", "book_published_year"], f"Inference file '{path}'")
    return df


def build_features(new_books: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Same features as the feature pipeline, computed from the stored catalogue."""
    out = new_books.copy().reset_index(drop=True)
    out["first_publish_year"] = out["book_published_year"].astype(int)

    # blank pages -> the author's median page count
    out["num_pages"] = pd.to_numeric(out["num_pages"] if "num_pages" in out else np.nan, errors="coerce")
    if out["num_pages"].isna().any() and "num_pages" in history.columns:
        out["num_pages"] = out["num_pages"].fillna(out["author_key"].map(history.groupby("author_key")["num_pages"].median()))

    # blank genre -> the author's most common genre so far; anything unrecognised -> "other"
    given = out["genre"] if "genre" in out else pd.Series(np.nan, index=out.index)
    given = given.where(given.astype(str).str.strip().replace("nan", "") != "", np.nan)
    if "genre" in history.columns:
        usual = history.dropna(subset=["genre"]).groupby("author_key")["genre"].agg(lambda g: g.mode().iat[0])
        given = given.fillna(out["author_key"].map(usual))
    out["genre"] = given.map(lambda g: clean_genre(g) if pd.notna(g) else np.nan)

    hist = compute_history(out, history)
    out = pd.concat([out.drop(columns=["first_publish_year"]), hist], axis=1)
    return add_genre_columns(out)


def run_inference(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Score books using ONLY pre-launch features. transform() only, never fit() again.
    Books the model was not trained for (too little history / missing values) are reported, not guessed."""
    cols = bundle["feature_columns"]
    check_columns(df, cols, "Inference data")
    out = df.copy()
    out["predicted_hit_miss"], out["hit_probability"], out["skip_reason"] = None, np.nan, None

    too_short = out["prior_n_books"] < MIN_PRIOR_BOOKS
    missing = out[cols].isna().any(axis=1)
    out.loc[too_short, "skip_reason"] = f"author has fewer than {MIN_PRIOR_BOOKS} earlier books in the catalogue"
    out.loc[~too_short & missing, "skip_reason"] = "missing feature values (e.g. no page count)"
    scorable = out["skip_reason"].isna()
    if (~scorable).any():
        print(f"[inference] skipped {int((~scorable).sum())} of {len(out)} book(s): see skip_reason")
    if scorable.any():
        X = bundle["scaler"].transform(out.loc[scorable, cols].to_numpy(dtype=float))
        out.loc[scorable, "predicted_hit_miss"] = [INT_TO_LABEL[int(p)] for p in bundle["model"].predict(X)]
        out.loc[scorable, "hit_probability"] = bundle["model"].predict_proba(X)[:, 1]  # class 1 = Hit
    return out


class CannotScore(ValueError):
    """The book can't be scored (unknown author, too little history, ...). The message says why."""


def predict_one(book: dict, bundle: dict, history: pd.DataFrame) -> dict:
    """Score a single book. Used by the API so it shares every line of logic with batch inference."""
    row = run_inference(build_features(pd.DataFrame([book]), history), bundle).iloc[0]
    if row["skip_reason"]:
        raise CannotScore(row["skip_reason"])
    return {
        "predicted_hit_miss": row["predicted_hit_miss"],
        "hit_probability": float(row["hit_probability"]),
        "features_used": {c: float(row[c]) for c in bundle["feature_columns"]},
        "author_earlier_books": int(row["prior_n_books"]),
    }


def author_info(history: pd.DataFrame, author_key: str):
    g = history[history["author_key"] == author_key]
    if g.empty:
        return None
    name = g["author_name"].dropna().iloc[0] if "author_name" in g and g["author_name"].notna().any() else None
    return {"author_key": author_key, "author_name": name, "books_in_catalogue": int(len(g)),
            "latest_book_year": int(g["first_publish_year"].max())}


def evaluate_if_outcomes_known(scored: pd.DataFrame, bundle: dict):
    """Monitoring only: label with the SAVED rule (never re-derived from this batch)."""
    if not set(LABEL_REQUIRED).issubset(scored.columns):
        return None
    known = scored.dropna(subset=LABEL_REQUIRED + ["predicted_hit_miss"])
    if known.empty:
        return None
    y_true = bundle["label_rule"].transform(known).map(LABEL_TO_INT).to_numpy()
    y_pred = known["predicted_hit_miss"].map(LABEL_TO_INT).to_numpy()
    metrics = evaluate_binary(y_true, y_pred)
    metrics["auc"] = (float(roc_auc_score(y_true, known["hit_probability"].astype(float)))
                      if len(np.unique(y_true)) == 2 else float("nan"))
    metrics.update({k: v for k, v in lift_at(y_true, known["hit_probability"].astype(float)).items() if k != "lift_k"})
    return metrics


def score_holdout(history: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Score the books that were held out of training. Inputs are the PRE-launch fields only; the real outcome
    is attached afterwards, purely to check the prediction."""
    ids = bundle.get("holdout_book_ids")
    if not ids:
        raise RuntimeError("This model has no recorded hold-out books (it was registered before hold-out "
                           "tracking). Run train_pipeline.py again to register one.")
    rows = history[history["book_id"].astype(str).isin(set(map(str, ids)))].reset_index(drop=True)
    lost = len(set(map(str, ids))) - len(rows)
    if lost:
        print(f"[inference] WARNING: {lost} hold-out book(s) are no longer in the feature store")
    prelaunch = ["book_id", "title", "author_key", "author_name", "book_published_year", "num_pages", "genre"]
    inputs = rows.rename(columns={"first_publish_year": "book_published_year"})
    inputs = inputs[[c for c in prelaunch if c in inputs.columns]]

    scored = run_inference(build_features(inputs, history), bundle)
    scored["readinglog_count"] = rows["readinglog_count"].to_numpy()      # real outcome, used only for checking
    known = scored.dropna(subset=LABEL_REQUIRED + ["predicted_hit_miss"])
    scored["actual_hit_miss"] = None
    scored.loc[known.index, "actual_hit_miss"] = bundle["label_rule"].transform(known).to_numpy()
    scored["correct"] = (scored["predicted_hit_miss"] == scored["actual_hit_miss"]).astype(object)
    scored.loc[scored["predicted_hit_miss"].isna(), "correct"] = None
    return scored


def holdout_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Per author: how many hold-out books and how many were called correctly."""
    d = scored.dropna(subset=["predicted_hit_miss"])
    g = d.groupby("author_name" if "author_name" in d else "author_key")
    return pd.DataFrame({"books": g.size(), "correct": g["correct"].apply(lambda c: int(c.astype(bool).sum()))}).assign(
        accuracy=lambda t: (t["correct"] / t["books"]).round(2)).sort_values("books", ascending=False)


def print_metrics(metrics: dict, mpath: str) -> None:
    print(f"[inference] {metrics['n_eval']} books with known outcomes: accuracy {metrics['accuracy']:.3f} vs baseline "
          f"{metrics['baseline_accuracy']:.3f} | Miss F1 {metrics['miss_f1']:.3f} | AUC {metrics['auc']:.3f}")
    print(f"[inference] usefulness: of the 20% rated most likely to MISS, {metrics['bottom_miss_rate']:.0%} really were "
          f"Misses (random pick: {metrics['base_miss_rate']:.0%}); of the 20% rated most likely to HIT, "
          f"{metrics['top_hit_rate']:.0%} really were Hits (random pick: {metrics['base_hit_rate']:.0%})  -> {mpath}")


def main(argv=None):
    import argparse
    from model_registry import load_production_bundle  # lazy: keeps this module importable without mlflow

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1], add_help=True)
    parser.add_argument("--file", help="CSV of NEW books to score (default: the real hold-out books)")
    args = parser.parse_args(argv)

    bundle = load_production_bundle()
    print(f"[inference] model version {bundle['version']} (run {bundle['run_id']})")
    history = load_history()

    if args.file:
        scored = run_inference(build_features(read_new_books(args.file), history), bundle)
    else:
        scored = score_holdout(history, bundle)
        print(f"[inference] scoring {len(scored)} real hold-out books by {scored['author_key'].nunique()} authors "
              f"the model trained on but never saw these exact books")

    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, "inference_predictions.csv")
    scored.to_csv(path, index=False)
    shown = [c for c in ["author_name", "title", "book_published_year", "predicted_hit_miss", "hit_probability",
                         "actual_hit_miss"] if c in scored]
    print(scored[shown].head(20).to_string(index=False))
    print(f"[inference] wrote {len(scored)} rows to {path}")

    metrics = evaluate_if_outcomes_known(scored, bundle)
    if metrics:
        mpath = os.path.join(METRICS_DIR, "inference_metrics.json")
        with open(mpath, "w") as f:
            json.dump(metrics, f, indent=4)
        print_metrics(metrics, mpath)
        if not args.file:
            print("\nper author (hold-out books):")
            print(holdout_summary(scored).to_string())
    else:
        print("[inference] no outcomes in this file: predictions only (no monitoring metrics)")


if __name__ == "__main__":
    main()