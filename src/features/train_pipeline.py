"""
Training pipeline:  Hopsworks feature store -> author-grouped evaluation -> pick a model -> MLflow registry.

Design rules
------------
* Only books with at least MIN_PRIOR_BOOKS earlier books are trained on (they have real history).
* The Hit/Miss label is derived HERE (labeling.HitMissRule), from the training rows only.
* HOLD-OUT: each author's NEWEST HOLDOUT_SHARE of books is set aside before anything else. The deployed
  model never sees them; their ids are saved with the model, and inference.py scores exactly those books
  ("same authors, books the model never trained on"). Model selection uses only the remaining books.
* Selection/CV evaluation is split BY AUTHOR: every book of a test author is held out together, so the score
  says "how well do we predict for authors the model has never seen", not "can it recognise an
  author it already knows". Random row splits would flatter the model.
* Candidates (logistic regression, random forest, XGBoost, and a soft-voting ENSEMBLE of them) are
  compared with the same repeated author-grouped cross-validation, next to the "always guess the
  majority class" baseline. A more complex candidate is chosen only if it beats the best simpler one
  by MIN_GAIN macro-F1; otherwise the simpler one wins.
* The newest 20% of EVERY author's books are held out and never fitted. The registered model is trained
  on the rest and judged on those real, unseen books of authors it knows (the "next book" situation).
  Their ids are saved with the model so the inference pipeline scores exactly those books.
  Scaler, label rule and column order are saved with it too.
* Same local files as before: confusion_matrix/confusion_matrix.png, inference_metrics/train_metrics.json.
"""
import json
import os
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import (
    MLFLOW_TRACKING_URI, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION,
    EXPERIMENT_NAME, REGISTERED_MODEL_NAME, FEATURE_COLUMNS, OUTCOME_COLUMNS,
    LABEL_MISS_QUANTILE, MIN_PRIOR_BOOKS, LABEL_TO_INT, HOLDOUT_LATEST_SHARE,
    CONFUSION_MATRIX_DIR, METRICS_DIR,
)
from evaluation import evaluate_binary, lift_at
from labeling import HitMissRule, REQUIRED as LABEL_REQUIRED
from model_registry import promote_to_production

GROUP_COL = "author_key"
MIN_ROWS = 100      # below this an evaluation says nothing; stop with a clear message
MIN_AUTHORS = 8     # author-grouped splits need several authors
MIN_GAIN = 0.01     # a more complex candidate must beat the best simpler one by this much macro-F1


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def fetch_df_from_feature_store() -> pd.DataFrame:
    from hopsworks_utils import login
    fs = login().get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    expected = _expected_rows()
    for attempt in range(1, 19):                       # up to ~3 minutes
        df = fg.read()
        if expected is None or len(df) >= expected:
            return df
        print(f"[train] feature store has {len(df)} of {expected} rows (still syncing); retry {attempt}/18 in 10s")
        time.sleep(10)
    raise RuntimeError(f"Feature store still has {len(df)} rows after waiting; the feature pipeline wrote {expected}.")


def _expected_rows():
    """Row count left by feature_pipeline (Docker only: FEATURES_READY_FILE). None = just read once."""
    path = os.environ.get("FEATURES_READY_FILE")
    if path and os.path.exists(path):
        with open(path) as f:
            return int(json.load(f)["rows"])
    return None


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep books that have history and complete values; say how many were dropped and why."""
    needed = sorted(set(LABEL_REQUIRED) | set(FEATURE_COLUMNS) | {GROUP_COL, "book_id"})
    absent = [c for c in needed if c not in df.columns]
    if absent:
        raise ValueError(
            f"Feature-store data is missing columns {absent}. Was feature group "
            f"'{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION} written by the current feature pipeline?")
    out = df.copy()
    n_read = len(out)
    if "book_id" in out.columns:
        out = out.drop_duplicates("book_id", keep="last")
    n_dupes = n_read - len(out)
    n_no_history = int((out["prior_n_books"] < MIN_PRIOR_BOOKS).sum())
    out = out[out["prior_n_books"] >= MIN_PRIOR_BOOKS]
    n_before = len(out)
    out = out.dropna(subset=needed).reset_index(drop=True)
    print(f"[train] rows read: {n_read} | duplicates: {n_dupes} | too little history (<{MIN_PRIOR_BOOKS} earlier books): "
          f"{n_no_history} | missing values: {n_before - len(out)} | usable: {len(out)} "
          f"by {out[GROUP_COL].nunique()} authors")
    if len(out) < MIN_ROWS or out[GROUP_COL].nunique() < MIN_AUTHORS:
        raise ValueError(f"Need at least {MIN_ROWS} usable rows by {MIN_AUTHORS}+ authors; have "
                         f"{len(out)} rows by {out[GROUP_COL].nunique()} authors.")
    return out


def split_by_author(df: pd.DataFrame, test_size: float, seed: int):
    """All books of an author go to the same side."""
    rng = np.random.default_rng(seed)
    authors = np.array(df[GROUP_COL].astype(str).unique(), dtype=object)
    rng.shuffle(authors)
    n_test = max(1, int(round(len(authors) * test_size)))
    in_test = df[GROUP_COL].astype(str).isin(set(authors[:n_test]))
    return df[~in_test].reset_index(drop=True), df[in_test].reset_index(drop=True)


def split_latest_per_author(df: pd.DataFrame, share: float):
    """Hold out the NEWEST `share` of every author's books (at least one each).

    The authors stay in training, but their newest books are never fitted. Newest-first also means no
    training row's history contains a held-out book, so nothing about the held-out outcomes leaks in.
    This is the "predict the author's next book" situation.
    """
    d = df.sort_values([GROUP_COL, "book_published_year", "book_id"]).reset_index(drop=True)
    newest_rank = d.groupby(GROUP_COL).cumcount(ascending=False)        # 0 = the author's newest book
    size = d.groupby(GROUP_COL)[GROUP_COL].transform("size")
    in_holdout = newest_rank < np.maximum(1, np.ceil(size * share))
    return d[~in_holdout].reset_index(drop=True), d[in_holdout].reset_index(drop=True)


def author_folds(df: pd.DataFrame, n_splits: int, n_repeats: int, seed: int):
    """Repeated K-fold over AUTHORS (not rows). Yields (train_idx, test_idx) positional indices."""
    groups = df[GROUP_COL].astype(str).to_numpy()
    authors = np.unique(groups)
    n_splits = max(2, min(n_splits, len(authors)))
    for rep in range(n_repeats):
        perm = np.random.default_rng(seed + rep).permutation(authors)
        for k in range(n_splits):
            test_mask = np.isin(groups, perm[k::n_splits])
            yield np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def make_logreg(y_train: np.ndarray, seed: int = 123, cv: int = 5):
    per_class = np.bincount(y_train, minlength=2)
    if per_class.min() < 2:
        raise ValueError(f"Training labels have {per_class[0]} Miss / {per_class[1]} Hit; "
                         f"each class needs at least 2 examples.")
    # L2 is the default (an explicit penalty="l2" is deprecated in scikit-learn 1.8, removed in 1.10).
    return LogisticRegressionCV(cv=max(2, min(cv, int(per_class.min()))), solver="lbfgs",
                                max_iter=1000, refit=True, scoring="neg_log_loss")


def make_xgboost(y_train: np.ndarray, seed: int = 123, cv: int = 5):
    """Deliberately shallow and regularised: ~1.5k rows from ~60 authors is still a small dataset."""
    from xgboost import XGBClassifier  # ImportError -> the caller skips this candidate
    return XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_lambda=5.0, eval_metric="logloss", random_state=seed, n_jobs=1)


def make_random_forest(y_train: np.ndarray, seed: int = 123, cv: int = 5):
    """Bagged trees: a different kind of error from boosting. Small leaves are forbidden on purpose."""
    return RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10,
                                  max_features="sqrt", n_jobs=1, random_state=seed)


def make_ensemble(y_train: np.ndarray, seed: int = 123, cv: int = 5):
    """Soft vote (average of predicted probabilities) of logistic regression + random forest + XGBoost.
    Models that err in different ways average out. XGBoost is left out when it isn't installed."""
    members = [("logreg", make_logreg(y_train, seed, cv)), ("rf", make_random_forest(y_train, seed, cv))]
    try:
        members.append(("xgb", make_xgboost(y_train, seed, cv)))
    except ImportError:
        pass
    return VotingClassifier(members, voting="soft")


# Simplest first: select_best only moves to a later (more complex) entry when it is clearly better.
MODEL_FACTORIES = {"logreg": make_logreg, "random_forest": make_random_forest,
                   "xgboost": make_xgboost, "ensemble": make_ensemble}


def available_models() -> list:
    names = []
    for name, factory in MODEL_FACTORIES.items():
        try:
            factory(np.array([0, 0, 1, 1]))
            names.append(name)
        except ImportError:
            print(f"[train] '{name}' is not installed: skipping it (pip install xgboost to enable)")
    return names


def evaluate_frame(df, model, scaler, rule):
    """Score a labelled frame with an already fitted model. Returns (metrics, y_true, y_pred)."""
    y_true = rule.transform(df).map(LABEL_TO_INT).to_numpy()
    X = scaler.transform(df[FEATURE_COLUMNS].to_numpy(dtype=float))
    y_pred, p_hit = model.predict(X), model.predict_proba(X)[:, 1]

    metrics = evaluate_binary(y_true, y_pred)
    metrics["macro_f1"] = (metrics["hit_f1"] + metrics["miss_f1"]) / 2
    metrics["auc"] = float(roc_auc_score(y_true, p_hit)) if len(np.unique(y_true)) == 2 else float("nan")
    metrics.update({k: v for k, v in lift_at(y_true, p_hit).items() if k != "lift_k"})
    return metrics, y_true, y_pred


def fit_and_score(train_df, test_df, model_name="logreg", seed=123, cv=5):
    """Label (rule fitted on train only) -> scale (train only) -> fit -> score on the held-out part."""
    rule = HitMissRule(miss_quantile=LABEL_MISS_QUANTILE)
    y_train = rule.fit_transform(train_df).map(LABEL_TO_INT).to_numpy()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLUMNS].to_numpy(dtype=float))
    model = MODEL_FACTORIES[model_name](y_train, seed, cv).fit(X_train, y_train)

    metrics, y_test, y_pred = evaluate_frame(test_df, model, scaler, rule)
    metrics["train_miss_share"] = float(np.mean(y_train == 0))
    return model, scaler, rule, metrics, y_test, y_pred


def train_and_evaluate(df, model_name="logreg", seed=123, cv=5, test_size=0.2):
    """One author-held-out split (pure sklearn, no MLflow). Expects data from prepare_data."""
    train_df, test_df = split_by_author(df, test_size, seed)
    model, scaler, rule, metrics, y_test, y_pred = fit_and_score(train_df, test_df, model_name, seed, cv)
    metrics["n_train"] = len(train_df)
    metrics["n_test_authors"] = int(test_df[GROUP_COL].nunique())
    return model, scaler, rule, metrics, (y_test, y_pred)


def cross_validated_metrics(df, model_name="logreg", n_splits=5, n_repeats=3, seed=123, cv=5) -> dict:
    """Repeated author-grouped K-fold; the label rule and scaler are re-fitted inside every fold."""
    rows = []
    for train_idx, test_idx in author_folds(df, n_splits, n_repeats, seed):
        try:
            *_, m, _, _ = fit_and_score(df.iloc[train_idx].reset_index(drop=True),
                                        df.iloc[test_idx].reset_index(drop=True), model_name, seed, cv)
        except ValueError:  # degenerate fold (one class missing): skip it, but count it
            continue
        rows.append(m)
    if not rows:
        return {}
    out = {"cv_folds_used": float(len(rows))}
    for key in ["accuracy", "baseline_accuracy", "macro_f1", "auc", "hit_f1", "miss_f1", "hit_recall", "miss_recall",
                "bottom_miss_rate", "top_hit_rate", "base_miss_rate", "base_hit_rate"]:
        vals = [r[key] for r in rows]
        out[f"cv_{key}_mean"], out[f"cv_{key}_std"] = float(np.nanmean(vals)), float(np.nanstd(vals))
    return out


def select_best(cv_by_model: dict) -> str:
    """Highest cross-validated macro-F1; the first (simplest) model wins unless another is clearly better."""
    names = [n for n, m in cv_by_model.items() if m]
    if not names:
        raise ValueError("No model produced usable cross-validation folds.")
    best = names[0]
    for n in names[1:]:
        if cv_by_model[n]["cv_macro_f1_mean"] > cv_by_model[best]["cv_macro_f1_mean"] + MIN_GAIN:
            best = n
    return best


def fit_final(df, model_name, seed=123, cv=5):
    """The model that gets deployed: label rule, scaler and model fitted on ALL usable rows."""
    rule = HitMissRule(miss_quantile=LABEL_MISS_QUANTILE)
    y = rule.fit_transform(df).map(LABEL_TO_INT).to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLUMNS].to_numpy(dtype=float))
    model = MODEL_FACTORIES[model_name](y, seed, cv).fit(X, y)
    return model, scaler, rule


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def make_confusion_figure(y_test, y_pred):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, labels=[0, 1]),
                           display_labels=["Miss", "Hit"]).plot(cmap=plt.cm.Blues, ax=ax)
    return fig


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def save_local_outputs(metrics: dict, fig) -> None:
    try:
        os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)
        fig.savefig(os.path.join(CONFUSION_MATRIX_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
        os.makedirs(METRICS_DIR, exist_ok=True)
        _write_json(os.path.join(METRICS_DIR, "train_metrics.json"), metrics)
    except OSError as exc:  # e.g. read-only container filesystem: MLflow still has everything
        print(f"[train] could not write local output files ({exc}); continuing")


def print_comparison(cv_by_model: dict, chosen: str) -> None:
    for name, m in cv_by_model.items():
        if not m:
            print(f"[train] {name}: no usable folds")
            continue
        print(f"[train] {name:8s} author-grouped CV ({int(m['cv_folds_used'])} folds): "
              f"accuracy {m['cv_accuracy_mean']:.3f} +/- {m['cv_accuracy_std']:.3f} "
              f"vs baseline {m['cv_baseline_accuracy_mean']:.3f} | macro-F1 {m['cv_macro_f1_mean']:.3f} | "
              f"AUC {m['cv_auc_mean']:.3f}" + ("   <- chosen" if name == chosen else ""))


def print_holdout(m: dict) -> None:
    print(f"[train] HOLD-OUT = newest books of authors the model trained on ({int(m['n_eval'])} books by "
          f"{int(m['n_holdout_authors'])} authors): accuracy {m['accuracy']:.3f} vs baseline {m['baseline_accuracy']:.3f} "
          f"| macro-F1 {m['macro_f1']:.3f} | Miss F1 {m['miss_f1']:.3f} | AUC {m['auc']:.3f}")
    print(f"[train] usefulness on the hold-out: of the 20% of books rated LEAST likely to be Hits, "
          f"{m['bottom_miss_rate']:.0%} are real Misses (base rate {m['base_miss_rate']:.0%}); of the 20% rated MOST "
          f"likely, {m['top_hit_rate']:.0%} are real Hits (base rate {m['base_hit_rate']:.0%})")


def print_cv_usefulness(cv: dict) -> None:
    if cv:
        print(f"[train] usefulness on unseen authors (CV): bottom 20% are real Misses {cv['cv_bottom_miss_rate_mean']:.0%} "
              f"(base {cv['cv_base_miss_rate_mean']:.0%}) | top 20% are real Hits {cv['cv_top_hit_rate_mean']:.0%} "
              f"(base {cv['cv_base_hit_rate_mean']:.0%})")


# --------------------------------------------------------------------------- #
# MLflow
# --------------------------------------------------------------------------- #
def build_and_register_model(df, seed=123, cv=5, models=None):
    import matplotlib.pyplot as plt
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException
    from mlflow.models import infer_signature

    data = prepare_data(df)
    train_part, holdout = split_latest_per_author(data, HOLDOUT_LATEST_SHARE)
    print(f"[train] hold-out: the newest {len(holdout)} books of {holdout[GROUP_COL].nunique()} authors are set aside; "
          f"training on {len(train_part)} books")

    names = models or available_models()
    cv_by_model = {n: cross_validated_metrics(train_part, n, seed=seed, cv=cv) for n in names}   # hold-out never used here
    chosen = select_best(cv_by_model)
    print_comparison(cv_by_model, chosen)
    print_cv_usefulness(cv_by_model[chosen])

    # The deployed model = fitted on the training part only, then scored on the hold-out.
    model, scaler, rule, metrics, y_test, y_pred = fit_and_score(train_part, holdout, chosen, seed, cv)
    metrics.update(n_train=float(len(train_part)), n_holdout_authors=float(holdout[GROUP_COL].nunique()))
    metrics.update(cv_by_model[chosen])
    for n, m in cv_by_model.items():   # keep every candidate's score for the record
        if m:
            metrics[f"cv_{n}_macro_f1"], metrics[f"cv_{n}_accuracy"] = m["cv_macro_f1_mean"], m["cv_accuracy_mean"]
    print_holdout(metrics)

    fig = make_confusion_figure(y_test, y_pred)
    save_local_outputs(metrics, fig)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()
    try:
        client.get_registered_model(REGISTERED_MODEL_NAME)
    except MlflowException:
        client.create_registered_model(name=REGISTERED_MODEL_NAME)

    with mlflow.start_run() as run:
        mlflow.set_tags({"feature_group": f"{FEATURE_GROUP_NAME}:v{FEATURE_GROUP_VERSION}",
                         "model_type": chosen,
                         "label": "reach vs the author's own earlier books (labeling.HitMissRule)"})
        params = {"model_type": chosen, "seed": seed, "cv": cv,
                  "n_rows": len(data), "n_train_rows": len(train_part), "n_holdout_rows": len(holdout),
                  "holdout_share": HOLDOUT_LATEST_SHARE, "n_authors": int(data[GROUP_COL].nunique()),
                  "n_features": len(FEATURE_COLUMNS), "features": ",".join(FEATURE_COLUMNS),
                  "miss_quantile": LABEL_MISS_QUANTILE, "min_prior_books": MIN_PRIOR_BOOKS,
                  "label_cutoff": rule.cutoff}
        try:
            params["chosen_C"] = float(np.ravel(model.C_)[0])
        except AttributeError:   # not a logistic regression
            pass
        if hasattr(model, "named_estimators_"):   # the ensemble: record who voted
            params["ensemble_members"] = ",".join(model.named_estimators_)
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(v)})
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # One run = one self-consistent bundle: model + scaler + label rule + column order.
        with tempfile.TemporaryDirectory() as tmp:
            for filename, write in (
                ("scaler.joblib", lambda p: joblib.dump(scaler, p)),
                ("label_rule.json", rule.save),
                ("train_metrics.json", lambda p: _write_json(p, metrics)),
                ("holdout_books.json", lambda p: _write_json(p, {
                    "book_ids": holdout["book_id"].astype(str).tolist(), "share": HOLDOUT_LATEST_SHARE,
                    "rule": "newest books of each author (by publication year); never used for fitting"})),
            ):
                path = os.path.join(tmp, filename)
                write(path)
                mlflow.log_artifact(path)
        mlflow.log_dict({"feature_columns": FEATURE_COLUMNS, "outcome_columns": OUTCOME_COLUMNS},
                        "feature_columns.json")

        example = scaler.transform(train_part[FEATURE_COLUMNS].head(5).to_numpy(dtype=float))
        mlflow.sklearn.log_model(model, "model", signature=infer_signature(example, model.predict(example)),
                                 input_example=example)

        run_id = run.info.run_id
        version = client.create_model_version(
            name=REGISTERED_MODEL_NAME, source=f"runs:/{run_id}/model", run_id=run_id)
        promote_to_production(client, REGISTERED_MODEL_NAME, version.version)
        print(f"[train] run {run_id} -> {REGISTERED_MODEL_NAME} v{version.version} ({chosen})")
    return run_id, metrics


def main():
    build_and_register_model(fetch_df_from_feature_store())


if __name__ == "__main__":
    main()