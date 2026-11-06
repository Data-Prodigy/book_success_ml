"""Metrics shared by training and inference, with the class convention made explicit.

Positive class = "Hit" (1). The earlier code used LabelEncoder, which sorts
alphabetically (Hit=0, Miss=1), so precision/recall/F1 were silently computed
for "Miss". For a publisher deciding where NOT to waste campaign money, the
Miss-side numbers matter just as much, so both are reported.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate_binary(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    hit_share = float(np.mean(y_true == 1))
    return {
        "n_eval": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Always-guess-the-majority-class accuracy: the number the model has to beat.
        "baseline_accuracy": max(hit_share, 1.0 - hit_share),
        "hit_precision": float(p[1]), "hit_recall": float(r[1]), "hit_f1": float(f[1]),
        "miss_precision": float(p[0]), "miss_recall": float(r[0]), "miss_f1": float(f[0]),
    }


def lift_at(y_true, p_hit, frac: float = 0.2) -> dict:
    """Is the ranking useful for triage? Look only at the books the model is most sure about.

    bottom_miss_rate: of the `frac` of books given the LOWEST hit probability, the share that really were Misses
    top_hit_rate:     of the `frac` of books given the HIGHEST hit probability, the share that really were Hits
    Compare each with the base rate (what you'd get by picking books at random).
    """
    y_true, p = np.asarray(y_true), np.asarray(p_hit, dtype=float)
    k = max(1, int(round(len(y_true) * frac)))
    order = np.argsort(p, kind="stable")
    base_miss = float(np.mean(y_true == 0))
    return {
        "bottom_miss_rate": float(np.mean(y_true[order[:k]] == 0)),
        "top_hit_rate": float(np.mean(y_true[order[-k:]] == 1)),
        "base_miss_rate": base_miss, "base_hit_rate": 1.0 - base_miss, "lift_k": k,
    }