"""
Hit / Miss label, built from the OUTCOME and fitted on TRAINING rows only.

  performance = log10(1 + readers who shelved the book)  -  author's typical past performance
                                                            (median log-reach of their EARLIER books)
  Miss        = performance in the bottom `miss_quantile` of the TRAINING set
                AND the book's own reach is below the typical (median) book of the TRAINING set
  Hit         = everything else

The second condition stops a famous book from being called a Miss just because its author's other
books are even bigger (a Jane Austen title with 20k readers is not a flop).

"Did this book meet the bar its own author usually sets?"  A publisher can use the prediction
to decide where early marketing money is needed.

* The author baseline (prior_median_log_reach) is known before launch and is computed from
  earlier books only (see author_history.py), so a book never defines its own baseline.
* The cut-off is learned from the training rows and saved as JSON next to the model; held-out
  and inference rows are labelled with that saved number, never re-derived from their own batch.
* Rows with missing values raise an error; they are never silently called "Miss".

Honest limit: reach is today's reading-log total, so older books have had longer to collect it.
book_published_year is a model feature so the model can account for that.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

OUTCOME_COL = "readinglog_count"
BASELINE_COL = "prior_median_log_reach"
REQUIRED = [OUTCOME_COL, BASELINE_COL]


@dataclass
class HitMissRule:
    miss_quantile: float = 0.4
    cutoff: float | None = None
    abs_quantile: float = 0.5          # "low absolute reach" = below this quantile of training reach
    abs_cutoff: float | None = None    # None (older saved models) = relative rule only

    @staticmethod
    def performance(df: pd.DataFrame) -> pd.Series:
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"Label needs columns {missing}")
        if df[REQUIRED].isna().any().any():
            raise ValueError("Rows with missing outcome/baseline cannot be labelled; drop them "
                             "explicitly first (never default them to 'Miss').")
        return pd.Series(np.log10(1.0 + df[OUTCOME_COL].astype(float)) - df[BASELINE_COL].astype(float),
                         index=df.index)

    def fit_transform(self, train_df: pd.DataFrame) -> pd.Series:
        perf = self.performance(train_df)
        self.cutoff = float(perf.quantile(self.miss_quantile))
        reach = self.log_reach(train_df)
        self.abs_cutoff = float(reach.quantile(self.abs_quantile))
        return self._to_labels(perf, reach)

    def transform(self, df: pd.DataFrame) -> pd.Series:
        if self.cutoff is None:
            raise RuntimeError("Rule is not fitted. Call fit_transform(train_df) or load().")
        return self._to_labels(self.performance(df), self.log_reach(df))

    @staticmethod
    def log_reach(df: pd.DataFrame) -> pd.Series:
        return pd.Series(np.log10(1.0 + df[OUTCOME_COL].astype(float)), index=df.index)

    def _to_labels(self, perf: pd.Series, reach: pd.Series) -> pd.Series:
        miss = perf < self.cutoff
        if self.abs_cutoff is not None:
            miss &= reach < self.abs_cutoff
        return pd.Series(np.where(miss, "Hit", "Hit"), index=perf.index, name="hit_miss").mask(miss, "Miss")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "HitMissRule":
        with open(path) as f:
            return cls(**json.load(f))