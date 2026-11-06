"""
Shared configuration for the Novel Success ML System.

Single source of truth for: secrets (read from the environment / .env),
the feature-store + registry names, and — most importantly — which
columns are PREDICTORS and which are OUTCOMES.

Secrets
-------
Locally, put them in a `.env` file next to this file (see .env.example;
`.env` must be git-ignored). In Docker Compose use `env_file: .env`. In
GitHub Actions use repository secrets. Real environment variables always
win over .env, so the same code works in all three places.

Secrets are read lazily: `from config import HOPSWORKS_API_KEY` only
fails if that particular key is missing, so e.g. the FastAPI container
does not need the Google Books key just to import this module.
"""
import os

from features.genre import GENRE_COLUMNS

try:  # python-dotenv is optional: in containers the env is injected directly
    # NOTE: the pip package is "python-dotenv" but the module is "dotenv".
    from dotenv import find_dotenv, load_dotenv

    # find_dotenv() walks UP from this file's folder until it finds a .env, so it
    # works whether config.py is in the project root or in src/.
    load_dotenv(find_dotenv(), override=False)
except ImportError:  # pragma: no cover
    pass


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. Copy .env.example "
            f"to .env and fill it in (or inject it via docker-compose "
            f"`env_file:` / CI secrets)."
        )
    return value


_SECRET_NAMES = ("GOOGLE_BOOKS_API_KEY", "HOPSWORKS_API_KEY")


def __getattr__(name: str):  # PEP 562: lazy module attributes
    if name in _SECRET_NAMES:
        return require_env(name)
    raise AttributeError(f"module 'config' has no attribute {name!r}")


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# --- Hopsworks ----------------------------------------------------------------
# Your project lives on the eu-west cloud, not the default host, so the host is explicit.
HOPSWORKS_HOST = os.environ.get("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT", "book_success_ml")

# --- Feature store -----------------------------------------------------------
# One row per book: author-history features + the outcomes readers produced AFTER launch.
# The label is NOT stored; it is derived at training time from the training split.
FEATURE_GROUP_NAME = "author_book_history"
FEATURE_GROUP_VERSION = 3   # v2: cadence+momentum; v3: pages_vs_author_median, genre features

# --- Model registry ----------------------------------------------------------
EXPERIMENT_NAME = "LR_Experiment_v4_author_history"
REGISTERED_MODEL_NAME = "logreg_cv"
# "Which model is live" is an ALIAS (MLflow's current mechanism; stages are deprecated).
# The legacy Production stage is still kept in sync on a best-effort basis.
MODEL_ALIAS = "production"
MODEL_STAGE = "Production"

# --- Local output folders (same names the original scripts used) -------------
CONFUSION_MATRIX_DIR = "confusion_matrix"
METRICS_DIR = "inference_metrics"

# --- Columns -----------------------------------------------------------------
# PREDICTORS: things known BEFORE a book launches (author history built from EARLIER books only).
FEATURE_COLUMNS = [
    "prior_n_books", "prior_mean_log_reach", "prior_median_log_reach", "prior_max_log_reach",
    "last_book_log_reach", "years_since_last_book", "author_career_years",
    "avg_years_between_books", "last3_mean_log_reach",
    "pages_vs_author_median", "author_genre_share",
    "book_published_year", "num_pages",
] + GENRE_COLUMNS
HOLDOUT_LATEST_SHARE = 0.2   # newest 20% of EACH author's books are never fitted: real unseen books of known authors
MIN_PRIOR_BOOKS = 2   # a book needs at least this many earlier books to be trained / scored on

# OUTCOMES: how readers responded AFTER launch. Used for the label and nothing else.
OUTCOME_COLUMNS = ["readinglog_count", "want_to_read_count", "edition_count"]

ID_COLUMNS = ["book_id", "author_key", "author_name", "title", "genre"]
STORED_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + OUTCOME_COLUMNS

# --- Label (see labeling.py) -------------------------------------------------
TARGET_COLUMN = "hit_miss"   # derived at training time; NOT stored in the feature group
LABEL_MISS_QUANTILE = 0.3    # bottom 30% of "reach vs. the author's own past" AND below-typical absolute reach = Miss

# Fixed class encoding: Miss=0, Hit=1 (Hit is the positive class).
LABEL_TO_INT = {"Miss": 0, "Hit": 1}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

# Hard guard: the leakage bug can never come back through a config edit.
_overlap = set(FEATURE_COLUMNS) & set(OUTCOME_COLUMNS)
assert not _overlap, f"Outcome columns used as features (target leakage): {sorted(_overlap)}"