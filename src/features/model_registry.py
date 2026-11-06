"""
Everything that talks to the MLflow model registry, in one place.

  promote_to_production()  - called by training_pipeline.py after a new model is logged
  load_production_bundle() - called by inference_pipeline.py and fastapi_app.py

Nobody hardcodes a run_id. "Which model is live" is the `production` ALIAS (MLflow's
current mechanism); the legacy "Production" stage is kept in sync on a best-effort basis
and is the fallback for registries that predate the alias.

mlflow is imported lazily inside functions, so importing this module (e.g. in unit tests
or CI) does not require mlflow to be installed.
"""
import json
import time
import warnings

import joblib

from config import (
    MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME, MODEL_STAGE, MODEL_ALIAS,
)
from labeling import HitMissRule


def wait_until_ready(client, name: str, version, timeout: float = 60.0, poll: float = 1.0) -> None:
    """create_model_version returns while the version is still PENDING_REGISTRATION."""
    deadline = time.time() + timeout
    while True:
        status = str(client.get_model_version(name, str(version)).status)
        if status == "READY":
            return
        if "FAILED" in status:
            raise RuntimeError(f"Registering {name} v{version} failed (status {status}).")
        if time.time() > deadline:
            raise TimeoutError(f"{name} v{version} was still {status} after {timeout:.0f}s.")
        time.sleep(poll)


def promote_to_production(client, name: str, version) -> None:
    """Make `version` the live model; the previously live version is archived."""
    version = str(version)
    wait_until_ready(client, name, version)

    client.set_registered_model_alias(name, MODEL_ALIAS, version)

    try:  # legacy stages: deprecated in MLflow >= 2.9, so never let them break the run
        for mv in client.search_model_versions(f"name='{name}'"):
            if mv.current_stage == MODEL_STAGE and str(mv.version) != version:
                client.transition_model_version_stage(name=name, version=str(mv.version), stage="Archived")
        client.transition_model_version_stage(name=name, version=version, stage=MODEL_STAGE)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Could not update the legacy '{MODEL_STAGE}' stage ({exc}). "
            f"The '{MODEL_ALIAS}' alias is set, and that is what the loader uses."
        )
    print(f"Promoted {name} v{version} (alias '{MODEL_ALIAS}')")


def resolve_production_version(client, name: str = REGISTERED_MODEL_NAME):
    """The live ModelVersion: by alias first, then by legacy stage."""
    problems = []
    try:
        return client.get_model_version_by_alias(name, MODEL_ALIAS)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"alias '{MODEL_ALIAS}': {exc}")
    try:
        versions = client.get_latest_versions(name, stages=[MODEL_STAGE])
        if versions:
            return versions[0]
        problems.append(f"stage '{MODEL_STAGE}': no versions")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"stage '{MODEL_STAGE}': {exc}")
    raise RuntimeError(
        f"No live model found for '{name}' at {MLFLOW_TRACKING_URI}. Has training_pipeline.py "
        f"run, and is the MLflow server reachable? Details: " + " | ".join(problems)
    )


def load_production_bundle() -> dict:
    """model + scaler + label rule + column order, all from the SAME training run."""
    import mlflow
    import mlflow.artifacts
    import mlflow.sklearn
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    mv = resolve_production_version(client)
    run_id = mv.run_id

    def artifact(path: str) -> str:
        return mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=path, tracking_uri=MLFLOW_TRACKING_URI
        )

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    scaler = joblib.load(artifact("scaler.joblib"))
    label_rule = HitMissRule.load(artifact("label_rule.json"))
    with open(artifact("feature_columns.json")) as f:
        feature_columns = json.load(f)["feature_columns"]  # the order THIS model was trained on

    try:   # which books were held out of training; absent for models registered before hold-out tracking
        with open(artifact("holdout_books.json")) as f:
            holdout_ids = json.load(f)["book_ids"]
    except Exception:  # noqa: BLE001
        holdout_ids = None

    return {
        "holdout_book_ids": holdout_ids,
        "model": model,
        "scaler": scaler,
        "label_rule": label_rule,
        "feature_columns": feature_columns,
        "run_id": run_id,
        "version": mv.version,
    }