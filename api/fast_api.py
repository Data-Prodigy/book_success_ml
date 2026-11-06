"""
FastAPI service: "how will this author's next book do?"   (port 8085 in docker-compose)

POST /predict/            author_key + year (+ pages)  ->  Hit / Miss + probability
GET  /authors/{key}       is this author in the catalogue, and how many books do we know?
GET  /holdout             the real held-out books of trained authors: prediction next to what really happened
POST /reload              re-read the live model and the catalogue (no restart needed)
GET  /health

Model + scaler + label rule come from MLflow's `production` alias; the author's history comes from the
feature store. The feature code is the same function training used (author_history.compute_history).
"""
import json
from contextlib import asynccontextmanager
from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.features.inference_pipeline import CannotScore, author_info, holdout_summary, load_history, predict_one, score_holdout
from src.features.model_registry import load_production_bundle

state: dict = {}

MAX_YEARS_AHEAD = 5   # no predictions further into the future than this


def load_state() -> None:
    state["bundle"] = load_production_bundle()
    state["history"] = load_history()
    try:   # real hold-out books (never fitted); None for models registered before hold-out tracking
        state["holdout"] = score_holdout(state["history"], state["bundle"])
    except RuntimeError:
        state["holdout"] = None
    print(f"Loaded model version {state['bundle']['version']} (run {state['bundle']['run_id']}) "
          f"and {len(state['history'])} catalogue books")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_state()
    yield
    state.clear()


app = FastAPI(title="Novel success predictor", lifespan=lifespan)


class BookInput(BaseModel):
    """Only what a publisher knows BEFORE launch. Reader outcomes are never inputs."""
    author_key: str = Field(..., examples=["OL19981A"], description="Open Library author id")
    book_published_year: int = Field(..., ge=1800, description=f"Planned publication year (at most {MAX_YEARS_AHEAD} years from now)")
    num_pages: Optional[int] = Field(None, ge=1, description="If omitted, the author's median is used")
    genre: Optional[str] = Field(None, examples=["thriller"], description=(
        "thriller | mystery_crime | romance | fantasy | scifi | horror | historical | literary | "
        "young_adult | other. If omitted, the author's most common genre is used"))
    title: Optional[str] = None

    @field_validator("book_published_year")
    @classmethod
    def not_too_far_ahead(cls, year: int) -> int:
        latest = date.today().year + MAX_YEARS_AHEAD
        if year > latest:
            raise ValueError(f"book_published_year must be {latest} or earlier (at most {MAX_YEARS_AHEAD} years ahead)")
        return year


@app.post("/predict/")
def predict(book: BookInput):
    payload = book.model_dump() if hasattr(book, "model_dump") else book.dict()
    try:
        result = predict_one(payload, state["bundle"], state["history"])
    except CannotScore as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {**result, "model_version": state["bundle"]["version"]}


@app.get("/authors/{author_key}")
def author(author_key: str):
    info = author_info(state["history"], author_key)
    if info is None:
        raise HTTPException(status_code=404, detail=f"{author_key} is not in the catalogue")
    return info


@app.get("/holdout")
def holdout(author_key: Optional[str] = None, limit: int = 50):
    """Books the model never saw during training (by authors it did train on): prediction vs reality."""
    scored = state.get("holdout")
    if scored is None:
        raise HTTPException(status_code=404, detail="This model has no recorded hold-out books; retrain.")
    d = scored.dropna(subset=["predicted_hit_miss"])
    if author_key:
        d = d[d["author_key"] == author_key]
    cols = [c for c in ["author_key", "author_name", "title", "book_published_year", "predicted_hit_miss",
                        "hit_probability", "actual_hit_miss", "correct"] if c in d]
    rows = d[cols].head(limit)
    return {"books": len(d), "accuracy": float(d["correct"].astype(float).mean()) if len(d) else None,
            "rows": json.loads(rows.to_json(orient="records"))}


@app.post("/reload")
def reload():
    load_state()
    return {"status": "reloaded", "model_version": state["bundle"]["version"],
            "catalogue_books": len(state["history"])}


@app.get("/health")
def health():
    return {"status": "ok" if state else "loading",
            "model_version": state.get("bundle", {}).get("version")}