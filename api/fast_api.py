from fastapi import FastAPI
import pandas as pd

from pydantic import BaseModel, Field
class BookInput(BaseModel):
    description: str
    author_avg_rating: float = Field(..., ge=0.0, le=5.0)
    author_rating_count: int = Field(..., ge=0)
    author_total_reviews: int = Field(..., ge=0)
    num_pages: int = Field(..., ge=1)
    book_published_year: int = Field(..., ge=0)
    book_avg_rating: float = Field(..., ge=0.0, le=5.0)
    book_ratings: int = Field(..., ge=0)
    book_reviews: int = Field(..., ge=0)

app = FastAPI()
mlflow.set_tracking_uri("file:///app/mlruns")
from mlflow.tracking import MlflowClient
import mlflow.sklearn

client = MlflowClient()
experiment = client.get_experiment_by_name("LR_Experiment")
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])

latest_run_id = runs[0].info.run_id
model = mlflow.sklearn.load_model(f"mlruns/{latest_run_id}/artifacts/model")

buzzwords = ['award', 'bestseller', 'classic', 'legendary', 'masterpiece', 
             'epic', 'thrilling', 'captivating', 'page-turner', 'unforgettable']

def extract_features_single(book: dict) -> dict:
    desc = book['description'].lower()
    book['buzzword_count'] = sum(desc.count(word) for word in buzzwords)
    book['desc_len_words'] = len(desc.split())
    book['num_sentences'] = desc.count('.') + desc.count('!') + desc.count('?')
    import re
    book['num_adjectives'] = len(re.findall(r'\b\w+(ly|ous|ive|able|ful|ish|ic|al|ant|ent)\b', desc))
    return book

@app.post("/predict/")
def predict(book: BookInput):
    book_dict = book.dict()
    
    # Extract text-based features
    book_features = extract_features_single(book_dict)
    
    feature_columns = [
        'buzzword_count', 'desc_len_words', 'num_sentences', 'num_adjectives',
        'author_avg_rating', 'author_rating_count', 'author_total_reviews',
        'num_pages', 'book_published_year', 'book_avg_rating', 'book_ratings', 'book_reviews'
    ]
    X = pd.DataFrame([book_features])[feature_columns]
    
    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, 1][0]
    
    book_features['predicted_hit'] = 'Hit' if prediction[0] == 1 else 'Miss'
    book_features['predicted_probability'] = float(probability)
    
    return book_features
