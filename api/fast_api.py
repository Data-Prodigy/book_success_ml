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

app = FastAPI()import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://mlflow:5000")
run_id = "659f77bea0694eb1a3c3cbe3819d0e34" 
model_uri = f"runs:/{run_id}/model"  
model = mlflow.sklearn.load_model(model_uri)

buzzwords = ['award', 'bestseller', 'classic', 'legendary', 'masterpiece', 
             'epic', 'thrilling', 'captivating', 'page-turner', 'unforgettable']

def extract_features_single(book: dict) -> dict:
    desc = book['description'].replace('\\"', '"').replace("\\'", "'").lower()

    book['buzzword_count'] = sum(desc.count(word) for word in buzzwords)
    book['desc_len_words'] = len(desc.split())
    book['num_sentences'] = desc.count('.') + desc.count('!') + desc.count('?')
    
    import re
    book['num_adjectives'] = len(
        re.findall(r'\b\w+(ly|ous|ive|able|ful|ish|ic|al|ant|ent)\b', desc)
    )
    return book

@app.post("/predict/")
def predict(book: BookInput):
    from sklearn.preprocessing import StandardScaler
    book_dict = book.dict()
    
    # Extract text-based features
    book_features = extract_features_single(book_dict)
    
    feature_columns = [
        'buzzword_count', 'desc_len_words', 'num_sentences', 'num_adjectives',
        'author_avg_rating', 'author_rating_count', 'author_total_reviews',
        'num_pages', 'book_published_year', 'book_avg_rating', 'book_ratings', 'book_reviews'
    ]
    X = pd.DataFrame([book_features])[feature_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    prediction = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)[:, 1][0]
    
    book_features['predicted_hit'] = 'Hit' if prediction[0] == 1 else 'Miss'
    
    return book_features
