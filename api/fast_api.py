from fastapi import FastAPI
from def_class import BookInput
import pandas as pd
import mlflow.sklearn

app = FastAPI()

# Load your trained model
model_name = 'logreg_cv'
stage = 'Production'
model_uri = f"models:/{model_name}/{stage}"
model = mlflow.sklearn.load_model(model_uri)

# Feature extraction function (single book)
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
