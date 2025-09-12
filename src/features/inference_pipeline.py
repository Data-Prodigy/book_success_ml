import mlflow
import mlflow.sklearn
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import os

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  
if src_path not in sys.path:
    sys.path.insert(0, src_path)


from mlflow.tracking import MlflowClient
model = None

def save_metrics(metrics_dict, filename='inference_metrics.json'):
    with open(filename, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

def get_inference_data(filename):
    import pandas as pd 
    df = pd.read_csv(filename)  
    from features.feature_pipeline import extract_features, create_hit_miss
    df = extract_features(df)
    df = create_hit_miss(df,quantile=0.35)
    return df

def main():
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    import mlflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    seed = 123
    inference_df = get_inference_data('novel_inference_data.csv')
    inference_df.drop(columns=['description'], inplace=True)

    y = inference_df['hit_miss']

    cols_to_norm = [
        'author_avg_rating', 
        'author_rating_count', 
        'author_total_reviews', 
        'book_avg_rating', 
        'num_pages',
        'book_ratings',
        'book_reviews',
        'buzzword_count',
        'desc_len_words', 
        'num_sentences',
       'num_adjectives'
    ]

    numeric_cols = cols_to_norm 

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_numeric = inference_df[numeric_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    X_train,X_test,y_train,y_test = train_test_split(X_scaled,y_encoded,test_size=0.2,stratify=y_encoded,random_state=seed)
    global model
    if model is None:
        import mlflow.sklearn
        run_id = "659f77bea0694eb1a3c3cbe3819d0e34" 
        model_uri = f"runs:/{run_id}/model"  
        model = mlflow.sklearn.load_model(model_uri)
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
    }

    print("Inference Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    save_metrics(metrics)

if __name__ == "__main__":
    main()
