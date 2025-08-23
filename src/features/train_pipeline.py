def fetch_df_from_feature_store():
    import hopsworks
    import pandas as pd 

    api_key = 'NhIODCOOM50hfmRh.f8KWceqZECE4l3Id7GAr47u5MAw9Wn28KWFRU9JNXFm1oVh4bdDtN5sRigsS9cD8'
    project = hopsworks.login(api_key_value=api_key)  
    fs = project.get_feature_store()

    feature_group = fs.get_feature_group(
        name="novel_data_v1",  
        version=1
    )

    df = feature_group.read()  
    return df


def build_LR_model(df):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import matplotlib.pyplot as plt
    import mlflow
    from mlflow import MlflowClient
    import os
    import json
    mlflow.set_tracking_uri("file:///app/mlruns")
    mlflow.set_experiment('LR_Experiment_v2')
    mlflow.sklearn.autolog()

    seed = 123
    cv = 5

    y = df['hit_miss']

    numeric_cols = [
        'book_avg_rating', 'book_ratings', 'book_reviews',
        'author_avg_rating', 'author_rating_count', 'author_total_reviews',
        'buzzword_count', 'num_sentences', 'num_adjectives','desc_len_words',
        'book_published_year','num_pages'
    ]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_numeric = df[numeric_cols].copy()
    X = X_numeric.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    client = MlflowClient()
    try:
        client.create_registered_model(name='logreg_cv')
    except:
        pass  

    with mlflow.start_run() as run:
        logreg_cv = LogisticRegressionCV(
            cv=cv,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            refit=True
        )

        logreg_cv.fit(X_train, y_train)
        y_pred = logreg_cv.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-score: {f1:.3f}")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.close(fig)

        save_dir = "confusion_matrix"
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')

        #path to save the mertrics
        metrics = { 
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        save_dir = "inference_metrics"
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "train_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)

        # Registering the Model to MLFLOW
        mlflow.sklearn.log_model(logreg_cv, "model")
        run_id = run.info.run_id

def main():
    novel_df = fetch_df_from_feature_store()
    build_LR_model(novel_df)

if __name__=="__main__":
    main()