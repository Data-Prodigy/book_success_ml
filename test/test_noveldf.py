import pandas as pd
import pytest
import sys
import os

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from src.features.feature_pipeline import create_hit_miss

def test_compute_hit_miss_basic():
    df = pd.DataFrame({
        "book_avg_rating": [4.5, 3.2],
        "book_reviews": [250, 50]
    })
    result = create_hit_miss(df,quantile=0.35)
    assert set(result['hit_miss'].unique()) == {"Hit", "Miss"}



from hopsworks import login
API_KEY = "NhIODCOOM50hfmRh.f8KWceqZECE4l3Id7GAr47u5MAw9Wn28KWFRU9JNXFm1oVh4bdDtN5sRigsS9cD8"


def test_read_feature_group_to_pandas():
    project = login(api_key_value=API_KEY)
    fs = project.get_feature_store()
    feature_group = fs.get_feature_group("novel_data", version=1)
    df = feature_group.read()  
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    
