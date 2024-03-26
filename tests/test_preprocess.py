import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from src.preprocess import Data_Preprocess
import numpy as np


def test_get_X_y():
    """Test get_X_y function"""
    df = pd.DataFrame({"date": ['2023-12-08', '2023-12-31','2023-12-29'], 
                       "search_query": ['iphone 8', 'maps google', 'polystichum dahlem'],
                       "market":['de-de','de-de','en-gb'],
                       "geo_country": ['AT', 'DE', np.nan],
                       "device_type":['Mobile','Tablet','Desktop'],
                       "browser_name":['Chromium','Chromium','Brave'],
                       "intent":['Shopping','OTHER','OTHER'],
                       "query_count":[2,2,2]
                      })
    

    target = "intent"
    preprocess=Data_Preprocess()
    df=preprocess.handle_nan(df,target=target)
    df=preprocess.feature_engineer(df)
    id_to_intent=preprocess.id_to_intent(df)
    X,y=preprocess.get_labels(df)
    print(X.columns)

    expected_X = pd.DataFrame(
                        {
                        "search_query": ['iphone 8', 'maps google', 'polystichum dahlem'],
                        "market":['de-de','de-de','en-gb'],
                        "geo_country": ['other', 'DE', 'other'],
                        "device_type":['Mobile','other','Desktop'],
                        "browser_name":['Chromium','Chromium','other'],
                        "query_count":[2,2,2]

                        })
    
    expected_y = pd.Series([0,1,1])
    
    assert_frame_equal(X, expected_X)
    assert_series_equal(y, expected_y, check_names=False)