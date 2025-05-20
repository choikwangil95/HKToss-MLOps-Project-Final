import pickle
import pandas as pd
import numpy as np

MODEL_PATH = './ml_models/ko-sbert-sts_model.pk'
DATA_PATH = './datas/mk_news_with_id_embedding.csv'

_model = None
_news_df = None

def get_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
    return _model

def get_news_df():
    global _news_df
    if _news_df is None:
        df = pd.read_csv(DATA_PATH)
        df['embedding'] = df['embedding'].apply(lambda x: np.array(list(map(float, x.split(',')))))
        _news_df = df

    return _news_df
