import pickle
import pandas as pd
import numpy as np

MODEL_PATH = "./ml_models/ko-sbert-sts_model.pk"

_model = None


def get_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model
