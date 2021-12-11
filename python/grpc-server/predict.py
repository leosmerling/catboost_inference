from typing import List
from time import time_ns

import numpy as np
import catboost as cb

from cb_pb2 import PredictRequest, PredictResponse, Features, Prediction

model = cb.CatBoost()
model.load_model(fname="model.cbm")

def preprocess(features: List[Features]) -> np.array:
    return np.ascontiguousarray(
        [
            [f.cat_feature1, f.cat_feature2, f.cat_feature3, f.float_feature4, f.float_feature5]
            for f in features
        ], dtype=object
    )


def predict(request: PredictRequest):
    start = time_ns()
    features = preprocess(request.features)
    prep_lat = time_ns() - start

    start = time_ns()
    scores = model.predict(features, thread_count=1)
    model_lat = time_ns() - start
    
    return PredictResponse(
        predictions=[
            Prediction(score=s) for s in scores
        ],
        preprocess_latency=prep_lat,
        model_latency=model_lat
    )
