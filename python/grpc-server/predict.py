import catboost as cb

from cb_pb2 import PredictRequest, PredictResponse

model = cb.CatBoost()
model.load_model(fname="model.cbm")

def predict(request: PredictRequest):
    score = model.predict([[
        request.cat_feature1, request.cat_feature2, request.cat_feature3,
        request.float_feature4, request.float_feature5
    ]], thread_count=1)[0]
    return PredictResponse(score=score)
