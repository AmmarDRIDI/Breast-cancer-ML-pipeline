from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Breast Cancer Risk Classifier")
model = joblib.load("artifacts/model.joblib")


class InferenceInput(BaseModel):
    features: list[float]


@app.post("/predict")
def predict(payload: InferenceInput):
    x = np.array(payload.features, dtype=float).reshape(1, -1)
    proba = float(model.predict_proba(x)[0, 1])
    pred = int(model.predict(x)[0])
    return {"prediction": pred, "probability_malignant": proba}
