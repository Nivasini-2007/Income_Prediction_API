
from fastapi import FastAPI, HTTPException
from typing import Any
import pandas as pd
import joblib
import os

app = FastAPI(
    title="Adult Income Prediction API",
    description="Real-time ML inference using the champion model",
    version="1.0.0"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "champion_model.joblib"
)

model = joblib.load(MODEL_PATH)

FEATURES = [
    "age", "workclass", "fnlwgt", "education",
    "education-num", "marital-status", "occupation",
    "relationship", "race", "sex", "capital-gain",
    "capital-loss", "hours-per-week", "native-country"
]

@app.get("/")
def home():
    return {
        "message": "Adult Income Prediction API is running",
        "status": "success"
    }

@app.post("/predict")
def predict(data: dict[str, Any]):

    missing = [f for f in FEATURES if f not in data]

    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {missing}"
        )

    try:
        input_data = pd.DataFrame(
            [[data[f] for f in FEATURES]],
            columns=FEATURES
        )

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        return {
            "prediction": int(prediction),
            "income_class": ">50K" if int(prediction) == 1 else "<=50K",
            "probabilities": {
                str(label): float(prob)
                for label, prob in zip(
                    model.classes_, probabilities
                )
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
