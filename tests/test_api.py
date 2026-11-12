
import sys
sys.path.insert(0, "/content/Real_Time_ML_API")

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

sample_data = {
    "age": 39,
    "workclass": "Private",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

def test_home():
    response = client.get("/")
    assert response.status_code == 200

def test_prediction():
    response = client.post("/predict", json=sample_data)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "income_class" in result
    assert "probabilities" in result

def test_missing_features():
    response = client.post("/predict", json={"age": 39})
    assert response.status_code == 422
