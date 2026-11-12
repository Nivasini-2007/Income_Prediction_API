
# Real-Time ML Inference REST API & Capstone

## Project Overview

This project deploys a trained Machine Learning champion model
as a REST API using FastAPI.

The API predicts whether a person's annual income is above
or below $50K using the Adult Income dataset.

## Technologies Used

- Python
- FastAPI
- Scikit-learn
- Pandas
- Joblib
- Docker
- Pytest

## Project Architecture

User JSON Input
       |
       v
FastAPI REST API
       |
       v
Input Validation
       |
       v
Champion ML Model
       |
       v
Prediction Probabilities
       |
       v
JSON Response

## API Endpoints

### GET /

Checks whether the API is running.

### POST /predict

Accepts 14 input features and returns:

- Predicted income class
- Prediction value
- Class probabilities

## Installation

Install dependencies:

pip install -r requirements.txt

## Run API

uvicorn app:app --reload

## API Documentation

Open:

http://127.0.0.1:8000/docs

## Run Tests

pytest -v

## Docker

Build Docker image:

docker build -t adult-income-api .

Run Docker container:

docker run -p 8000:8000 adult-income-api

## Model Information

Model: Task 04 Champion Model

Dataset: Adult Income Dataset

Target:
- 0 = <=50K
- 1 = >50K

## Conclusion

The project successfully packages a trained machine
learning model as a REST API with prediction probabilities,
automated tests, Docker configuration and documentation.
