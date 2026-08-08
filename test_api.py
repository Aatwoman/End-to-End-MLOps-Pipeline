"""
test_api.py
Basic tests for the FastAPI service. Run after training a model:
    python train.py
    pytest
"""

import pytest
from fastapi.testclient import TestClient

from api import app


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager triggers FastAPI's startup event
    # (which loads the model) — without "with", startup/shutdown never fire.
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "class_names" in body
    assert len(body["class_names"]) == 3


def test_predict_valid_input(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in ("setosa", "versicolor", "virginica")
    assert 0.0 <= body["class_probabilities"][body["predicted_class"]] <= 1.0


def test_predict_missing_field(client):
    payload = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # FastAPI validation error


def test_predict_batch(client):
    payload = [
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3},
    ]
    response = client.post("/predict-batch", json=payload)
    assert response.status_code == 200
    assert len(response.json()) == 2
