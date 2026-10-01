"""Integration tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

import app.main as main_module


class FakeModel:
    def __init__(self, loaded=True):
        self.loaded = loaded

    def is_loaded(self):
        return self.loaded

    def predict(self, user_id, movie_id):
        if user_id == "boom":
            raise RuntimeError("synthetic failure")
        return round(1.0 + (sum(map(ord, user_id + movie_id)) % 401) / 100, 2)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_module, "model", FakeModel())
    return TestClient(main_module.app)


def test_root_and_health(client):
    root = client.get("/")
    assert root.status_code == 200
    assert {"name", "version", "docs", "health"} <= set(root.json())
    assert client.get("/health").json() == {"status": "healthy", "model_loaded": True}


def test_prediction_response_contract(client):
    response = client.post("/predict", json={"user_id": "196", "movie_id": "242"})
    assert response.status_code == 200
    assert set(response.json()) == {"user_id", "movie_id", "predicted_rating", "model_version"}
    assert 1.0 <= response.json()["predicted_rating"] <= 5.0


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"user_id": "196"},
        {"movie_id": "242"},
        {"user_id": "", "movie_id": "242"},
        {"user_id": "196", "movie_id": " "},
    ],
)
def test_invalid_payloads_return_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_prediction_preserves_count_and_order(client):
    payload = {
        "predictions": [
            {"user_id": "196", "movie_id": "242"},
            {"user_id": "186", "movie_id": "302"},
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == len(body["predictions"]) == 2
    assert [item["user_id"] for item in body["predictions"]] == ["196", "186"]


def test_empty_batch_returns_422(client):
    assert client.post("/predict/batch", json={"predictions": []}).status_code == 422


def test_unavailable_model_returns_503(client, monkeypatch):
    monkeypatch.setattr(main_module, "model", None)
    assert client.post("/predict", json={"user_id": "1", "movie_id": "2"}).status_code == 503
    assert (
        client.post(
            "/predict/batch", json={"predictions": [{"user_id": "1", "movie_id": "2"}]}
        ).status_code
        == 503
    )
    assert client.get("/health").json()["status"] == "unhealthy"


def test_prediction_errors_return_500(client):
    assert client.post("/predict", json={"user_id": "boom", "movie_id": "2"}).status_code == 500
    response = client.post(
        "/predict/batch", json={"predictions": [{"user_id": "boom", "movie_id": "2"}]}
    )
    assert response.status_code == 500


def test_model_info(client):
    body = client.get("/model/info").json()
    assert body["is_loaded"] is True
    assert body["model_type"] == "SVD (Collaborative Filtering)"
