"""Unit tests for request and response contracts."""

import pytest
from pydantic import ValidationError

from app.schemas import (
    BatchPredictionRequest,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)


def test_prediction_request_trims_identifiers():
    request = PredictionRequest(user_id=" 196 ", movie_id=" 242 ")
    assert request.user_id == "196"
    assert request.movie_id == "242"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"user_id": "1"},
        {"movie_id": "1"},
        {"user_id": "", "movie_id": "2"},
        {"user_id": "1", "movie_id": "   "},
    ],
)
def test_invalid_prediction_requests_are_rejected(payload):
    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


def test_prediction_response_enforces_rating_range():
    valid = PredictionResponse(
        user_id="1", movie_id="2", predicted_rating=4.2, model_version="1.0.0"
    )
    assert valid.predicted_rating == 4.2
    with pytest.raises(ValidationError):
        PredictionResponse(user_id="1", movie_id="2", predicted_rating=8.0, model_version="1.0.0")


def test_health_response_shape():
    response = HealthResponse(status="healthy", model_loaded=True)
    assert response.model_dump() == {"status": "healthy", "model_loaded": True}


def test_batch_limits_and_item_validation():
    request = BatchPredictionRequest(predictions=[{"user_id": "1", "movie_id": "2"}])
    assert len(request.predictions) == 1
    with pytest.raises(ValidationError):
        BatchPredictionRequest(predictions=[])
    with pytest.raises(ValidationError):
        BatchPredictionRequest(predictions=[{"user_id": " ", "movie_id": "2"}])
