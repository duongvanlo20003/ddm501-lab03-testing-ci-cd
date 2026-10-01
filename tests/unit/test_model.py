"""Unit tests for the movie-rating model wrapper."""

from types import SimpleNamespace

import pytest

from app.model import MovieRatingModel, get_model, reset_model


class DummyRecommender:
    def predict(self, user_id, movie_id):
        value = 3.0 + (int(user_id) % 5) * 0.1 if str(user_id).isdigit() else 3.5
        return SimpleNamespace(est=value)


@pytest.fixture
def wrapper():
    instance = MovieRatingModel.__new__(MovieRatingModel)
    instance.model_path = "unused"
    instance.model = DummyRecommender()
    return instance


def test_predict_returns_a_clipped_float(wrapper):
    rating = wrapper.predict("196", "242")
    assert isinstance(rating, float)
    assert 1.0 <= rating <= 5.0


def test_predict_batch_preserves_order(wrapper):
    pairs = [("196", "242"), ("186", "302"), ("22", "377")]
    assert wrapper.predict_batch(pairs) == [wrapper.predict(*pair) for pair in pairs]


def test_model_loaded_state(wrapper):
    assert wrapper.is_loaded()
    wrapper.model = None
    assert not wrapper.is_loaded()
    with pytest.raises(RuntimeError):
        wrapper.predict("1", "1")
    with pytest.raises(RuntimeError):
        wrapper.predict_batch([("1", "1")])


def test_missing_and_corrupt_model_files_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        MovieRatingModel(str(tmp_path / "missing.pkl"))
    corrupt = tmp_path / "corrupt.pkl"
    corrupt.write_bytes(b"not a pickle")
    with pytest.raises(Exception):
        MovieRatingModel(str(corrupt))


def test_singleton_can_be_reset(monkeypatch, wrapper):
    import app.model as model_module

    reset_model()
    monkeypatch.setattr(model_module, "MovieRatingModel", lambda: wrapper)
    assert get_model() is wrapper
    assert get_model() is wrapper
    reset_model()
