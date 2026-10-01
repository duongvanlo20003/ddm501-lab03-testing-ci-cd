"""Behavioural tests for the trained recommender artifact."""

import time


def test_same_input_has_deterministic_output(trained_model):
    first = trained_model.predict("196", "242")
    assert trained_model.predict("196", "242") == first


def test_predictions_stay_in_the_rating_contract(trained_model, known_user_movie_pairs):
    for pair in known_user_movie_pairs:
        rating = trained_model.predict(pair["user_id"], pair["movie_id"])
        assert 1.0 <= rating <= 5.0


def test_batch_order_invariance(trained_model, known_user_movie_pairs):
    pairs = [(row["user_id"], row["movie_id"]) for row in known_user_movie_pairs]
    forward = trained_model.predict_batch(pairs)
    reverse = trained_model.predict_batch(list(reversed(pairs)))
    assert forward == list(reversed(reverse))


def test_batch_matches_individual_predictions(trained_model, known_user_movie_pairs):
    pairs = [(row["user_id"], row["movie_id"]) for row in known_user_movie_pairs]
    assert trained_model.predict_batch(pairs) == [trained_model.predict(*pair) for pair in pairs]


def test_unknown_users_and_movies_are_handled(trained_model, unknown_users, unknown_movies):
    for user_id, movie_id in zip(unknown_users, unknown_movies):
        rating = trained_model.predict(user_id, movie_id)
        assert 1.0 <= rating <= 5.0


def test_predictions_are_not_constant(trained_model, known_user_movie_pairs):
    ratings = {
        trained_model.predict(row["user_id"], row["movie_id"]) for row in known_user_movie_pairs
    }
    assert len(ratings) > 1


def test_small_batch_is_fast_enough_for_an_api(trained_model, known_user_movie_pairs):
    pairs = [(row["user_id"], row["movie_id"]) for row in known_user_movie_pairs] * 20
    started = time.perf_counter()
    result = trained_model.predict_batch(pairs)
    assert len(result) == len(pairs)
    assert time.perf_counter() - started < 2.0
