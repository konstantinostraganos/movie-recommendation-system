"""Tests for recommendation models — verify they fit and recommend."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.models.popularity import PopularityRecommender
from src.models.content_based import ContentBasedRecommender


def get_test_data():
    """Load dev data for testing."""
    train = pd.read_parquet("data/dev/dev_train.parquet")
    movies = pd.read_parquet("data/processed/movies_clean.parquet")
    return train, movies


def test_popularity_fit_and_recommend():
    """Popularity model fits and returns correct number of recs."""
    train, _ = get_test_data()
    model = PopularityRecommender()
    model.fit(train)

    assert model.is_fitted

    user_idx = train["user_idx"].iloc[0]
    recs = model.recommend(user_idx, k=10)

    assert len(recs) == 10
    assert len(set(recs)) == 10  # no duplicates


def test_popularity_excludes_seen():
    """Popularity model doesn't recommend already-rated movies."""
    train, _ = get_test_data()
    model = PopularityRecommender()
    model.fit(train)

    user_idx = train["user_idx"].iloc[0]
    seen = set(
        train[train["user_idx"] == user_idx]["movie_idx"]
    )
    recs = model.recommend(user_idx, k=10)

    assert len(set(recs) & seen) == 0


def test_content_based_fit_and_recommend():
    """Content-based model fits and returns recs."""
    train, movies = get_test_data()
    model = ContentBasedRecommender()
    model.fit(train, movies=movies)

    assert model.is_fitted

    user_idx = train["user_idx"].iloc[0]
    recs = model.recommend(user_idx, k=10)

    assert len(recs) == 10


def test_content_based_similar_movies():
    """Similar movies returns correct number of results."""
    train, movies = get_test_data()
    model = ContentBasedRecommender()
    model.fit(train, movies=movies)

    movie_idx = train["movie_idx"].iloc[0]
    similar = model.similar_movies(movie_idx, k=5)

    assert len(similar) == 5
    # Each result is (movie_idx, score)
    assert all(isinstance(s, tuple) for s in similar)
    assert all(0 <= s[1] <= 1 for s in similar)


def test_content_based_similar_excludes_self():
    """Similar movies doesn't include the source movie."""
    train, movies = get_test_data()
    model = ContentBasedRecommender()
    model.fit(train, movies=movies)

    movie_idx = train["movie_idx"].iloc[0]
    similar = model.similar_movies(movie_idx, k=5)

    similar_ids = [s[0] for s in similar]
    assert movie_idx not in similar_ids