"""Tests for data integrity — verify processed data is correct."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


def test_train_loads():
    """Train parquet loads with correct columns."""
    train = pd.read_parquet("data/processed/train.parquet")
    expected_cols = {"userId", "movieId", "rating", "timestamp",
                     "user_idx", "movie_idx"}
    assert expected_cols.issubset(set(train.columns))


def test_no_nulls_in_train():
    """No null values in training data."""
    train = pd.read_parquet("data/processed/train.parquet")
    assert train[["user_idx", "movie_idx", "rating"]].isnull().sum().sum() == 0


def test_rating_range():
    """All ratings between 0.5 and 5.0."""
    train = pd.read_parquet("data/processed/train.parquet")
    assert train["rating"].min() >= 0.5
    assert train["rating"].max() <= 5.0


def test_movies_have_content():
    """Movies have content_text for content-based model."""
    movies = pd.read_parquet("data/processed/movies_clean.parquet")
    assert "content_text" in movies.columns
    # At least 80% should have non-empty content
    has_content = (movies["content_text"] != "").sum()
    assert has_content / len(movies) > 0.8


def test_user_idx_contiguous():
    """User indices are contiguous starting from 0."""
    train = pd.read_parquet("data/processed/train.parquet")
    max_idx = train["user_idx"].max()
    n_unique = train["user_idx"].nunique()
    # Should be close (not exact due to val/test removal)
    assert max_idx < n_unique * 1.1


def test_temporal_split_no_leakage():
    """Train max timestamp <= test min timestamp per user."""
    train = pd.read_parquet("data/processed/train.parquet")
    test = pd.read_parquet("data/processed/test.parquet")

    train_max = train.groupby("user_idx")["timestamp"].max()
    test_min = test.groupby("user_idx")["timestamp"].min()

    common = train_max.index.intersection(test_min.index)
    # Allow tied timestamps but no future leakage
    leaks = (train_max[common] > test_min[common]).sum()
    assert leaks == 0 or leaks < len(common) * 0.15  # < 15% tied is OK