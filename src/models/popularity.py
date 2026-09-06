"""
Popularity-based recommender (non-personalized baseline).

Recommends the same "most popular" movies to every user.
Every other model must beat this — if it can't, it's not
adding value over a trivial approach.

Popularity is defined using a Bayesian average (IMDb-style weighted
rating) to avoid the problem of movies with very few ratings but
a perfect 5.0 average dominating the list.

Formula:
    score = (n / (n + m)) * avg_rating + (m / (n + m)) * C

Where:
    n = number of ratings for this movie
    m = minimum ratings threshold (we use the median)
    C = global mean rating
    avg_rating = this movie's mean rating
"""

import logging

import numpy as np
import pandas as pd

from src.models.base import BaseRecommender

logger = logging.getLogger(__name__)


class PopularityRecommender(BaseRecommender):

    def __init__(self):
        super().__init__(name="Popularity")
        self.ranked_movies = None
        self.user_seen = None

    def fit(self, train: pd.DataFrame, **kwargs) -> None:
        """Compute popularity scores from training data.

        Uses Bayesian average to balance rating count vs rating value.
        A movie needs enough ratings AND high average to rank well.
        """
        movie_stats = train.groupby("movie_idx").agg(
            avg_rating=("rating", "mean"),
            n_ratings=("rating", "count"),
        )

        # Bayesian average parameters
        C = train["rating"].mean()                # global mean
        m = movie_stats["n_ratings"].median()      # minimum threshold

        # Weighted score
        n = movie_stats["n_ratings"]
        avg = movie_stats["avg_rating"]
        movie_stats["score"] = (n / (n + m)) * avg + (m / (n + m)) * C

        # Sort by score descending
        self.ranked_movies = (
            movie_stats.sort_values("score", ascending=False)
            .index.tolist()
        )

        # Track what each user has already seen (for exclusion)
        self.user_seen = (
            train.groupby("user_idx")["movie_idx"]
            .apply(set)
            .to_dict()
        )

        self.is_fitted = True

        logger.info(
            "Popularity model fitted. Top movie score: %.3f, "
            "Global mean: %.3f, Min threshold (m): %.0f",
            movie_stats["score"].max(),
            C,
            m,
        )

    def recommend(
        self, user_idx: int, k: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        """Return top-K popular movies, excluding already-rated ones."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if exclude_seen:
            seen = self.user_seen.get(user_idx, set())
            recs = [m for m in self.ranked_movies if m not in seen]
        else:
            recs = self.ranked_movies

        return recs[:k]