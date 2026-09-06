"""
Collaborative Filtering recommenders.

These models use the user-item interaction matrix — they don't know
anything about movie content (genres, tags). They find patterns like:
"Users who liked movies A, B, C also liked movie D."

Two approaches:
1. SVD (Surprise library) — classic matrix factorization for explicit
   ratings. Decomposes the user-item matrix into latent factors.
2. ALS (implicit library) — Alternating Least Squares, optimized for
   large sparse matrices.

Both learn low-dimensional representations (embeddings) for users and
movies. A user's predicted rating for a movie is the dot product of
their embeddings.
"""

import logging

import numpy as np
import pandas as pd
from scipy import sparse

from src.models.base import BaseRecommender
from src.utils.config import cfg

logger = logging.getLogger(__name__)


class SVDRecommender(BaseRecommender):
    """SVD-based collaborative filtering using Surprise library.

    Learns latent factor vectors for each user and movie.
    Predicted rating = global_mean + user_bias + movie_bias +
                       dot(user_vector, movie_vector)
    """

    def __init__(self):
        super().__init__(name="SVD")
        self.algo = None
        self.trainset = None
        self.user_seen = None
        self.all_movie_indices = None

    def fit(self, train: pd.DataFrame, **kwargs) -> None:
        """Train SVD on the ratings data."""
        from surprise import Dataset, Reader, SVD

        # Surprise needs its own data format
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            train[["user_idx", "movie_idx", "rating"]],
            reader,
        )
        self.trainset = data.build_full_trainset()

        # Configure and train SVD
        self.algo = SVD(
            n_factors=cfg.models.svd.n_factors,
            n_epochs=cfg.models.svd.n_epochs,
            lr_all=cfg.models.svd.lr_all,
            reg_all=cfg.models.svd.reg_all,
            random_state=cfg.project.seed,
            verbose=True,
        )
        self.algo.fit(self.trainset)

        # Track seen movies per user
        self.user_seen = (
            train.groupby("user_idx")["movie_idx"]
            .apply(set)
            .to_dict()
        )
        self.all_movie_indices = sorted(train["movie_idx"].unique())

        self.is_fitted = True
        logger.info("SVD fitted: %d factors.", cfg.models.svd.n_factors)

    def recommend(
        self, user_idx: int, k: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        """Predict ratings for all unseen movies and return top-K.

        Uses vectorized prediction instead of per-item loops for speed.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        seen = self.user_seen.get(user_idx, set()) if exclude_seen else set()

        # Vectorized: compute predicted ratings for all movies at once
        try:
            inner_uid = self.trainset.to_inner_uid(user_idx)
        except ValueError:
            return []

        bu = self.algo.bu[inner_uid]
        pu = self.algo.pu[inner_uid]

        # Score all movies at once with matrix multiplication
        scores = (
            self.algo.trainset.global_mean
            + bu
            + self.algo.bi
            + self.algo.qi.dot(pu)
        )

        # Map inner movie IDs back to our movie_idx
        candidates = []
        for inner_iid in range(len(scores)):
            try:
                movie_idx = self.trainset.to_raw_iid(inner_iid)
                if movie_idx not in seen:
                    candidates.append((movie_idx, scores[inner_iid]))
            except ValueError:
                continue

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [movie_idx for movie_idx, _ in candidates[:k]]


class ALSRecommender(BaseRecommender):
    """ALS-based collaborative filtering using implicit library.

    Alternating Least Squares on a confidence-weighted matrix.
    Optimized for large sparse datasets — much faster than SVD
    on our 25M ratings.

    The implicit library expects a user-item matrix where values
    represent confidence, not raw ratings. We convert ratings to
    confidence: higher rating = higher confidence that the user
    likes the movie.
    """

    def __init__(self):
        super().__init__(name="ALS")
        self.model = None
        self.user_items = None
        self.user_seen = None
        self.n_users = None
        self.n_movies = None

    def fit(self, train: pd.DataFrame, **kwargs) -> None:
        """Train ALS on the user-item interaction matrix."""
        from implicit.als import AlternatingLeastSquares

        self.n_users = train["user_idx"].max() + 1
        self.n_movies = train["movie_idx"].max() + 1

        # Build sparse user-item matrix
        self.user_items = sparse.csr_matrix(
            (
                train["rating"].values.astype(np.float32),
                (
                    train["user_idx"].values,
                    train["movie_idx"].values,
                ),
            ),
            shape=(self.n_users, self.n_movies),
        )

        # Track seen movies per user
        self.user_seen = (
            train.groupby("user_idx")["movie_idx"]
            .apply(set)
            .to_dict()
        )

        # Train ALS
        self.model = AlternatingLeastSquares(
            factors=cfg.models.als.factors,
            iterations=cfg.models.als.iterations,
            regularization=cfg.models.als.regularization,
            random_state=cfg.project.seed,
        )

        logger.info(
            "Training ALS: %d factors, %d iterations ...",
            cfg.models.als.factors,
            cfg.models.als.iterations,
        )
        self.model.fit(self.user_items)

        self.is_fitted = True
        logger.info("ALS fitted.")

    def recommend(
        self, user_idx: int, k: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        """Use ALS to recommend top-K movies for a user."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if user_idx >= self.n_users:
            return []

        ids, scores = self.model.recommend(
            user_idx,
            self.user_items[user_idx],
            N=k,
            filter_already_liked_items=exclude_seen,
        )

        return ids.tolist()