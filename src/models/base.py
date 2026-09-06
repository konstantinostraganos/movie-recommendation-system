"""
Abstract base class for all recommender models.

Every model implements the same interface:
- fit(train_data)      → learn from training data
- recommend(user_id, k) → return top-K movie recommendations

This makes model comparison trivial — swap the model object,
everything else stays the same.
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseRecommender(ABC):
    """Interface that all recommender models must implement."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, train: pd.DataFrame, **kwargs) -> None:
        """Train the model on the training set."""
        pass

    @abstractmethod
    def recommend(
        self, user_idx: int, k: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        """Generate top-K recommendations for a user.

        Parameters
        ----------
        user_idx : int
            Contiguous user index (from ID remapping).
        k : int
            Number of recommendations to return.
        exclude_seen : bool
            If True, don't recommend movies the user already rated.

        Returns
        -------
        list of movie_idx, ordered by predicted relevance (best first).
        """
        pass

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.name} ({status})"