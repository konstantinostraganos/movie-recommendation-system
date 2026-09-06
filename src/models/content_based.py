"""
Content-based recommender using TF-IDF on genres + tags.

Logic: represent each movie as a text vector (genres + user tags),
compute cosine similarity between movies, and recommend movies
similar to what the user has liked in the past.

For a given user:
1. Find movies they rated highly (>= threshold)
2. Build a "user profile" by averaging those movies' vectors
3. Rank all unseen movies by cosine similarity to the profile
4. Return top-K

This is Content-Based v1 (TF-IDF). v2 will use sentence embeddings.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base import BaseRecommender
from src.utils.config import cfg

logger = logging.getLogger(__name__)


class ContentBasedRecommender(BaseRecommender):

    def __init__(self):
        super().__init__(name="Content-Based (TF-IDF)")
        self.tfidf_matrix = None
        self.movie_idx_to_pos = None
        self.user_seen = None
        self.user_profiles = None

    def fit(self, train: pd.DataFrame, movies: pd.DataFrame, **kwargs) -> None:
        """Build TF-IDF matrix from movie content text and user profiles.

        Steps:
        1. Vectorize movie content (genres + tags) with TF-IDF
        2. For each user, build a profile from their liked movies
        """
        # --- Step 1: TF-IDF vectorization ---
        # Map movie_idx to position in the TF-IDF matrix
        movie_list = movies.sort_values("movie_idx")
        self.movie_idx_to_pos = {
            idx: pos for pos, idx in enumerate(movie_list["movie_idx"])
        }

        vectorizer = TfidfVectorizer(
            max_features=cfg.models.content.tfidf_max_features,
            stop_words="english",
        )
        self.tfidf_matrix = vectorizer.fit_transform(
            movie_list["content_text"].fillna("")
        )

        logger.info(
            "TF-IDF matrix: %d movies x %d features.",
            self.tfidf_matrix.shape[0],
            self.tfidf_matrix.shape[1],
        )

        # --- Step 2: Build user profiles ---
        threshold = cfg.evaluation.relevance_threshold

        # Get movies each user liked
        liked = train[train["rating"] >= threshold]

        self.user_profiles = {}
        self.user_seen = {}

        for user_idx, group in train.groupby("user_idx"):
            self.user_seen[user_idx] = set(group["movie_idx"])

            # Get liked movies for this user
            user_liked = group[group["rating"] >= threshold]["movie_idx"]

            if len(user_liked) == 0:
                continue

            # Average TF-IDF vectors of liked movies = user profile
            positions = [
                self.movie_idx_to_pos[m]
                for m in user_liked
                if m in self.movie_idx_to_pos
            ]

            if len(positions) == 0:
                continue

            profile = np.asarray(
                self.tfidf_matrix[positions].mean(axis=0)
            )
            self.user_profiles[user_idx] = profile

        self.is_fitted = True

        logger.info(
            "Built profiles for %s / %s users.",
            f"{len(self.user_profiles):,}",
            f"{train['user_idx'].nunique():,}",
        )

    def recommend(
        self, user_idx: int, k: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        """Recommend movies similar to user's content profile."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if user_idx not in self.user_profiles:
            return []

        # Compute similarity between user profile and all movies
        profile = self.user_profiles[user_idx]
        similarities = cosine_similarity(profile, self.tfidf_matrix)[0]

        # Get movie indices sorted by similarity
        pos_to_movie = {
            pos: idx for idx, pos in self.movie_idx_to_pos.items()
        }

        # Rank by similarity descending
        ranked_positions = np.argsort(similarities)[::-1]

        seen = self.user_seen.get(user_idx, set()) if exclude_seen else set()

        recs = []
        for pos in ranked_positions:
            movie_idx = pos_to_movie[pos]
            if movie_idx not in seen:
                recs.append(movie_idx)
                if len(recs) == k:
                    break

        return recs

    def similar_movies(
        self, movie_idx: int, k: int = 10
    ) -> list[tuple[int, float]]:
        """Find movies most similar to a given movie.

        Returns list of (movie_idx, similarity_score) tuples.
        This powers the "I liked X, what's similar?" use case.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if movie_idx not in self.movie_idx_to_pos:
            return []

        pos = self.movie_idx_to_pos[movie_idx]
        movie_vector = self.tfidf_matrix[pos]

        similarities = cosine_similarity(movie_vector, self.tfidf_matrix)[0]

        pos_to_movie = {
            pos: idx for idx, pos in self.movie_idx_to_pos.items()
        }

        ranked = np.argsort(similarities)[::-1]

        results = []
        for p in ranked:
            mid = pos_to_movie[p]
            if mid == movie_idx:
                continue
            results.append((mid, float(similarities[p])))
            if len(results) == k:
                break

        return results        




class EmbeddingContentRecommender(BaseRecommender):
    """Content-based recommender using sentence embeddings.

    Instead of TF-IDF (bag-of-words), this uses a pre-trained
    transformer model to create dense semantic vectors.

    Key difference from TF-IDF:
    - TF-IDF: "space adventure" and "interstellar journey" = 0 similarity
    - Embeddings: "space adventure" ≈ "interstellar journey" (semantic match)

    Uses sentence-transformers/all-MiniLM-L6-v2 (~80MB, no GPU needed).
    """

    def __init__(self):
        super().__init__(name="Content-Based (Embeddings)")
        self.embeddings = None
        self.movie_idx_to_pos = None
        self.user_seen = None
        self.user_profiles = None

    def fit(self, train: pd.DataFrame, movies: pd.DataFrame, **kwargs) -> None:
        """Encode movie content text into dense vectors and build user profiles."""
        from sentence_transformers import SentenceTransformer

        model_name = cfg.models.content.embedding_model

        logger.info("Loading sentence transformer: %s ...", model_name)
        encoder = SentenceTransformer(model_name)

        # Sort movies by movie_idx for consistent ordering
        movie_list = movies.sort_values("movie_idx")
        self.movie_idx_to_pos = {
            idx: pos for pos, idx in enumerate(movie_list["movie_idx"])
        }

        # Encode all movie content texts
        texts = movie_list["content_text"].fillna("").tolist()
        logger.info("Encoding %d movie texts ...", len(texts))
        self.embeddings = encoder.encode(
            texts, show_progress_bar=True, batch_size=256
        )

        logger.info(
            "Embedding matrix: %d movies x %d dimensions.",
            self.embeddings.shape[0],
            self.embeddings.shape[1],
        )

        # Build user profiles (same logic as TF-IDF version)
        threshold = cfg.evaluation.relevance_threshold
        self.user_profiles = {}
        self.user_seen = {}

        for user_idx, group in train.groupby("user_idx"):
            self.user_seen[user_idx] = set(group["movie_idx"])

            user_liked = group[group["rating"] >= threshold]["movie_idx"]

            if len(user_liked) == 0:
                continue

            positions = [
                self.movie_idx_to_pos[m]
                for m in user_liked
                if m in self.movie_idx_to_pos
            ]

            if len(positions) == 0:
                continue

            profile = np.mean(self.embeddings[positions], axis=0, keepdims=True)
            self.user_profiles[user_idx] = profile

        self.is_fitted = True

        logger.info(
            "Built profiles for %s / %s users.",
            f"{len(self.user_profiles):,}",
            f"{train['user_idx'].nunique():,}",
        )

    def recommend(
        self, user_idx: int, k: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        """Recommend movies by cosine similarity to user embedding profile."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if user_idx not in self.user_profiles:
            return []

        profile = self.user_profiles[user_idx]
        similarities = cosine_similarity(profile, self.embeddings)[0]

        pos_to_movie = {
            pos: idx for idx, pos in self.movie_idx_to_pos.items()
        }

        ranked_positions = np.argsort(similarities)[::-1]

        seen = self.user_seen.get(user_idx, set()) if exclude_seen else set()

        recs = []
        for pos in ranked_positions:
            movie_idx = pos_to_movie[pos]
            if movie_idx not in seen:
                recs.append(movie_idx)
                if len(recs) == k:
                    break

        return recs        