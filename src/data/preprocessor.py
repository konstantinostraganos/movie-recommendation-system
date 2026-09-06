"""
Data preprocessing pipeline (strict temporal version).

Pipeline order:
  1. Validation       — basic sanity checks on raw ratings
  2. [Temporal split is done by splitter.py BEFORE this]
  3. Train-only filtering — user/movie min counts from train only
  4. Temporal tag cutoff  — only tags before training period end
  5. Content text build   — genres + temporally-filtered tags
  6. ID remapping         — keep originals, add contiguous columns

This ordering guarantees zero data leakage: the test set is never
used to decide which users/movies to keep, and no future tags
leak into content features.
"""

import logging

import pandas as pd

from src.utils.config import cfg

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# Step 1: Validation (runs on raw ratings, before split)
# -----------------------------------------------------------------

def validate_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Run basic sanity checks on raw ratings data."""
    n_before = len(ratings)

    ratings = ratings.dropna()

    valid_mask = ratings["rating"].between(0.5, 5.0)
    ratings = ratings[valid_mask]

    ratings = ratings[ratings["timestamp"] > 0]

    n_after = len(ratings)
    if n_before != n_after:
        logger.warning(
            "Dropped %s invalid rows during validation.",
            f"{n_before - n_after:,}",
        )
    else:
        logger.info(
            "Validation passed — all %s rows clean.",
            f"{n_after:,}",
        )

    return ratings


# -----------------------------------------------------------------
# Step 3: Train-only filtering
# -----------------------------------------------------------------

def filter_by_train_counts(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    min_user_ratings: int | None = None,
    min_movie_ratings: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter users and movies based on train set counts only.

    Iteratively removes users/movies that don't meet minimum
    thresholds, using ONLY training data to compute counts.
    Then removes those same users/movies from val and test.

    This avoids leakage: we never look at val/test to decide
    what to keep.
    """
    min_user = min_user_ratings or cfg.data.min_ratings_per_user
    min_movie = min_movie_ratings or cfg.data.min_ratings_per_movie

    logger.info(
        "Train-only filtering: min %d ratings/user, "
        "min %d ratings/movie.",
        min_user,
        min_movie,
    )

    prev_len = 0
    iteration = 0

    while len(train) != prev_len:
        prev_len = len(train)
        iteration += 1

        # Filter users based on train counts
        user_counts = train["userId"].value_counts()
        valid_users = user_counts[user_counts >= min_user].index
        train = train[train["userId"].isin(valid_users)]

        # Filter movies based on train counts
        movie_counts = train["movieId"].value_counts()
        valid_movies = movie_counts[movie_counts >= min_movie].index
        train = train[train["movieId"].isin(valid_movies)]

        logger.info(
            "  Iteration %d: %s train ratings, %s users, %s movies.",
            iteration,
            f"{len(train):,}",
            f"{train['userId'].nunique():,}",
            f"{train['movieId'].nunique():,}",
        )

    logger.info("Filtering converged after %d iterations.", iteration)

    # Apply the same user/movie sets to val and test
    final_users = set(train["userId"].unique())
    final_movies = set(train["movieId"].unique())

    val_before = len(val)
    test_before = len(test)

    val = val[
        val["userId"].isin(final_users)
        & val["movieId"].isin(final_movies)
    ].reset_index(drop=True)

    test = test[
        test["userId"].isin(final_users)
        & test["movieId"].isin(final_movies)
    ].reset_index(drop=True)

    logger.info(
        "Val: %s -> %s (%s removed). Test: %s -> %s (%s removed).",
        f"{val_before:,}",
        f"{len(val):,}",
        f"{val_before - len(val):,}",
        f"{test_before:,}",
        f"{len(test):,}",
        f"{test_before - len(test):,}",
    )

    return train, val, test


# -----------------------------------------------------------------
# Step 4: Temporal tag cutoff
# -----------------------------------------------------------------

def filter_tags_temporally(
    tags: pd.DataFrame,
    train: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only tags created before the end of the training period.

    Uses the global max timestamp in the training set as cutoff.
    This prevents future user opinions from leaking into content
    features.
    """
    train_cutoff = train["timestamp"].max()

    n_before = len(tags)
    tags = tags[tags["timestamp"] <= train_cutoff].copy()
    n_after = len(tags)

    logger.info(
        "Temporal tag filter: %s -> %s tags (%s removed, cutoff=%d).",
        f"{n_before:,}",
        f"{n_after:,}",
        f"{n_before - n_after:,}",
        train_cutoff,
    )

    return tags


# -----------------------------------------------------------------
# Step 5: Content text building
# -----------------------------------------------------------------

def aggregate_tags(tags: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tags into one text string per movie.

    Lowercases, deduplicates per movie, and joins into a single
    space-separated string suitable for TF-IDF.
    """
    logger.info("Aggregating tags per movie ...")

    tags = tags.dropna(subset=["tag"])
    tags["tag"] = tags["tag"].str.lower().str.strip()

    agg = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(sorted(set(x))))
        .reset_index()
    )
    agg.columns = ["movieId", "tags_text"]

    logger.info("Aggregated tags for %s movies.", f"{len(agg):,}")
    return agg


def parse_genres(movies: pd.DataFrame) -> pd.DataFrame:
    """Convert pipe-separated genres to clean space-separated text."""
    movies = movies.copy()
    movies["genres_text"] = (
        movies["genres"]
        .str.replace("|", " ", regex=False)
        .str.replace("(no genres listed)", "", regex=False)
        .str.strip()
        .str.lower()
    )
    return movies


def build_content_text(
    movies: pd.DataFrame,
    tags_agg: pd.DataFrame,
) -> pd.DataFrame:
    """Combine genres and filtered tags into content text per movie.

    Movies without tags fall back to genres only.
    """
    movies = movies.merge(tags_agg, on="movieId", how="left")
    movies["tags_text"] = movies["tags_text"].fillna("")
    movies["content_text"] = (
        movies["genres_text"] + " " + movies["tags_text"]
    ).str.strip()

    n_with_tags = (movies["tags_text"] != "").sum()
    logger.info(
        "%s / %s movies (%.1f%%) have user tags.",
        f"{n_with_tags:,}",
        f"{len(movies):,}",
        100 * n_with_tags / len(movies),
    )
    return movies


# -----------------------------------------------------------------
# Step 6: ID remapping (keep originals)
# -----------------------------------------------------------------

def remap_ids(
    ratings: pd.DataFrame,
    user_pool: set | None = None,
    movie_pool: set | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """Add contiguous 0-indexed ID columns alongside originals.

    Creates 'user_idx' and 'movie_idx' columns. The original
    'userId' and 'movieId' columns are kept for interpretability.
    """
    users = sorted(user_pool or ratings["userId"].unique())
    movies = sorted(movie_pool or ratings["movieId"].unique())

    user_map = {old: new for new, old in enumerate(users)}
    movie_map = {old: new for new, old in enumerate(movies)}

    ratings = ratings.copy()
    ratings["user_idx"] = (
        ratings["userId"].map(user_map).astype("int32")
    )
    ratings["movie_idx"] = (
        ratings["movieId"].map(movie_map).astype("int32")
    )

    logger.info(
        "Remapped IDs: %s users -> [0, %d], %s movies -> [0, %d].",
        f"{len(users):,}",
        len(users) - 1,
        f"{len(movies):,}",
        len(movies) - 1,
    )
    return ratings, user_map, movie_map


# -----------------------------------------------------------------
# Full pipeline orchestrator
# -----------------------------------------------------------------

def run_preprocessing(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    movies: pd.DataFrame,
    tags: pd.DataFrame,
) -> dict:
    """Execute the full preprocessing pipeline on already-split data.

    Pipeline:
      1. Validation already done before split
      2. Split already done by splitter.py
      3. Train-only filtering
      4. Temporal tag cutoff
      5. Content text building
      6. ID remapping

    Returns
    -------
    dict with keys:
        train, val, test  — filtered and ID-remapped
        movies            — enriched with content_text
        user_map          — original userId -> contiguous int
        movie_map         — original movieId -> contiguous int
    """
    logger.info("=" * 60)
    logger.info("Starting preprocessing pipeline (strict temporal)")
    logger.info("=" * 60)

    # Step 3: Train-only filtering
    train, val, test = filter_by_train_counts(train, val, test)

    # Step 4: Temporal tag cutoff
    tags = filter_tags_temporally(tags, train)

    # Step 5: Content text
    surviving_movies = set(train["movieId"].unique())
    movies = movies[movies["movieId"].isin(surviving_movies)].copy()
    movies = parse_genres(movies)
    tags_agg = aggregate_tags(tags)
    movies = build_content_text(movies, tags_agg)

    # Step 6: Remap IDs (keep originals, shared mapping)
    user_pool = set(train["userId"].unique())
    movie_pool = set(train["movieId"].unique())

    train, user_map, movie_map = remap_ids(
        train, user_pool, movie_pool
    )
    val, _, _ = remap_ids(val, user_pool, movie_pool)
    test, _, _ = remap_ids(test, user_pool, movie_pool)

    # Add remapped IDs to movies too
    movies["movie_idx"] = (
        movies["movieId"].map(movie_map).astype("int32")
    )

    logger.info("=" * 60)
    logger.info("Preprocessing complete.")
    logger.info(
        "Final: %s train, %s val, %s test, %s users, %s movies.",
        f"{len(train):,}",
        f"{len(val):,}",
        f"{len(test):,}",
        f"{train['user_idx'].nunique():,}",
        f"{train['movie_idx'].nunique():,}",
    )
    logger.info("=" * 60)

    return {
        "train": train,
        "val": val,
        "test": test,
        "movies": movies,
        "user_map": user_map,
        "movie_map": movie_map,
    }