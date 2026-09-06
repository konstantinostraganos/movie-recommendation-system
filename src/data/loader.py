"""
Data loader for MovieLens 25M.

Handles loading each CSV into a pandas DataFrame with correct dtypes.
We read CSVs with explicit dtypes to avoid silent type coercion and
to catch data issues early.
"""

import logging
from pathlib import Path

import pandas as pd

from src.utils.config import cfg

logger = logging.getLogger(__name__)


def load_ratings(raw_path: str | Path | None = None) -> pd.DataFrame:
    """Load ratings.csv with correct dtypes.

    Columns: userId (int), movieId (int), rating (float), timestamp (int).
    We keep timestamp as int here — conversion to datetime happens in
    preprocessing so that raw loading stays fast and lossless.
    """
    path = Path(raw_path or cfg.paths.raw_data) / "ratings.csv"
    logger.info("Loading ratings from %s ...", path)

    df = pd.read_csv(
        path,
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32",
            "timestamp": "int64",
        },
    )
    logger.info("Loaded %s ratings.", f"{len(df):,}")
    return df


def load_movies(raw_path: str | Path | None = None) -> pd.DataFrame:
    """Load movies.csv.

    Columns: movieId (int), title (str), genres (str — pipe-separated).
    """
    path = Path(raw_path or cfg.paths.raw_data) / "movies.csv"
    logger.info("Loading movies from %s ...", path)

    df = pd.read_csv(path, dtype={"movieId": "int32"})
    logger.info("Loaded %s movies.", f"{len(df):,}")
    return df


def load_tags(raw_path: str | Path | None = None) -> pd.DataFrame:
    """Load tags.csv — user-generated free-text tags on movies.

    Columns: userId (int), movieId (int), tag (str), timestamp (int).
    """
    path = Path(raw_path or cfg.paths.raw_data) / "tags.csv"
    logger.info("Loading tags from %s ...", path)

    df = pd.read_csv(
        path,
        dtype={"userId": "int32", "movieId": "int32", "timestamp": "int64"},
    )
    logger.info("Loaded %s tag applications.", f"{len(df):,}")
    return df


def load_links(raw_path: str | Path | None = None) -> pd.DataFrame:
    """Load links.csv — mappings to IMDb and TMDb IDs.

    Columns: movieId (int), imdbId (int), tmdbId (float — has NaNs).
    """
    path = Path(raw_path or cfg.paths.raw_data) / "links.csv"
    logger.info("Loading links from %s ...", path)

    df = pd.read_csv(
        path,
        dtype={"movieId": "int32", "imdbId": "int64"},
    )
    logger.info("Loaded links for %s movies.", f"{len(df):,}")
    return df


def load_genome_scores(raw_path: str | Path | None = None) -> pd.DataFrame:
    """Load genome-scores.csv — tag relevance scores per movie.

    Columns: movieId (int), tagId (int), relevance (float).
    Warning: ~15M rows — only load when needed.
    """
    path = Path(raw_path or cfg.paths.raw_data) / "genome-scores.csv"
    logger.info("Loading genome scores from %s (large file) ...", path)

    df = pd.read_csv(
        path,
        dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"},
    )
    logger.info("Loaded %s genome score entries.", f"{len(df):,}")
    return df


def load_genome_tags(raw_path: str | Path | None = None) -> pd.DataFrame:
    """Load genome-tags.csv — tag ID to tag name mapping.

    Columns: tagId (int), tag (str). 1128 unique tags.
    """
    path = Path(raw_path or cfg.paths.raw_data) / "genome-tags.csv"
    logger.info("Loading genome tags from %s ...", path)

    df = pd.read_csv(path, dtype={"tagId": "int32"})
    logger.info("Loaded %s genome tags.", f"{len(df):,}")
    return df