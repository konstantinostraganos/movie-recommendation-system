"""
Run the full data preparation pipeline and save results.

Usage:
    python scripts/prepare_data.py

Pipeline:
    1. Load raw data
    2. Validate ratings
    3. Temporal split
    4. Train-only filtering + temporal tag cutoff
    5. Save processed splits + movies + ID mappings
    6. Create and save dev subset
"""

import logging
import pickle
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_ratings, load_movies, load_tags
from src.data.preprocessor import validate_ratings, run_preprocessing
from src.data.splitter import (
    temporal_split,
    create_dev_subset,
    save_splits,
)
from src.utils.config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("DATA PREPARATION PIPELINE")
    logger.info("=" * 60)

    # Step 1: Load raw data
    ratings = load_ratings()
    movies = load_movies()
    tags = load_tags()

    # Step 2: Validate
    ratings = validate_ratings(ratings)

    # Step 3: Temporal split (BEFORE filtering)
    train, val, test = temporal_split(ratings)

    # Step 4: Preprocessing (train-only filtering, tag cutoff, etc.)
    result = run_preprocessing(train, val, test, movies, tags)

    # Step 5: Save processed data
    processed_dir = Path(cfg.paths.processed_data)
    processed_dir.mkdir(parents=True, exist_ok=True)

    save_splits(
        result["train"],
        result["val"],
        result["test"],
        cfg.paths.processed_data,
    )

    result["movies"].to_parquet(
        processed_dir / "movies_clean.parquet", index=False
    )

    artifacts_dir = Path(cfg.paths.artifacts)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with open(artifacts_dir / "id_mappings.pkl", "wb") as f:
        pickle.dump(
            {
                "user_map": result["user_map"],
                "movie_map": result["movie_map"],
            },
            f,
        )

    logger.info("Saved processed data to %s", processed_dir)
    logger.info("Saved ID mappings to %s", artifacts_dir)

    # Step 6: Dev subset
    dev_ratings = create_dev_subset(result["train"])
    dev_train, dev_val, dev_test = temporal_split(dev_ratings)

    dev_dir = Path(cfg.paths.dev_data)
    save_splits(dev_train, dev_val, dev_test, dev_dir, prefix="dev")

    # Final summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(
        "Full — train: %s  val: %s  test: %s",
        f"{len(result['train']):,}",
        f"{len(result['val']):,}",
        f"{len(result['test']):,}",
    )
    logger.info(
        "Dev  — train: %s  val: %s  test: %s",
        f"{len(dev_train):,}",
        f"{len(dev_val):,}",
        f"{len(dev_test):,}",
    )
    logger.info(
        "Users: %s  Movies: %s",
        f"{result['train']['user_idx'].nunique():,}",
        f"{result['train']['movie_idx'].nunique():,}",
    )


if __name__ == "__main__":
    main()