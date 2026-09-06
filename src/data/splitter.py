"""
Temporal train/validation/test splitting.

Strategy: leave-last-K-out per user, ordered by timestamp.
- Last rating per user       -> test set
- Second-to-last per user    -> validation set
- Everything else            -> train set

CRITICAL: This runs BEFORE any user/movie filtering, so that
the split reflects real future interactions. Filtering happens
afterward using only training set statistics.
"""

import logging

import numpy as np
import pandas as pd

from src.utils.config import cfg

logger = logging.getLogger(__name__)


def temporal_split(
    ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ratings into train/val/test using temporal ordering.

    For each user, we sort by timestamp and hold out the last
    ratings for test and the preceding ones for validation.
    """
    test_k = cfg.evaluation.test_holdout
    val_k = cfg.evaluation.val_holdout

    logger.info(
        "Temporal split: hold out last %d for test, %d for val "
        "per user.",
        test_k,
        val_k,
    )

    # Sort globally by user then timestamp
    ratings = ratings.sort_values(
        ["userId", "timestamp"]
    ).reset_index(drop=True)

    # Assign reverse rank within each user (0 = most recent)
    ratings["_rank"] = ratings.groupby("userId").cumcount(
        ascending=False
    )

    # Test: rank < test_k (the last test_k ratings)
    test_mask = ratings["_rank"] < test_k

    # Val: test_k <= rank < test_k + val_k
    val_mask = (ratings["_rank"] >= test_k) & (
        ratings["_rank"] < test_k + val_k
    )

    # Train: everything else
    train_mask = ratings["_rank"] >= (test_k + val_k)

    train = ratings[train_mask].drop(columns=["_rank"]).reset_index(
        drop=True
    )
    val = ratings[val_mask].drop(columns=["_rank"]).reset_index(
        drop=True
    )
    test = ratings[test_mask].drop(columns=["_rank"]).reset_index(
        drop=True
    )

    logger.info(
        "Split sizes — train: %s, val: %s, test: %s",
        f"{len(train):,}",
        f"{len(val):,}",
        f"{len(test):,}",
    )

    _validate_split(train, val, test)

    return train, val, test


def _validate_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Verify no data leakage in the temporal split."""
    train_users = set(train["userId"].unique())
    test_users = set(test["userId"].unique())
    val_users = set(val["userId"].unique())

    orphan_test = test_users - train_users
    orphan_val = val_users - train_users

    if orphan_test:
        logger.warning(
            "%d test users have no training data!",
            len(orphan_test),
        )
    if orphan_val:
        logger.warning(
            "%d val users have no training data!",
            len(orphan_val),
        )

    # Temporal check: max train timestamp < min test timestamp
    train_max_ts = train.groupby("userId")["timestamp"].max()
    test_min_ts = test.groupby("userId")["timestamp"].min()
    common = train_max_ts.index.intersection(test_min_ts.index)

    leaks = (
        train_max_ts[common] >= test_min_ts[common]
    ).sum()
    if leaks > 0:
        logger.warning(
            "%d users have train timestamps >= test timestamps "
            "(tied timestamps).",
            leaks,
        )
    else:
        logger.info(
            "Temporal split validation passed — no leakage detected."
        )


def create_dev_subset(
    ratings: pd.DataFrame,
    fraction: float | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Sample a fraction of users for faster development iteration.

    We sample users (not ratings) so that per-user distributions
    stay realistic. All ratings of sampled users are kept.
    """
    frac = fraction or cfg.data.dev_user_fraction
    rng_seed = seed or cfg.project.seed

    all_users = ratings["userId"].unique()
    n_sample = int(len(all_users) * frac)

    rng = np.random.RandomState(rng_seed)
    sampled_users = rng.choice(
        all_users, size=n_sample, replace=False
    )

    dev_ratings = ratings[
        ratings["userId"].isin(set(sampled_users))
    ].copy()

    logger.info(
        "Dev subset: %s users (%.0f%%), %s ratings.",
        f"{n_sample:,}",
        frac * 100,
        f"{len(dev_ratings):,}",
    )
    return dev_ratings


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str,
    prefix: str = "",
) -> None:
    """Save split DataFrames as parquet files."""
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    p = f"{prefix}_" if prefix else ""

    train.to_parquet(out / f"{p}train.parquet", index=False)
    val.to_parquet(out / f"{p}val.parquet", index=False)
    test.to_parquet(out / f"{p}test.parquet", index=False)

    logger.info("Saved splits to %s (prefix='%s').", out, prefix)