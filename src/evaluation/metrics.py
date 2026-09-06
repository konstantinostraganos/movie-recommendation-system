"""
Ranking evaluation metrics for recommendation systems.

All metrics follow the same pattern:
1. For each user, get the model's top-K recommended items
2. Compare against that user's actual relevant items
3. Compute per-user metric
4. Average across all users

Implemented from scratch so we understand exactly what we measure.
This matters in interviews — "what does NDCG actually compute?"
"""

import logging

import numpy as np
import pandas as pd

from src.utils.config import cfg

logger = logging.getLogger(__name__)


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """What fraction of our top-K recommendations are relevant?

    Example: recommended=[A,B,C,D,E], relevant={B,D,F}, k=5
    Hits = {B, D} → precision = 2/5 = 0.4
    """
    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """What fraction of relevant items did we find in top-K?

    Example: recommended=[A,B,C,D,E], relevant={B,D,F}, k=5
    Hits = {B, D}, total relevant = 3 → recall = 2/3 = 0.667
    """
    if len(relevant) == 0:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Unlike precision, NDCG rewards correct items HIGHER in the list.
    A relevant item at position 1 counts more than at position 10.

    DCG = sum of (1 / log2(position + 1)) for each hit
    IDCG = best possible DCG (all relevant items at the top)
    NDCG = DCG / IDCG
    """
    top_k = recommended[:k]

    # DCG: actual score based on where hits landed
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed

    # IDCG: best possible (all relevant items at top positions)
    n_relevant_in_k = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant_in_k))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    """Did we get at least one relevant item in top-K?

    Returns 1.0 if yes, 0.0 if no. Averaged across users, this gives
    the percentage of users who got at least one good recommendation.
    """
    top_k = recommended[:k]
    return 1.0 if len(set(top_k) & relevant) > 0 else 0.0


def ap_at_k(recommended: list, relevant: set, k: int) -> float:
    """Average Precision at K.

    Computes precision at each position where a relevant item appears,
    then averages. Rewards models that place relevant items earlier.

    Used to compute MAP@K (Mean Average Precision) across users.
    """
    if len(relevant) == 0:
        return 0.0

    top_k = recommended[:k]
    score = 0.0
    hits = 0

    for i, item in enumerate(top_k):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(relevant), k)


def evaluate_model(
    recommendations: dict[int, list[int]],
    test_relevant: dict[int, set[int]],
    k: int | None = None,
) -> dict[str, float]:
    """Evaluate a recommender model across all test users.

    Parameters
    ----------
    recommendations : dict
        {user_id: [ranked list of movie_ids]} — model output
    test_relevant : dict
        {user_id: {set of relevant movie_ids}} — ground truth
    k : int, optional
        Top-K cutoff. Defaults to cfg.evaluation.k (10).

    Returns
    -------
    dict with metric names and their averaged values
    """
    k = k or cfg.evaluation.k

    precisions = []
    recalls = []
    ndcgs = []
    hit_rates = []
    aps = []

    # Only evaluate users who appear in both recommendations and test
    eval_users = set(recommendations.keys()) & set(test_relevant.keys())

    if len(eval_users) == 0:
        logger.warning("No common users between recommendations and test set!")
        return {
            f"Precision@{k}": 0.0,
            f"Recall@{k}": 0.0,
            f"NDCG@{k}": 0.0,
            f"HitRate@{k}": 0.0,
            f"MAP@{k}": 0.0,
        }

    for user_id in eval_users:
        rec = recommendations[user_id]
        rel = test_relevant[user_id]

        if len(rel) == 0:
            continue

        precisions.append(precision_at_k(rec, rel, k))
        recalls.append(recall_at_k(rec, rel, k))
        ndcgs.append(ndcg_at_k(rec, rel, k))
        hit_rates.append(hit_rate_at_k(rec, rel, k))
        aps.append(ap_at_k(rec, rel, k))

    results = {
        f"Precision@{k}": np.mean(precisions),
        f"Recall@{k}": np.mean(recalls),
        f"NDCG@{k}": np.mean(ndcgs),
        f"HitRate@{k}": np.mean(hit_rates),
        f"MAP@{k}": np.mean(aps),
    }

    logger.info("Evaluation results (K=%d, %d users):", k, len(precisions))
    for metric, value in results.items():
        logger.info("  %s: %.4f", metric, value)

    return results


def build_test_relevant(
    test: pd.DataFrame,
    threshold: float | None = None,
) -> dict[int, set[int]]:
    """Build ground truth: relevant items per user from test set.

    Parameters
    ----------
    test : DataFrame
        Test set with columns: user_idx, movie_idx, rating
    threshold : float, optional
        Minimum rating to consider relevant.
        Defaults to cfg.evaluation.relevance_threshold (4.0).

    Returns
    -------
    dict {user_idx: set of relevant movie_idx}
    """
    threshold = threshold or cfg.evaluation.relevance_threshold

    relevant = test[test["rating"] >= threshold]

    test_relevant = (
        relevant.groupby("user_idx")["movie_idx"]
        .apply(set)
        .to_dict()
    )

    logger.info(
        "Built test relevant items: %s users, avg %.1f relevant items/user.",
        f"{len(test_relevant):,}",
        np.mean([len(v) for v in test_relevant.values()]),
    )
    return test_relevant