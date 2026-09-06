"""Quick experiment: at what activity level does ALS beat Content-Based?"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
from src.models.collaborative import ALSRecommender
from src.models.content_based import ContentBasedRecommender
from src.evaluation.metrics import evaluate_model, build_test_relevant

train = pd.read_parquet("data/dev/dev_train.parquet")
test = pd.read_parquet("data/dev/dev_test.parquet")
movies = pd.read_parquet("data/processed/movies_clean.parquet")

# Fit both models
als = ALSRecommender()
als.fit(train)
cb = ContentBasedRecommender()
cb.fit(train, movies=movies)

test_rel = build_test_relevant(test)

# Group users by activity level
user_counts = train.groupby("user_idx").size()

brackets = [(18, 30), (30, 50), (50, 100), (100, 500), (500, 99999)]

print()
print(f"{'Bracket':>15} {'Users':>8} {'ALS NDCG':>10} {'CB NDCG':>10} {'Winner':>10}")
print("-" * 60)

for lo, hi in brackets:
    users_in_bracket = set(
        user_counts[(user_counts >= lo) & (user_counts < hi)].index
    )
    test_users = {
        u: v for u, v in test_rel.items() if u in users_in_bracket
    }
    if len(test_users) < 10:
        continue

    als_recs = {u: als.recommend(u, k=10) for u in test_users}
    cb_recs = {u: cb.recommend(u, k=10) for u in test_users}

    als_res = evaluate_model(als_recs, test_users)
    cb_res = evaluate_model(cb_recs, test_users)

    winner = "ALS" if als_res["NDCG@10"] > cb_res["NDCG@10"] else "Content"

    print(
        f"{lo:>6}-{hi:<6} {len(test_users):>8,}"
        f" {als_res['NDCG@10']:>10.4f}"
        f" {cb_res['NDCG@10']:>10.4f}"
        f" {winner:>10}"
    )