"""
FastAPI service for movie recommendations.

Endpoints:
  GET /recommend/user/{user_id}   → personalized recommendations
  GET /recommend/movie/{movie_id} → similar movies
  GET /health                     → service health check

The API loads pre-trained models at startup. No training happens
during requests — only inference.
"""

import logging
import pickle
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.models.collaborative import ALSRecommender
from src.models.content_based import ContentBasedRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Movie Recommendation API",
    description="Personalized movie recommendations using ALS + Content-Based filtering",
    version="0.1.0",
)

# --- Global model objects (loaded at startup) ---
als_model = None
cb_model = None
movies_df = None
movie_idx_to_info = None
id_mappings = None


@app.on_event("startup")
def load_models():
    """Load trained models and data at startup."""
    global als_model, cb_model, movies_df, movie_idx_to_info, id_mappings

    logger.info("Loading data and training models ...")

    train = pd.read_parquet("data/dev/dev_train.parquet")
    movies_df = pd.read_parquet("data/processed/movies_clean.parquet")

    with open("artifacts/id_mappings.pkl", "rb") as f:
        id_mappings = pickle.load(f)

    # Build reverse mappings (contiguous → original)
    id_mappings["user_map_reverse"] = {
        v: k for k, v in id_mappings["user_map"].items()
    }
    id_mappings["movie_map_reverse"] = {
        v: k for k, v in id_mappings["movie_map"].items()
    }

    # Movie info lookup: movie_idx → {title, genres}
    movie_idx_to_info = {}
    for _, row in movies_df.iterrows():
        movie_idx_to_info[row["movie_idx"]] = {
            "movie_idx": int(row["movie_idx"]),
            "movieId": int(row["movieId"]),
            "title": row["title"],
            "genres": row["genres"],
        }

    # Train ALS
    logger.info("Training ALS model ...")
    als_model = ALSRecommender()
    als_model.fit(train)

    # Train Content-Based
    logger.info("Training Content-Based model ...")
    cb_model = ContentBasedRecommender()
    cb_model.fit(train, movies=movies_df)

    logger.info("All models loaded and ready.")


@app.get("/health")
def health_check():
    """Service health check."""
    return {
        "status": "healthy",
        "models": {
            "als": als_model.is_fitted if als_model else False,
            "content_based": cb_model.is_fitted if cb_model else False,
        },
        "n_movies": len(movie_idx_to_info) if movie_idx_to_info else 0,
    }


@app.get("/recommend/user/{user_id}")
def recommend_for_user(user_id: int, k: int = 10):
    """Get personalized recommendations for a user.

    Parameters:
        user_id: Original MovieLens userId
        k: Number of recommendations (default 10)
    """
    # Map original userId to contiguous index
    user_idx = id_mappings["user_map"].get(user_id)

    if user_idx is None:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found. Valid range: check dataset.",
        )

    # Get ALS recommendations
    rec_indices = als_model.recommend(user_idx, k=k)

    if not rec_indices:
        # Fallback to content-based
        rec_indices = cb_model.recommend(user_idx, k=k)

    # Build response with movie details
    recommendations = []
    for movie_idx in rec_indices:
        info = movie_idx_to_info.get(movie_idx, {})
        if info:
            recommendations.append(info)

    return {
        "user_id": user_id,
        "model": "ALS" if rec_indices else "Content-Based (fallback)",
        "recommendations": recommendations,
    }


@app.get("/recommend/movie/{movie_id}")
def similar_movies(movie_id: int, k: int = 10):
    """Find movies similar to a given movie.

    Parameters:
        movie_id: Original MovieLens movieId
        k: Number of similar movies (default 10)
    """
    # Map original movieId to contiguous index
    movie_idx = id_mappings["movie_map"].get(movie_id)

    if movie_idx is None:
        raise HTTPException(
            status_code=404,
            detail=f"Movie {movie_id} not found.",
        )

    # Get similar movies
    similar = cb_model.similar_movies(movie_idx, k=k)

    # Get source movie info
    source_info = movie_idx_to_info.get(movie_idx, {})

    # Build response
    results = []
    for sim_idx, score in similar:
        info = movie_idx_to_info.get(sim_idx, {})
        if info:
            results.append({
                **info,
                "similarity_score": round(score, 4),
            })

    return {
        "source_movie": source_info,
        "similar_movies": results,
    }


@app.get("/movies/search/{query}")
def search_movies(query: str, limit: int = 10):
    """Search movies by title (simple substring match).

    Useful for finding movie IDs to use with other endpoints.
    """
    query_lower = query.lower()
    results = []

    for _, row in movies_df.iterrows():
        if query_lower in row["title"].lower():
            results.append({
                "movieId": int(row["movieId"]),
                "movie_idx": int(row["movie_idx"]),
                "title": row["title"],
                "genres": row["genres"],
            })
            if len(results) >= limit:
                break

    return {"query": query, "results": results}