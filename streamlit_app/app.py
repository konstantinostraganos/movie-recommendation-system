"""
Streamlit demo for Movie Recommendation System.

Three modes:
1. Personalized: select a user → get ALS recommendations
2. Similar movies: select a movie → get content-based similar movies
3. Model Performance: comparison table of all models
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pickle
import pandas as pd
import streamlit as st

from src.models.collaborative import ALSRecommender
from src.models.content_based import ContentBasedRecommender


@st.cache_resource
def load_models():
    """Load data and train models (cached — runs only once)."""
    train = pd.read_parquet("data/dev/dev_train.parquet")
    movies = pd.read_parquet("data/processed/movies_clean.parquet")

    with open("artifacts/id_mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    als = ALSRecommender()
    als.fit(train)

    cb = ContentBasedRecommender()
    cb.fit(train, movies=movies)

    return als, cb, movies, mappings, train


def get_movie_info(movies_df, movie_idx):
    """Get full movie info from movie_idx."""
    row = movies_df[movies_df["movie_idx"] == movie_idx]
    if len(row) > 0:
        return row.iloc[0]
    return None


# --- Page config ---
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Movie Recommendation System")
st.markdown(
    "Personalized recommendations using **ALS Collaborative Filtering** "
    "and **Content-Based TF-IDF** similarity."
)

# --- Load models ---
with st.spinner("Loading models (first time only) ..."):
    als, cb, movies, mappings, train = load_models()

# --- Sidebar ---
st.sidebar.header("Choose mode")
mode = st.sidebar.radio(
    "What would you like?",
    [
        "🎯 Personalized Recommendations",
        "🔍 Similar Movies",
        "📊 Model Performance",
    ],
)

# About section
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.markdown(
    "End-to-end recommendation system trained on "
    "**MovieLens 25M** (25 million ratings, 32K movies).\n\n"
    "**Models:** ALS · TF-IDF Content-Based · Hybrid\n\n"
    "**Evaluation:** Temporal split, no data leakage"
)


# --- Mode: Personalized ---
if mode == "🎯 Personalized Recommendations":
    st.header("🎯 Personalized Recommendations")
    st.markdown("Select a user to get their top-10 movie recommendations.")

    reverse_user_map = {v: k for k, v in mappings["user_map"].items()}
    valid_user_indices = sorted(train["user_idx"].unique())

    user_idx = st.selectbox(
        "Select User (index)",
        valid_user_indices[:100],
        format_func=lambda x: (
            f"User {reverse_user_map.get(x, x)} "
            f"({train[train['user_idx']==x].shape[0]} ratings)"
        ),
    )

    if st.button("Get Recommendations", key="user_rec"):
        recs = als.recommend(user_idx, k=10)

        if recs:
            st.subheader("Top 10 Recommendations")

            user_history = train[
                train["user_idx"] == user_idx
            ].sort_values("rating", ascending=False).head(5)

            with st.expander(
                "📊 User's top-rated movies (training data)"
            ):
                for _, row in user_history.iterrows():
                    info = get_movie_info(movies, row["movie_idx"])
                    if info is not None:
                        st.write(
                            f"⭐ **{row['rating']}** — "
                            f"{info['title']} ({info['genres']})"
                        )

            cols = st.columns(2)
            for i, movie_idx in enumerate(recs):
                info = get_movie_info(movies, movie_idx)
                if info is not None:
                    with cols[i % 2]:
                        st.markdown(
                            f"**{i+1}. {info['title']}**\n\n"
                            f"Genres: {info['genres']}"
                        )
                        st.divider()
        else:
            st.warning("No recommendations found for this user.")


# --- Mode: Similar Movies ---
elif mode == "🔍 Similar Movies":
    st.header("🔍 Find Similar Movies")
    st.markdown(
        "Select a movie to find the 10 most similar movies "
        "based on genres and tags."
    )

    movie_options = (
        movies[["movie_idx", "title", "genres"]]
        .sort_values("title")
        .reset_index(drop=True)
    )

    selected_title = st.selectbox(
        "Search for a movie",
        movie_options["title"].tolist(),
        index=None,
        placeholder="Type to search ...",
    )

    if selected_title:
        selected_row = movie_options[
            movie_options["title"] == selected_title
        ].iloc[0]
        selected_idx = selected_row["movie_idx"]

        st.markdown(
            f"**Selected:** {selected_title} — "
            f"*{selected_row['genres']}*"
        )

        if st.button("Find Similar Movies", key="similar"):
            similar = cb.similar_movies(selected_idx, k=10)

            if similar:
                st.subheader("Top 10 Similar Movies")

                cols = st.columns(2)
                for i, (movie_idx, score) in enumerate(similar):
                    info = get_movie_info(movies, movie_idx)
                    if info is not None:
                        with cols[i % 2]:
                            st.markdown(
                                f"**{i+1}. {info['title']}**\n\n"
                                f"Genres: {info['genres']}\n\n"
                                f"Similarity: {score:.1%}"
                            )
                            st.divider()
            else:
                st.warning("No similar movies found.")


# --- Mode: Model Performance ---
elif mode == "📊 Model Performance":
    st.header("📊 Model Comparison")
    st.markdown(
        "All models evaluated on the same test set using "
        "temporal split (no data leakage). "
        "Relevance threshold: rating ≥ 4.0."
    )

    results = pd.DataFrame({
        "Model": [
            "Popularity",
            "Content-Based (TF-IDF)",
            "Content-Based (Embeddings)",
            "SVD",
            "ALS",
        ],
        "Precision@10": [0.0011, 0.0053, 0.0026, 0.0018, 0.0110],
        "Recall@10": [0.0109, 0.0530, 0.0258, 0.0176, 0.1105],
        "NDCG@10": [0.0039, 0.0280, 0.0122, 0.0081, 0.0543],
        "HitRate@10": [0.0109, 0.0530, 0.0258, 0.0176, 0.1105],
        "MAP@10": [0.0020, 0.0205, 0.0082, 0.0053, 0.0374],
    })

    st.dataframe(
        results.style.highlight_max(
            subset=[
                "Precision@10", "Recall@10", "NDCG@10",
                "HitRate@10", "MAP@10",
            ],
            color="#d4edda",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.subheader("Key Findings")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Model", "ALS", "10x vs Popularity")
    with col2:
        st.metric("Best NDCG@10", "0.0543", "+94% vs TF-IDF")
    with col3:
        st.metric("Hit Rate", "11.1%", "1 in 9 users")

    st.markdown(
        """
        **Key takeaways:**
        - **ALS** dominates all metrics — collaborative filtering
          captures user preferences far better than content alone
        - **TF-IDF** beats **Embeddings** because our content is
          keyword-based (tags), not natural language
        - **SVD** underperforms because it optimizes for RMSE,
          not ranking
        - **Hybrid note:** A switching hybrid (ALS + Content-Based
          fallback) is implemented for cold-start users (<5 ratings).
          In this evaluation all users have ≥20 ratings, so the
          fallback never activates — metrics match ALS.
        """
    )