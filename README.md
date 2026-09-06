# 🎬 Movie Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movie-recommendation-system-niypzgljuxxyazeiamdn5s.streamlit.app/)

An end-to-end movie recommendation system built on the **MovieLens 25M** dataset.

An end-to-end movie recommendation system built on the **MovieLens 25M** dataset. Compares multiple recommendation approaches — from simple popularity baselines to collaborative filtering — with a strict temporal evaluation protocol that prevents data leakage.

**This is not a notebook with cosine similarity.** It's a production-style ML project with modular code, a FastAPI service, a Streamlit demo, and rigorous evaluation.

---

## Problem Statement

Given a user's movie rating history, recommend the top-10 movies they are most likely to enjoy. Additionally, given any movie, find the 10 most similar movies based on content.

**Two use cases:**
- 🎯 **Personalized recommendations** — "What should User X watch next?"
- 🔍 **Similar movies** — "I liked Interstellar, what else is similar?"

---

## Key Results

| Model | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 |
|-------|-------------|-----------|---------|------------|
| Popularity Baseline | 0.0011 | 0.0109 | 0.0039 | 1.1% |
| Content-Based (TF-IDF) | 0.0053 | 0.0530 | 0.0280 | 5.3% |
| Content-Based (Embeddings) | 0.0026 | 0.0258 | 0.0122 | 2.6% |
| SVD | 0.0018 | 0.0176 | 0.0081 | 1.8% |
| **ALS (best)** | **0.0110** | **0.1105** | **0.0543** | **11.1%** |

**ALS achieves ~10× higher Recall@10 than the popularity baseline and ~2× higher Recall@10 than TF-IDF Content-Based**, demonstrating that collaborative filtering captures user preferences far more effectively than content similarity alone.

### Key Findings

- **TF-IDF beats Embeddings** — our content features are keyword-based (genre tags), not natural language. Embeddings are designed for semantic understanding of sentences, not keyword lists.
- **SVD underperforms ALS** — SVD (Surprise) optimizes for RMSE (rating prediction), while ALS optimizes for ranking. Since we evaluate with ranking metrics, ALS wins.
- **Hybrid (ALS + Content-Based)** — A switching strategy where ALS handles users with ≥5 ratings and Content-Based TF-IDF serves as fallback for cold-start users (<5 ratings). In our evaluation dataset all users have ≥20 ratings, so the fallback is never activated and Hybrid produces the same results as ALS. In production, the fallback would serve new users who lack rating history.

---

## Dataset

### Raw
- **MovieLens 25M** — 25 million ratings from ~162,000 users on ~62,000 movies
- Released 12/2019 by GroupLens Research
- Includes user-generated tags (~1.1M tag applications)

### After temporal split + preprocessing
- **24.5M** training ratings
- **~154K** users, **~32.6K** movies
- **99.5% sparsity** — justifies matrix factorization approach
- **88.1%** of movies have user-generated tags
- ~29.7K movies were removed during filtering (< 5 ratings in training set)

---

## Evaluation Protocol

A strict temporal evaluation that mirrors real-world deployment:

1. **Temporal split first** — before any filtering, to prevent information leakage
2. **Leave-last-1-out** — each user's most recent rating is the test item
3. **Train-only filtering** — minimum rating thresholds computed on training data only
4. **Temporal tag cutoff** — only tags created before the training period end are used
5. **Relevance threshold** — rating ≥ 4.0 = relevant (49.8% of ratings qualify)

**Why temporal, not random split?** A random split lets the model "see" future ratings during training — this inflates metrics unrealistically. Our temporal split reflects the real scenario: train on past, predict the future.

**Why not just RMSE?** This is a ranking problem, not a regression problem. We care about whether the right movies appear in the top-10, not whether we predicted 3.7 vs 4.1.

---

## Models

### Popularity Baseline
Recommends the same popular movies to everyone using Bayesian average (IMDb-style weighted rating) to avoid the "5 ratings, all 5.0" problem.

### Content-Based (TF-IDF)
Represents each movie as a TF-IDF vector of its genres + user tags. Builds a user profile by averaging the vectors of movies they liked (≥ 4.0). Recommends movies most similar to the profile.

### Content-Based (Embeddings)
Uses `sentence-transformers/all-MiniLM-L6-v2` to create dense semantic vectors. Tested to compare against TF-IDF — found to underperform on keyword-based content.

### SVD (Surprise)
Classic matrix factorization that decomposes the user-item matrix into latent factors. Optimizes for RMSE — effective for rating prediction but not optimal for ranking.

### ALS (implicit)
Alternating Least Squares optimized for ranking on sparse matrices. Best performer across all metrics.

### Hybrid (ALS + Content-Based)
Switching strategy: ALS for users with ≥5 ratings, Content-Based TF-IDF fallback for cold-start users (<5 ratings). In the current evaluation dataset, all users have ≥20 ratings, so the cold-start fallback is not activated and Hybrid produces the same results as ALS.

---

## Project Structure

## Project Structure

```
movie-recommendation-system/
│
├── api/
│   └── main.py                  # FastAPI service
│
├── configs/
│   └── config.yaml              # All parameters in one place
│
├── data/
│   ├── raw/ml-25m/              # Raw MovieLens CSVs (gitignored)
│   ├── processed/               # Clean parquet files (gitignored)
│   └── dev/                     # 10% subset for development (gitignored)
│
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory Data Analysis
│
├── scripts/
│   ├── prepare_data.py          # Full data pipeline
│   └── cold_start_analysis.py   # Cold-start threshold experiment
│
├── src/
│   ├── data/
│   │   ├── loader.py            # Load raw CSVs
│   │   ├── preprocessor.py      # Clean, filter, remap
│   │   └── splitter.py          # Temporal train/val/test split
│   │
│   ├── models/
│   │   ├── base.py              # Abstract base recommender
│   │   ├── popularity.py        # Popularity baseline
│   │   ├── content_based.py     # TF-IDF + Embeddings
│   │   ├── collaborative.py     # SVD + ALS
│   │   └── hybrid.py            # ALS + Content-Based switching
│   │
│   ├── evaluation/
│   │   └── metrics.py           # Precision, Recall, NDCG, MAP, HitRate
│   │
│   └── utils/
│       └── config.py            # YAML config loader
│
├── streamlit_app/
│   └── app.py                   # Demo UI
│
├── tests/
│   ├── test_data.py             # Data integrity tests
│   ├── test_evaluation.py       # Metric correctness tests
│   └── test_models.py           # Model contract tests
│
├── .gitignore
├── requirements.txt
└── README.md
```


---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
cd movie-recommendation-system
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 2. Download data

Download [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) and extract to `data/raw/ml-25m/`.

### 3. Run data pipeline

```bash
python scripts/prepare_data.py
```

### 4. Run tests

```bash
pytest tests/ -v
```

### 5. Start API

```bash
uvicorn api.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

### 6. Start Streamlit demo

```bash
streamlit run streamlit_app/app.py
```

---

## API Endpoints

| Endpoint | Description | Example |
|----------|-------------|---------|
| `GET /health` | Service health check | `/health` |
| `GET /recommend/user/{id}` | Personalized top-10 | `/recommend/user/1` |
| `GET /recommend/movie/{id}` | Similar movies | `/recommend/movie/109487` |
| `GET /movies/search/{query}` | Search by title | `/movies/search/interstellar` |

**Example response** for `/recommend/movie/109487` (Interstellar):
```json
{
  "source_movie": {"title": "Interstellar (2014)", "genres": "Sci-Fi|IMAX"},
  "similar_movies": [
    {"title": "Inception (2010)", "similarity_score": 0.42},
    {"title": "Gravity (2013)", "similarity_score": 0.42},
    {"title": "Star Wars: Episode IV (1977)", "similarity_score": 0.39}
  ]
}
```

---

## Technical Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Split strategy | Temporal (not random) | Prevents data leakage — train on past, predict future |
| Filtering | Train-only counts | User/movie thresholds computed without test data |
| Tag cutoff | Temporal | Only tags before training period used in content features |
| Relevance threshold | Rating ≥ 4.0 | 49.8% of ratings qualify — enough signal without being too loose |
| Primary metric | NDCG@10 | Rewards correct items higher in the ranking |
| Content representation | TF-IDF over embeddings | Keyword-based content favors exact matching over semantic similarity |
| CF approach | ALS over SVD | ALS optimizes for ranking; SVD optimizes for RMSE |

---

## Limitations

- **Memory constraints** — Full dataset (25M ratings) requires 16GB+ RAM for training. API and Streamlit demo use a 10% user subset.
- **Static dataset** — No mechanism for incorporating new ratings in real-time.
- **Cold-start evaluation** — Cannot evaluate hybrid cold-start fallback because all users have ≥20 ratings after filtering.
- **No deep learning** — Neural approaches (NCF, autoencoders) were not explored. Classic approaches were prioritized for interpretability and strong baselines.

---

## Future Improvements

- **TMDb enrichment** — Add plot descriptions for richer content-based features with sentence embeddings
- **Neural Collaborative Filtering** — Test whether deep learning improves over ALS
- **Model serialization** — Save trained models to disk instead of retraining at API startup
- **Online learning** — Incorporate new ratings without full retraining
- **A/B testing framework** — Compare models in a simulated online setting
- **Cloud deployment** — Deploy API and Streamlit on cloud infrastructure with proper scaling

---

## Tech Stack

Python · pandas · NumPy · SciPy · scikit-learn · Surprise · implicit · sentence-transformers · FastAPI · Streamlit · pytest

---

## Author

**Konstantinos Traganos**
MSc Data Science — Deree, The American College of Greece