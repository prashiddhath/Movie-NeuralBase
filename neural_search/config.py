import os
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
movies_csv = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
credits_csv = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")

features_weight = {
    "keywords": 1,
    "cast": 1,
    "director": 1,
    "genres": 1,
    "production_companies": 1,
}

model = SentenceTransformer("multi-qa-distilbert-cos-v1")

tfidf_coll_name = "plot_tf-idf"
metadata_coll_name = "metadata_count"
titles_coll_name = "titles"