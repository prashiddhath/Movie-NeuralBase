import os
from sentence_transformers import SentenceTransformer

DATA_DIR = os.environ.get("DATA_DIR", "data")
movies_csv = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
credits_csv = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")
TEMPLATE_DIR = os.environ.get("TEMPLATE_DIR", "demo/templates")

# Features used to create the metadata soup, change weight according to need
features_weight = {
    "keywords": 1,
    "cast": 1,
    "director": 1,
    "genres": 1,
    "production_companies": 1,
}

# ML model to use for title embedding, currently a symmetric semantic search model
model = SentenceTransformer("multi-qa-distilbert-cos-v1")

tfidf_coll_name = "plot_tf-idf"
metadata_coll_name = "metadata_count"
titles_coll_name = "titles"

# Maximum number of rows to read from tmbd data set
max_data = os.environ.get("MAX_DATA", None)

# Configure connection to Qdrant cluster
host = os.environ.get("HOST", "localhost")
api_key = os.environ.get("API_KEY", None)
