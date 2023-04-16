from qdrant_client import QdrantClient

from neural_search.config import (
    tfidf_coll_name,
    titles_coll_name,
    metadata_coll_name,
    model,
)
from neural_search.prepare_data import load_movie_data
from neural_search.upload import upload_data, establish_conn

from neural_search.metric import (
    construct_tfidf_plot,
    construct_metadata_vectors,
    construct_title_vectors,
)

from parse import parse_flags

if __name__ == "__main__":
    host, api_key, max_data = parse_flags()
    qdrant_client = establish_conn(host, api_key)

    df = load_movie_data(max_data)

    print("Constructing movie plot TF-IDF vectors...")
    vectors_tfidf, payload = construct_tfidf_plot(df)

    print("Constructing movie metadata Count vectors...")
    vectors_metadata, _ = construct_metadata_vectors(df)

    print("Embedding movie titles...")
    vectors_title = construct_title_vectors(model, df["title"])

    print("Uploading embedded title vectors to Qdrant cluster...")
    upload_data(
        qdrant_client,
        titles_coll_name,
        vectors_title,
        payload,
    )

    print("Uploading plot-based `TF-IDF vectors to Qdrant cluster...")
    upload_data(
        qdrant_client,
        tfidf_coll_name,
        vectors_tfidf,
        payload,
    )

    print("Uploading metadata based Count vectors to Qdrant cluster...")
    upload_data(
        qdrant_client,
        metadata_coll_name,
        vectors_metadata,
        payload,
    )

    print("Successfully uploaded all vectors to Qdrant!")