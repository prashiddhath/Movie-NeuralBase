from qdrant_client.models import VectorParams
from qdrant_client import QdrantClient

from .config import (
    distance,
    tfidf_coll_name,
    titles_coll_name,
    metadata_coll_name,
    model,
)
from .prepare_data import load_movie_data

from .metric import (
    construct_tfidf_plot,
    construct_metadata_vectors,
    construct_title_vectors,
)

def upload_data(qdrant_client, collection_name, vectors, payload=None, ids=None):
    size = vectors.shape[1]
    re_init_collection(qdrant_client, collection_name, size)

    qdrant_client.upload_collection(
        collection_name=collection_name,
        vectors=vectors,
        payload=payload,
        ids=ids,
        batch_size=256,  # How many vectors will be uploaded in a single request?
    )


def re_init_collection(qdrant_client, collection_name, size):
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=size, distance=distance),
    )


if __name__ == "__main__":
    qdrant_client = QdrantClient(host="localhost", port=6333)

    df = load_movie_data()

    vectors_tfidf, payload = construct_tfidf_plot(df)
    vectors_metadata, _ = construct_metadata_vectors(df)

    vectors_title = construct_title_vectors(model, df["title"])

    upload_data(
        qdrant_client,
        titles_coll_name,
        vectors_title,
        payload,
    )
    upload_data(
        qdrant_client,
        tfidf_coll_name,
        vectors_tfidf,
        payload,
    )
    upload_data(
        qdrant_client,
        metadata_coll_name,
        vectors_metadata,
        payload,
    )
