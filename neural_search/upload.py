from qdrant_client.models import VectorParams
from qdrant_client import QdrantClient
from neural_search.config import host, api_key

from typing import List, Optional


def upload_data(
    qdrant_client: QdrantClient,
    collection_name: str,
    vectors: List,
    payload: Optional[List] = None,
    ids: Optional[List] = None,
) -> None:
    """
    Uploads Points to a collect on the provided Qdrant cluster.

    """
    size = vectors.shape[1]
    re_init_collection(qdrant_client, collection_name, size)

    qdrant_client.upload_collection(
        collection_name=collection_name,
        vectors=vectors,
        payload=payload,
        ids=ids,
        batch_size=256,
    )


def re_init_collection(qdrant_client: QdrantClient, collection_name: str, size) -> None:
    """
    Re-initialises the collection, ensuring it doesn't consist of old data. As the free-tier Qdrant
    cluster only offers 1GB of memory, the on_disk_payload has been set to True which stores the
    payload to disk instead of memory.

    """
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=size, distance="Cosine"),
        on_disk_payload=True,
    )


def establish_conn() -> QdrantClient:
    """
    Connects to the Qdrant cluster.

    """
    if api_key:
        qdrant_client = QdrantClient(url=host, api_key=api_key)
    else:
        qdrant_client = QdrantClient(host=host)

    return qdrant_client
