from qdrant_client.models import VectorParams
from qdrant_client import QdrantClient

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
        vectors_config=VectorParams(size=size, distance="Cosine"),
        on_disk_payload=True
    )

def establish_conn(host, api_key):
    if api_key:
        qdrant_client = QdrantClient(url=host, api_key=api_key)
    else:
        qdrant_client = QdrantClient(host=host)
    
    return qdrant_client