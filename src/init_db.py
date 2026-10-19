from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, SparseVectorParams, VectorParams

# Names of the two vectors stored for each point (must be the same when storing and searching)
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

def init_collection(
        client: QdrantClient,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
) -> None:
    """
    Initialize a Qdrant collection for hybrid search: one dense vector (semantic search)
    and one sparse vector (BM25 keyword search) per point.

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The name of the collection to create.
        vector_size (int): The size of the dense vectors to be stored in the collection.
        distance (Distance): The distance metric to use for dense vector similarity.
    """
    try:
        # Check if the collection already exists
        if client.collection_exists(collection_name=collection_name):
            return  # Collection already exists, no need to create it again

        # Create the collection
        client.create_collection(
            collection_name=collection_name,
            # Named dense vector (embeddings e5)
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=vector_size, distance=distance),
            },
            # Named sparse vector (BM25): Qdrant computes the IDF from the whole collection
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
            },
        )
        print(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        print(f"Error initializing collection '{collection_name}': {e}")
