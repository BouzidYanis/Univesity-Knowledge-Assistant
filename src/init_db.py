from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def init_collection(
        client: QdrantClient,
        collection_name: str,
        vector_size: int,
        distance: Distance.COSINE,
) -> None:
    """
    Initialize a Qdrant collection with the specified parameters.

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The name of the collection to create.
        vector_size (int): The size of the vectors to be stored in the collection.
        distance (Distance): The distance metric to use for vector similarity.
    """
    try:
        # Check if the collection already exists
        if client.collection_exists(collection_name=collection_name):
            return  # Collection already exists, no need to create it again
        
        # Create the collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance)
        )
        print(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        print(f"Error initializing collection '{collection_name}': {e}")

