from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.init_db import init_collection
from langchain_huggingface import HuggingFaceEmbeddings
import torch



client = QdrantClient(url="http://localhost:6333")

_embedding_model = None

def get_embedding_model():
    """
    Load the embedding model once and reuse it (indexing and querying must use the same model).

    Returns:
        HuggingFaceEmbeddings: The embedding model.
    """
    global _embedding_model
    if _embedding_model is None:
        #the embedding_size is 384
        _embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
    return _embedding_model

def build_vector_store(chunks):
    """
    Build a vector store from the provided document chunks.

    Args:
        chunks (list): A list of document chunks.
    Returns:
        vector_store: The constructed vector store.
    """
    # 4. Création d'un vecteur d'embeddings pour chaque chunk
    # e5 models expect the "passage: " prefix for documents ("query: " for questions)
    texts = [f"passage: {chunk.page_content}" for chunk in chunks]

    vector = get_embedding_model().embed_documents(texts)
    return vector

def store_embbedings_in_qdrant(chunks, vector_store, collection_name="knowledge_base"):
    """
    Store the embeddings in a Qdrant collection.

    Args:
        chunks (list): A list of document chunks.
        vector_store: The constructed vector store.
        collection_name (str): The name of the Qdrant collection to store the embeddings.
    """
    # Initialize the Qdrant collection
    vector_size = len(vector_store[0]) if vector_store else 0
    init_collection(client, collection_name, vector_size=vector_size, distance=models.Distance.COSINE)

    # Prepare points for insertion
    points = []
    for i, chunk in enumerate(chunks):
        point = models.PointStruct(
            id=i,
            vector=vector_store[i],
            payload={**chunk.metadata, "text": chunk.page_content}  # Store text along with metadata
        )
        points.append(point)

    # Insert points into the Qdrant collection
    try:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Successfully stored {len(points)} embeddings in Qdrant collection '{collection_name}'.")
    except Exception as e:
        print(f"Error storing embeddings in Qdrant: {e}")

def search_in_qdrant(query, collection_name="knowledge_base", top_k=5):
    """
    Search for similar documents in the Qdrant collection based on the query.

    Args:
        query (str): The search query.
        collection_name (str): The name of the Qdrant collection to search in.
        top_k (int): The number of top results to return.

    Returns:
        list: A list of search results with their metadata and text.
    """
    # Embed the query
    query_vector = get_embedding_model().embed_query(f"query: {query}")

    # Perform the search in Qdrant
    try:
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k
        )
        print(search_results)
        # Extract relevant information from the search results
        results = []
        for result in search_results.points:
            payload = result.payload
            results.append({
                "id": result.id,
                "score": result.score,
                "metadata": {k: v for k, v in payload.items() if k != "text"},
                "text": payload.get("text", "")
            })
        
        return results

    except Exception as e:
        print(f"Error searching in Qdrant: {e}")
        return []