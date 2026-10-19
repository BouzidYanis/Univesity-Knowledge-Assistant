from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.init_db import init_collection, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
import torch



client = QdrantClient(url="http://localhost:6333")

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

_embedding_model = None
_sparse_model = None

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
            model_name=EMBEDDING_MODEL_NAME,
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

def get_sparse_model():
    """
    Load the BM25 sparse encoder once and reuse it (indexing and querying must use the same encoder).

    Returns:
        SparseTextEmbedding: The BM25 encoder.
    """
    global _sparse_model
    if _sparse_model is None:
        # language="french" -> French stopwords and stemming ("inscriptions" ~ "inscription")
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25", language="french")
    return _sparse_model

def build_sparse_vectors(chunks):
    """
    Build one BM25 sparse vector per chunk, in the same order as the chunks.

    Args:
        chunks (list): A list of document chunks.
    Returns:
        list: The sparse vectors (models.SparseVector with indices and values).
    """
    # No "passage: " prefix here: it is specific to e5, BM25 works on the raw words
    texts = [chunk.page_content for chunk in chunks]

    # embed() returns a generator: it is consumed once by the list comprehension
    return [
        models.SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())
        for embedding in get_sparse_model().embed(texts)
    ]

def build_sparse_query(query):
    """
    Build the BM25 sparse vector of a question.

    Args:
        query (str): The search query.
    Returns:
        models.SparseVector: The sparse vector of the query.
    """
    # query_embed() (not embed()) for questions: weight 1 per word, Qdrant applies the IDF.
    # It returns a generator: next() takes its only element
    embedding = next(iter(get_sparse_model().query_embed(query)))
    return models.SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())

def store_embbedings_in_qdrant(chunks, vector_store, sparse_vectors=None, collection_name="knowledge_base"):
    """
    Store the dense and sparse (BM25) vectors in a Qdrant collection.

    Args:
        chunks (list): A list of document chunks.
        vector_store: The constructed vector store (dense vectors).
        sparse_vectors (list): The BM25 sparse vectors, computed from the chunks if not provided.
        collection_name (str): The name of the Qdrant collection to store the embeddings.
    """
    # Initialize the Qdrant collection
    vector_size = len(vector_store[0]) if vector_store else 0
    init_collection(client, collection_name, vector_size=vector_size, distance=models.Distance.COSINE)

    if sparse_vectors is None:
        sparse_vectors = build_sparse_vectors(chunks)

    # Prepare points for insertion: each point has a dense and a sparse named vector
    points = []
    for i, chunk in enumerate(chunks):
        point = models.PointStruct(
            id=i,
            vector={
                DENSE_VECTOR_NAME: vector_store[i],
                SPARSE_VECTOR_NAME: sparse_vectors[i],
            },
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

def search_in_qdrant(query, collection_name="knowledge_base", top_k=5, mode="hybrid"):
    """
    Search for similar documents in the Qdrant collection based on the query.

    Args:
        query (str): The search query.
        collection_name (str): The name of the Qdrant collection to search in.
        top_k (int): The number of top results to return.
        mode (str): The search mode, either "hybrid" (dense + sparse), "dense" (only dense vectors), or "bm25" (only sparse vectors).
    Returns:
        list: A list of search results with their metadata and text.

    Raises:
        Exception: The Qdrant error if the search fails (e.g. Qdrant down, missing collection).
    """
    if mode not in ("hybrid", "dense", "bm25"):
        raise ValueError("Invalid mode. Please choose 'hybrid', 'dense', or 'bm25'.")

    # Perform the search in Qdrant
    try:
        if mode == "dense":
            search_results = client.query_points(
                collection_name=collection_name,
                query=get_embedding_model().embed_query(f"query: {query}"),
                using=DENSE_VECTOR_NAME,  # the collection has 2 named vectors: search with the dense one
                limit=top_k
            )
        elif mode == "bm25":
            search_results = client.query_points(
                collection_name=collection_name,
                query=build_sparse_query(query),  # no "query: " prefix: it is specific to e5
                using=SPARSE_VECTOR_NAME,  # the collection has 2 named vectors: search with the sparse one
                limit=top_k
            )
        else:  # hybrid
            # Each prefetch runs its own search, then RRF fuses the two rankings.
            # Prefetch limit > top_k so that the fusion has enough candidates
            prefetch_limit = max(20, top_k)
            search_results = client.query_points(
                collection_name=collection_name,
                prefetch=[
                    models.Prefetch(
                        query=get_embedding_model().embed_query(f"query: {query}"),
                        using=DENSE_VECTOR_NAME,
                        limit=prefetch_limit,
                    ),
                    models.Prefetch(
                        query=build_sparse_query(query),
                        using=SPARSE_VECTOR_NAME,
                        limit=prefetch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                # All the fused candidates (at most 2 x prefetch_limit): the cut to top_k is done after the
                # tie-break below, otherwise Qdrant picks arbitrarily between ex aequo at the limit
                limit=2 * prefetch_limit
            )

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

        # RRF gives many ex aequo (1st in one list + 2nd in the other = 2nd + 1st) that Qdrant returns in any order:
        # break the ties by id so that the same question always gets the same ranking
        results.sort(key=lambda r: (-r["score"], str(r["id"])))
        return results[:top_k]

    except Exception as e:
        # Re-raise: a Qdrant failure must not look like "no relevant document" to the caller
        print(f"Error searching in Qdrant: {e}")
        raise
