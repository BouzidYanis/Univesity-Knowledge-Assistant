import torch
from sentence_transformers import CrossEncoder

# Multilingual cross-encoder: the documents are in French (ms-marco-MiniLM is English only)
RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

_reranker_model = None

def get_reranker() -> CrossEncoder:
    """
    Load the reranker model once and reuse it.

    Returns:
        CrossEncoder: The reranker model.
    """
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(
            RERANKER_MODEL_NAME,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    return _reranker_model

def rerank(query, results, top_k=5):
    """
    Rerank the results using the reranker model.

    Args:
        query (str): The search query.
        results (list): The initial search results to rerank (returned by search_in_qdrant).
        top_k (int): The number of top results to return after reranking.

    Returns:
        list: The reranked results (copies with a "rerank_score" key), sorted by score in descending order.
    """
    # Nothing to rerank: no need to load the model
    if not results:
        return []

    # Prepare pairs of (query, result_text) for scoring (no e5 prefix: raw text)
    pairs = [(query, r["text"]) for r in results]
    scores = get_reranker().predict(pairs)

    # Attach scores to copies of the results (the input list is not modified),
    # float() converts numpy.float32 to a Python float (JSON serializable)
    reranked_results = [
        {**r, "rerank_score": float(score)}
        for r, score in zip(results, scores)
    ]
    reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)

    return reranked_results[:top_k]
