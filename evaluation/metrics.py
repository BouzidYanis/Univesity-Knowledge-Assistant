
def hit_at_k(retrieved, relevant, k):
    """
    Compute the Hit@k metric.

    Args:
        retrieved (list): The list of retrieved document IDs.
        relevant (set): The set of relevant document IDs.
        k (int): The cutoff rank.

    Returns:
        int: 1 if at least one relevant document is in the top-k retrieved documents, 0 otherwise.
    """
    return int(any(doc_id in relevant for doc_id in retrieved[:k]))

def recall_at_k(retrieved, relevant, k):
    """
    Compute the Recall@k metric.

    Args:
        retrieved (list): The list of retrieved document IDs.
        relevant (set): The set of relevant document IDs.
        k (int): The cutoff rank.

    Returns:
        float: The recall at k, i.e., the proportion of relevant documents that are in the top-k retrieved documents.
    """
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def reciprocal_rank(retrieved, relevant):
    """
    Compute the Reciprocal Rank (RR) metric.

    Args:
        retrieved (list): The list of retrieved document IDs.
        relevant (set): The set of relevant document IDs.

    Returns:
        float: The reciprocal rank, i.e., 1 divided by the rank of the first relevant document in the retrieved list.
               Returns 0 if no relevant document is found.
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0