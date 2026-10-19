"""
Compare the 3 search modes (dense, bm25, hybrid) on the "keywords" and "reformulated" questions of the dataset.

Run from the project root:
    python -m evaluation.compare_search_modes [--split dev|test]
"""
import argparse

from evaluation.dataset import load_dataset
from src.embedding import build_sparse_query, search_in_qdrant

MODES = ["dense", "bm25", "hybrid"]
TOP_K = 3

# Dataset category -> label of the group: what BM25 must find / what the dense search must find
GROUPS = {"mots_cles": "mots-clés", "reformulee": "sens"}


def is_expected(result, expected_sources):
    """True if the result is one of the expected (file, page) passages."""
    source_file = result["metadata"].get("source_file", "")
    page = result["metadata"].get("page", -1) + 1  # PyMuPDF counts pages from 0, the dataset from 1
    return any(source_file == source.file and page in source.pages for source in expected_sources)


def compare_modes(items):
    """Print the top-k of each mode side by side and count the hits per mode and question group."""
    hits = {(mode, group): 0 for mode in MODES for group in GROUPS.values()}
    totals = {group: 0 for group in GROUPS.values()}

    for item in items:
        group = GROUPS[item.category]
        totals[group] += 1
        print(f"\n=== [{group}] {item.id} : {item.question}")
        for mode in MODES:
            results = search_in_qdrant(item.question, top_k=TOP_K, mode=mode)
            found = any(is_expected(r, item.expected_sources) for r in results)
            hits[(mode, group)] += found
            print(f"  {mode:6} {'✔' if found else '✘'}")
            for r in results:
                mark = "*" if is_expected(r, item.expected_sources) else " "
                text = r["text"][:60].replace("\n", " ")
                print(f"     {mark} {r['score']:7.3f}  {r['metadata']['source_file'][:28]:28} p.{r['metadata']['page'] + 1:<3} {text}")

    print(f"\n=== Hit@{TOP_K} par mode (questions où un passage attendu est dans le top-{TOP_K})")
    print(f"  {'mode':6}  {'mots-clés':>10}  {'sens':>6}")
    for mode in MODES:
        print(f"  {mode:6}  {hits[(mode, 'mots-clés')]:>4}/{totals['mots-clés']:<5}  {hits[(mode, 'sens')]:>2}/{totals['sens']}")


def check_pitfalls():
    """The 3 pitfalls of the hybrid search: stemming, very frequent words, score scale."""
    print("\n=== Piège 1 : stemming (même indice = même racine pour BM25)")
    for word in ["inscription", "inscriptions", "inscrit", "inscrire"]:
        print(f"  {word:13} -> {build_sparse_query(word).indices}")

    print("\n=== Piège 2 : mots très fréquents (IDF faible -> BM25 peu discriminant)")
    for r in search_in_qdrant("université étudiant", top_k=TOP_K, mode="bm25"):
        print(f"  {r['score']:7.3f}  {r['metadata']['source_file'][:28]:28} p.{r['metadata']['page'] + 1}")

    print("\n=== Piège 3 : l'échelle des scores dépend du mode")
    for mode in MODES:
        scores = [r["score"] for r in search_in_qdrant("Loi EgAlim", top_k=TOP_K, mode=mode)]
        print(f"  {mode:6} {[round(s, 3) for s in scores]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], help="Only the dev or test questions (default: all)")
    args = parser.parse_args()

    compare_modes(load_dataset(split=args.split, categories=GROUPS))
    check_pitfalls()
