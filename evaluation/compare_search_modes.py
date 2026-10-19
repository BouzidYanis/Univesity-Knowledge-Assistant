"""
Palier 5 - Compare the 3 search modes (dense, bm25, hybrid) on keyword and semantic questions.

Run from the project root:
    python -m evaluation.compare_search_modes
"""
from src.embedding import search_in_qdrant, build_sparse_query

MODES = ["dense", "bm25", "hybrid"]
TOP_K = 3

# Each question has the chunks expected in the results: (start of the file name, set of pages, numbered from 1)
QUESTIONS = [
    # --- "keywords" questions: article numbers, figures, acronyms, rare words
    {"type": "mots-clés", "question": "Que dit l'article 23 du règlement intérieur ?",
     "expected": [("RI-AU", {16})]},
    {"type": "mots-clés", "question": "Article 11 accès aux locaux",
     "expected": [("RI-AU", {11})]},
    {"type": "mots-clés", "question": "Loi EgAlim",
     "expected": [("Charte-alimentation", {1})]},
    {"type": "mots-clés", "question": "60% de produits durables",
     "expected": [("Charte-alimentation", {1})]},
    {"type": "mots-clés", "question": "Quel est le rôle de la CFVU pour les associations ?",
     "expected": [("Charte_des_Associations", {2, 3, 4})]},
    {"type": "mots-clés", "question": "30 ETCS",
     "expected": [("Guide-des-modalites", {6})]},

    # --- "meaning" questions: reformulated, without the words of the documents
    {"type": "sens", "question": "Est-ce que je peux donner mon mot de passe à un ami ?",
     "expected": [("Charte-pour-lusage", {1, 3})]},
    {"type": "sens", "question": "Quelle part de nourriture bio est servie à la cantine ?",
     "expected": [("Charte-alimentation", {1})]},
    {"type": "sens", "question": "Que faire si un enseignant se moque de moi sans arrêt ?",
     "expected": [("RI-AU", {3, 6, 17})]},
    {"type": "sens", "question": "Comment une association peut-elle obtenir une salle sur le campus ?",
     "expected": [("Charte_des_Associations", {2, 3})]},
    {"type": "sens", "question": "Qui peut présider une association étudiante ?",
     "expected": [("Charte_des_Associations", {1, 2})]},
]


def is_expected(result, expected):
    """True if the result is one of the expected (file, page) chunks."""
    source_file = result["metadata"].get("source_file", "")
    page = result["metadata"].get("page", -1) + 1  # PyMuPDF counts pages from 0
    return any(source_file.startswith(prefix) and page in pages for prefix, pages in expected)


def compare_modes():
    """Print the top-k of each mode side by side and count the hits per mode and question type."""
    hits = {(mode, q_type): 0 for mode in MODES for q_type in ("mots-clés", "sens")}
    totals = {"mots-clés": 0, "sens": 0}

    for item in QUESTIONS:
        totals[item["type"]] += 1
        print(f"\n=== [{item['type']}] {item['question']}")
        for mode in MODES:
            results = search_in_qdrant(item["question"], top_k=TOP_K, mode=mode)
            found = any(is_expected(r, item["expected"]) for r in results)
            hits[(mode, item["type"])] += found
            print(f"  {mode:6} {'✔' if found else '✘'}")
            for r in results:
                mark = "*" if is_expected(r, item["expected"]) else " "
                text = r["text"][:60].replace("\n", " ")
                print(f"     {mark} {r['score']:7.3f}  {r['metadata']['source_file'][:28]:28} p.{r['metadata']['page'] + 1:<3} {text}")

    print(f"\n=== Hit@{TOP_K} par mode (questions où un chunk attendu est dans le top-{TOP_K})")
    print(f"  {'mode':6}  {'mots-clés':>10}  {'sens':>6}")
    for mode in MODES:
        print(f"  {mode:6}  {hits[(mode, 'mots-clés')]:>4}/{totals['mots-clés']:<5}  {hits[(mode, 'sens')]:>2}/{totals['sens']}")


def check_pitfalls():
    """The 3 pitfalls of the palier: stemming, very frequent words, score scale."""
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
    compare_modes()
    check_pitfalls()
