"""
Check the evaluation dataset: schema, unique ids, consistency of the fields, sources present in Qdrant, statistics.

Run from the project root (Qdrant must be running for the source check):
    python -m evaluation.validate_dataset [--path evaluation/dataset/eval_dataset_v1.jsonl] [--no-qdrant]

Exit code 1 if a problem is found.
"""
import argparse
import sys
from collections import Counter, defaultdict

from evaluation.dataset import DEFAULT_DATASET, load_dataset

COLLECTION_NAME = "knowledge_base"

# Coverage plan of the v1 (see evaluation/dataset/README.md): minimum number of questions per category
COVERAGE_PLAN = {
    "factuelle": 15, "mots_cles": 8, "reformulee": 8, "multi_sources": 5, "partielle": 4,
    "non_couverte": 6, "hors_domaine": 4, "conversation": 3, "suivi": 5, "detournement": 2,
}
MAX_SYNTHETIC_SHARE = 0.30

# Behaviour -> router type that must go with it
BEHAVIOR_TYPES = {
    "reponse": {"documentation"},
    "reponse_partielle": {"documentation"},
    "refus": {"documentation", "hors_domaine"},  # documentation for questions not covered by the corpus
    "conversation": {"conversation"},
}


def normalize(text):
    return " ".join(text.lower().split())


def check_items(items):
    """Checks that don't need Qdrant. Returns the list of errors."""
    errors = []

    for name, counter in (("id", Counter(i.id for i in items)),
                          ("question", Counter((normalize(i.question), str(i.history)) for i in items))):
        errors += [f"{name} en double : {value!r}" for value, count in counter.items() if count > 1]

    for item in items:
        where = f"[{item.id}]"
        if item.expected_type not in BEHAVIOR_TYPES[item.expected_behavior]:
            errors.append(f"{where} expected_type={item.expected_type} incohérent avec expected_behavior={item.expected_behavior}")

        if item.expected_behavior in ("reponse", "reponse_partielle"):
            if not item.expected_sources:
                errors.append(f"{where} une réponse attendue exige des expected_sources")
            if not item.reference_answer:
                errors.append(f"{where} une réponse attendue exige une reference_answer")
            if not item.key_facts:
                errors.append(f"{where} une réponse attendue exige des key_facts")
        elif item.expected_sources:
            errors.append(f"{where} un {item.expected_behavior} ne doit pas avoir de expected_sources")

        if item.category == "suivi" and not item.history:
            errors.append(f"{where} une question de suivi exige un history")
        if item.category == "partielle" and item.expected_behavior != "reponse_partielle":
            errors.append(f"{where} une question partielle doit attendre une reponse_partielle")
        if any(not fact.strip() for fact in item.key_facts):
            errors.append(f"{where} key_fact vide")

    return errors


def check_sources_in_qdrant(items):
    """Every expected file must be indexed, and every expected page must exist for this file."""
    from src.embedding import client

    pages_by_file = defaultdict(set)
    offset = None
    while True:
        points, offset = client.scroll(COLLECTION_NAME, limit=256, offset=offset, with_payload=["source_file", "page"])
        for point in points:
            # PyMuPDF counts pages from 0, the dataset counts from 1
            pages_by_file[point.payload["source_file"]].add(point.payload["page"] + 1)
        if offset is None:
            break

    errors = []
    for item in items:
        for source in item.expected_sources:
            if source.file not in pages_by_file:
                errors.append(f"[{item.id}] fichier non indexé : {source.file!r}")
                continue
            missing = sorted(set(source.pages) - pages_by_file[source.file])
            if missing:
                errors.append(f"[{item.id}] pages sans chunk dans {source.file} : {missing}")
    return errors


def print_statistics(items):
    print(f"\n{len(items)} questions")

    by_category = Counter(i.category for i in items)
    print("\nPar catégorie (réel / plan) :")
    for category, planned in COVERAGE_PLAN.items():
        flag = "" if by_category[category] >= planned else "   <-- sous le plan"
        print(f"  {category:14} {by_category[category]:3} / {planned}{flag}")

    by_file = Counter(s.file for i in items for s in i.expected_sources)
    print("\nPar document (questions citant ce document) :")
    for file, count in by_file.most_common():
        print(f"  {count:3}  {file}")

    for field in ("expected_behavior", "difficulty", "origin", "split"):
        counter = Counter(str(getattr(i, field)) for i in items)
        print(f"\nPar {field} : " + ", ".join(f"{k}={v}" for k, v in sorted(counter.items())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=DEFAULT_DATASET)
    parser.add_argument("--no-qdrant", action="store_true", help="Skip the check of the sources in Qdrant")
    args = parser.parse_args()

    try:
        items = load_dataset(args.path)  # schema check (Pydantic)
    except ValueError as e:
        print(f"ERREUR de schéma : {e}")
        sys.exit(1)

    errors = check_items(items)
    if not args.no_qdrant:
        errors += check_sources_in_qdrant(items)

    synthetic_share = sum(i.origin == "synthetique" for i in items) / len(items)
    if synthetic_share > MAX_SYNTHETIC_SHARE:
        errors.append(f"{synthetic_share:.0%} de questions synthétiques (maximum {MAX_SYNTHETIC_SHARE:.0%})")

    print_statistics(items)

    if errors:
        print(f"\n{len(errors)} problème(s) :")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    print("\nDataset valide.")


if __name__ == "__main__":
    main()
