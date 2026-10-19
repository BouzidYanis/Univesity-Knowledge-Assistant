"""
Evaluation of the generated answers (minimal version, no LLM judge): behaviour, key facts, citations.

1. Generation: answer() is called ONCE per question (with its history); the answers and the excerpts given
   to the LLM are saved in evaluation/results/generation_<label>_<date>.json.
2. Scoring: the metrics are computed from that file, so they can be recomputed without calling the LLM again.

Run from the project root (1 or 2 LLM calls per question: start with a few categories):
    python -m evaluation.eval_generation --split dev --label baseline
    python -m evaluation.eval_generation --split dev --categories factuelle partielle --label essai
    python -m evaluation.eval_generation --answers evaluation/results/generation_baseline_<date>.json
"""
import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from evaluation.dataset import DEFAULT_DATASET, load_dataset
from evaluation.eval_retriever import RESULTS_DIR, git_commit, relevant_pages
from evaluation.generation_metrics import citation_metrics, key_fact_coverage
from evaluation.out_of_scope_questions import is_refusal
from src.rag import answer, format_history

ANSWERING = ("reponse", "reponse_partielle")


# ---------- 1. Generation ----------

def generate_answers(items):
    """Call answer() once per question and keep everything needed to score it later (excerpts included)."""
    rows = []
    for item in items:
        history = format_history([message.model_dump() for message in item.history])
        start = time.perf_counter()
        try:
            result = answer(item.question, history=history)
            error = None
        except Exception as e:  # one failing question (LLM quota, timeout) must not lose the whole run
            result, error = {"answer": "", "sources": [], "type": None, "rewritten_question": None}, str(e)
        rows.append({
            "id": item.id,
            "category": item.category,
            "question": item.question,
            "rewritten_question": result.get("rewritten_question"),
            "type": result.get("type"),
            "answer": result["answer"],
            "sources": result["sources"],  # with the text of each excerpt
            "error": error,
            "seconds": round(time.perf_counter() - start, 1),
        })
        print(f"[{item.category:14}] {item.id:28} {'ERREUR ' + error[:60] if error else result['answer'][:70]!r}")
    return rows


# ---------- 2. Scoring ----------

def score_row(row, item):
    """Metrics of one answer, from the saved row and the dataset item."""
    refusal = is_refusal(row)
    if item.expected_behavior == "refus":
        behavior_ok = refusal
    elif item.expected_behavior == "conversation":
        behavior_ok = not refusal and not row["sources"]
    else:
        behavior_ok = not refusal

    scores = {"behavior_ok": behavior_ok, "refusal": refusal, "router_ok": row["type"] == item.expected_type}
    if item.expected_behavior in ANSWERING:
        # A refusal of an answerable question finds no fact: it counts as 0, not as "not measured"
        scores["key_facts"] = key_fact_coverage(row["answer"], item.key_facts)
        if not refusal and row["sources"]:
            scores.update(citation_metrics(row["answer"], row["sources"], relevant_pages(item)))
    return scores


def ratio(rows, condition):
    """'k/n' for the rows that meet the condition (n/a for an empty list)."""
    return f"{sum(bool(condition(row)) for row in rows)}/{len(rows)}" if rows else "n/a"


def mean(values):
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 2) if values else None


def aggregate(rows):
    """Global metrics over scored rows ({"expected_behavior", "scores", "seconds", ...})."""
    refuse = [row for row in rows if row["expected_behavior"] == "refus"]
    answering = [row for row in rows if row["expected_behavior"] in ANSWERING]
    conversations = [row for row in rows if row["expected_behavior"] == "conversation"]
    with_facts = [row for row in answering if row["scores"].get("key_facts") is not None]
    cited = [row for row in answering if "has_citation" in row["scores"]]
    return {
        "questions": len(rows),
        "comportement_correct": ratio(rows, lambda row: row["scores"]["behavior_ok"]),
        "refus_corrects": ratio(refuse, lambda row: row["scores"]["refusal"]),
        "faux_refus": ratio(answering, lambda row: row["scores"]["refusal"]),
        "conversation_ok": ratio(conversations, lambda row: row["scores"]["behavior_ok"]),
        "routeur_correct": ratio(rows, lambda row: row["scores"]["router_ok"]),
        "faits_cles_moyenne": mean(row["scores"]["key_facts"] for row in with_facts),
        "faits_cles_complets": ratio(with_facts, lambda row: row["scores"]["key_facts"] == 1),
        "reponses_avec_citation": ratio(cited, lambda row: row["scores"]["has_citation"]),
        "citations_invalides": sum(row["scores"]["invalid"] for row in cited),
        "citations_page_attendue": mean(row["scores"]["on_expected_page"] for row in cited),
        "erreurs": sum(bool(row["error"]) for row in rows),
        "latence_moyenne_s": mean(row["seconds"] for row in rows),
    }


def score_run(rows, items_by_id):
    """Add the scores to every row, then aggregate globally and per category."""
    for row in rows:
        item = items_by_id[row["id"]]
        row["expected_behavior"] = item.expected_behavior
        row["key_facts"] = item.key_facts
        row["scores"] = score_row(row, item)

    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    return {
        "global": aggregate(rows),
        "par_categorie": {category: aggregate(group) for category, group in sorted(by_category.items())},
    }


# ---------- 3. Printing ----------

def print_metrics(metrics):
    print("\n=== Métriques globales")
    for name, value in metrics["global"].items():
        print(f"  {name:24} {value}")

    print("\n=== Par catégorie")
    print(f"  {'catégorie':14} {'n':>3} {'comportement':>13} {'faits clés':>11} {'complets':>9} {'citée':>7}")
    for category, values in metrics["par_categorie"].items():
        facts = "-" if values["faits_cles_moyenne"] is None else f"{values['faits_cles_moyenne']:.2f}"
        print(f"  {category:14} {values['questions']:>3} {values['comportement_correct']:>13} {facts:>11} "
              f"{values['faits_cles_complets']:>9} {values['reponses_avec_citation']:>7}")

    print("\n  (faits clés : correspondance de texte, une paraphrase n'est pas reconnue ; à relire dans le fichier)")
    print("\n  Ratés à relire :")
    for row in metrics.get("_rows", []):
        scores = row["scores"]
        if not scores["behavior_ok"] or scores.get("key_facts") not in (None, 1):
            print(f"    {row['id']:28} comportement={'ok' if scores['behavior_ok'] else 'FAUX'}  "
                  f"faits={scores.get('key_facts')}")


def evaluate(label, split=None, categories=None, answers_path=None):
    items = load_dataset(split=split, categories=categories)
    if answers_path:
        # Rescore an existing run: no LLM call
        previous = json.loads(Path(answers_path).read_text(encoding="utf-8"))
        rows, context = previous["rows"], previous["context"]
        items_by_id = {item.id: item for item in load_dataset()}
        label = previous["label"] + "_rescore"
    else:
        print(f"{len(items)} questions (split={split or 'tous'}, catégories={categories or 'toutes'})")
        rows = generate_answers(items)
        items_by_id = {item.id: item for item in items}
        context = {
            "dataset": DEFAULT_DATASET.name,
            "split": split or "tous",
            "categories": categories,
            "git_commit": git_commit(),
        }

    metrics = score_run(rows, items_by_id)
    print_metrics({**metrics, "_rows": rows})

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"generation_{label}_{datetime.now():%Y%m%d_%H%M%S}.json"
    run = {"label": label, "date": datetime.now().isoformat(timespec="seconds"), "context": context,
           "metrics": metrics, "rows": rows}
    path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats enregistrés dans {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run", help="Name of the run (e.g. baseline, prompt-strict)")
    parser.add_argument("--split", choices=["dev", "test"], help="dev to tune, test only at the end (default: all)")
    parser.add_argument("--categories", nargs="+", help="Only these categories (default: all)")
    parser.add_argument("--answers", help="Rescore an existing generation_*.json without calling the LLM")
    args = parser.parse_args()
    evaluate(args.label, split=args.split, categories=args.categories, answers_path=args.answers)
