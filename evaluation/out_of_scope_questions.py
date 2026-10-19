"""
Behaviour of the assistant on the dataset: refusals, false refusals, conversation messages, router.

Run from the project root (calls the LLM once or twice per question: start with a split or a few categories):
    python -m evaluation.out_of_scope_questions --label avant --split dev
    python -m evaluation.out_of_scope_questions --label apres --categories hors_domaine non_couverte conversation

The answers are saved in evaluation/results/ for a manual review (hallucinations, partial answers).
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from evaluation.dataset import load_dataset
from src.rag import OUT_OF_SCOPE_ANSWER, answer, format_history, is_not_found

RESULTS_DIR = Path(__file__).parent / "results"


def is_refusal(result):
    """A refusal is either the NOT_FOUND answer or the out-of-domain message."""
    return is_not_found(result["answer"]) or result["answer"] == OUT_OF_SCOPE_ANSWER


def evaluate(label, split=None, categories=None):
    """Run answer() on every selected question, print the metrics and save the answers for a manual review."""
    items = load_dataset(split=split, categories=categories)
    rows = []
    for item in items:
        history = format_history([message.model_dump() for message in item.history])
        start = time.perf_counter()
        result = answer(item.question, history=history)
        rows.append({
            "id": item.id,
            "category": item.category,
            "question": item.question,
            "expected_behavior": item.expected_behavior,
            "expected_type": item.expected_type,
            "type": result.get("type"),
            "rewritten_question": result.get("rewritten_question"),
            "refusal": is_refusal(result),
            "answer": result["answer"],
            "reference_answer": item.reference_answer,
            "sources": [f"{s['source_file']} p.{s['page']}" for s in result["sources"]],
            "seconds": round(time.perf_counter() - start, 1),
        })
        print(f"[{item.category:14}] {'REFUS ' if rows[-1]['refusal'] else 'répond'}  "
              f"type={rows[-1]['type']:13} {item.id}")

    def ratio(selected, condition):
        return f"{sum(condition(r) for r in selected)}/{len(selected)}" if selected else "n/a"

    should_refuse = [r for r in rows if r["expected_behavior"] == "refus"]
    should_answer = [r for r in rows if r["expected_behavior"] in ("reponse", "reponse_partielle")]
    conversations = [r for r in rows if r["expected_behavior"] == "conversation"]
    metrics = {
        "dataset": f"v1, split={split or 'tous'}, {len(rows)} questions",
        "refus_corrects": ratio(should_refuse, lambda r: r["refusal"]),
        "faux_refus": ratio(should_answer, lambda r: r["refusal"]),
        # A conversation message must get a real answer, without sources
        "conversation_ok": ratio(conversations, lambda r: not r["refusal"] and not r["sources"]),
        "routeur_correct": ratio(rows, lambda r: r["type"] == r["expected_type"]),
        "latence_moyenne_s": round(sum(r["seconds"] for r in rows) / len(rows), 1) if rows else None,
    }
    print("\n=== Métriques")
    for name, value in metrics.items():
        print(f"  {name:20} {value}")
    print("  (hallucinations et réponses partielles : à relire dans le fichier de résultats)")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"out_of_scope_{label}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({"label": label, "metrics": metrics, "rows": rows}, ensure_ascii=False, indent=2))
    print(f"\nRésultats enregistrés dans {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run", help="Name of the run (e.g. avant, apres)")
    parser.add_argument("--split", choices=["dev", "test"], help="Only the dev or test questions (default: all)")
    parser.add_argument("--categories", nargs="+", help="Only these categories (default: all)")
    args = parser.parse_args()
    evaluate(args.label, split=args.split, categories=args.categories)
