"""
Error analysis of a retriever run: show the questions a configuration misses, to classify the causes by hand.

Reads the JSON of a run (no new search for the pages), then shows the text of the first passages
(one search per missed question) so that you can see WHY a wrong page came first.

Run from the project root:
    python -m evaluation.error_analysis                                   # last run, hybrid+rerank20, hit@1
    python -m evaluation.error_analysis --config hybrid --metric hit@5
    python -m evaluation.error_analysis --run evaluation/results/retriever_baseline_20261019_122033.json
"""
import argparse
import json
from pathlib import Path

from evaluation.eval_retriever import CONFIGS, RESULTS_DIR
from src.embedding import search_in_qdrant
from src.reranker import rerank

TOP_SHOWN = 5
TEXTS_SHOWN = 3
TEXT_CHARS = 200


def last_run():
    """The most recent retriever run of evaluation/results/."""
    runs = sorted(RESULTS_DIR.glob("retriever_*.json"))
    if not runs:
        raise FileNotFoundError("Aucun run : lance d'abord python -m evaluation.eval_retriever")
    return runs[-1]


def missed_rows(run, config, metric):
    """The rows of a configuration with metric == 0 (the raw follow-ups are only an experiment: skipped)."""
    return [row for row in run["rows"][config] if row["variant"] != "brute" and row["metrics"][metric] == 0]


def first_relevant_rank(row):
    """Rank (from 1) of the first expected page in the retrieved list, or None if it is not in the top MAX_K."""
    expected = {tuple(page) for page in row["expected"]}
    for rank, page in enumerate(row["retrieved"], start=1):
        if tuple(page) in expected:
            return rank
    return None


def first_passages(query, config):
    """The first chunks (with their text) that the configuration retrieves, to read them."""
    if config.get("use_reranker"):
        candidates = search_in_qdrant(query, top_k=config.get("candidates", 20), mode=config["mode"])
        return rerank(query, candidates, top_k=TEXTS_SHOWN)
    return search_in_qdrant(query, top_k=TEXTS_SHOWN, mode=config["mode"])


def show(row, config):
    expected = {tuple(page) for page in row["expected"]}
    rank = first_relevant_rank(row)
    print(f"\n{'=' * 100}\n{row['id']}  [{row['category']}, {row['difficulty']}]  "
          f"première bonne page : {'rang ' + str(rank) if rank else 'absente du top 10'}")
    print(f"  question cherchée : {row['query']}")
    print(f"  attendu  : {', '.join(f'{file[:40]} p.{page}' for file, page in row['expected'])}")
    print(f"  renvoyé  :")
    for position, (file, page) in enumerate(row["retrieved"][:TOP_SHOWN], start=1):
        mark = "*" if (file, page) in expected else " "
        print(f"    {mark} {position}. {file[:40]} p.{page}")
    print("  passages lus en premier :")
    for result in first_passages(row["query"], config):
        page = (result["metadata"]["source_file"], result["metadata"]["page"] + 1)
        mark = "*" if page in expected else " "
        text = " ".join(result["text"].split())[:TEXT_CHARS]
        print(f"    {mark} {page[0][:25]} p.{page[1]} : {text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, help="JSON file of the run (default: the last one)")
    parser.add_argument("--config", default="hybrid+rerank20", help="Configuration to analyse")
    parser.add_argument("--metric", default="hit@1", help="A question is missed when this metric is 0")
    args = parser.parse_args()

    path = args.run or last_run()
    run = json.loads(path.read_text(encoding="utf-8"))
    missed = missed_rows(run, args.config, args.metric)
    print(f"Run : {path.name}  |  {args.config}  |  {len(missed)} questions avec {args.metric} = 0")
    for row in missed:
        show(row, CONFIGS[args.config])
