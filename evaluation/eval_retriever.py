"""
Evaluation of the retriever alone (no LLM): are the right pages retrieved?

Unit of relevance: the (file, page) pair, pages numbered from 1 like in the dataset.

Run from the project root:
    python -m evaluation.eval_retriever --split dev
    python -m evaluation.eval_retriever --split dev --configs hybrid hybrid+rerank20 --label reranker
    python -m evaluation.eval_retriever --split dev --refresh-rewrites   # call the LLM again for the follow-ups
"""
import argparse
import json
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from evaluation.dataset import DEFAULT_DATASET, load_dataset
from evaluation.metrics import hit_at_k, reciprocal_rank
from src.embedding import EMBEDDING_MODEL_NAME, search_in_qdrant
from src.rag import format_history, route_question
from src.reranker import RERANKER_MODEL_NAME, rerank

# Always retrieve at least MAX_K pages, so that every metric @k (k <= MAX_K) is computed on the same list
MAX_K = 10
KS = [1, 3, 5, 10]

# Palier 3: the configurations to compare, by name
CONFIGS = {
    "dense": {"mode": "dense"},
    "bm25": {"mode": "bm25"},
    "hybrid": {"mode": "hybrid"},
    "hybrid+rerank20": {"mode": "hybrid", "use_reranker": True, "candidates": 20},
    "hybrid+rerank40": {"mode": "hybrid", "use_reranker": True, "candidates": 40},
}

RESULTS_DIR = Path(__file__).parent / "results"
# id -> rewritten question of the follow-ups (the LLM is not deterministic: rewrite once, then reuse)
REWRITES_CACHE = RESULTS_DIR / "rewrites_cache.json"


def load_eval_items(split="dev"):
    """
    The questions that have pages to retrieve (expected behaviour "reponse" or "reponse_partielle").

    Args:
        split (str | None): "dev" (to tune), "test" (only at the end) or None (all).

    Returns:
        list[EvalItem]: The questions with expected_sources.
    """
    return [item for item in load_dataset(split=split) if item.expected_sources]


def relevant_pages(item):
    """The expected (file, page) pairs of a question."""
    return {(source.file, page) for source in item.expected_sources for page in source.pages}


def retrieve(question, config, k=MAX_K):
    """
    Ranked list of the (file, page) pairs retrieved for a question, without duplicates.

    Args:
        question (str): The question (already standalone for follow-up questions).
        config (dict): {"mode": "hybrid" | "dense" | "bm25", "use_reranker": bool, "candidates": int}
        k (int): The number of results asked to the retriever.

    Returns:
        list[tuple[str, int]]: (file, page) pairs, best first, each page only once.
    """
    if config.get("use_reranker"):
        # The reranker cuts the list to top_k: ask it k results, not 5
        candidates = search_in_qdrant(question, top_k=max(config.get("candidates", 20), k), mode=config["mode"])
        results = rerank(question, candidates, top_k=k)
    else:
        results = search_in_qdrant(question, top_k=k, mode=config["mode"])

    pages = []
    for result in results:
        # PyMuPDF counts pages from 0, the dataset from 1
        page = (result["metadata"]["source_file"], result["metadata"]["page"] + 1)
        # Several chunks of the same page count only once (otherwise the metrics are inflated)
        if page not in pages:
            pages.append(page)
    return pages


# ---------- Palier 4: follow-up questions ----------

def load_rewrites(items, refresh=False):
    """
    Rewritten (standalone) question of every follow-up item, read from the cache or asked to the router.

    The cache stores the original question too: if the question of the dataset changes, it is rewritten again.

    Args:
        items (list[EvalItem]): The evaluated items (only those with a history are rewritten).
        refresh (bool): Ignore the cache and call the LLM again.

    Returns:
        dict: id -> rewritten question.
    """
    cache = {}
    if REWRITES_CACHE.exists() and not refresh:
        cache = json.loads(REWRITES_CACHE.read_text(encoding="utf-8"))

    updated = False
    for item in items:
        if not item.history:
            continue
        cached = cache.get(item.id)
        if cached and cached["question"] == item.question:
            continue
        history = format_history([message.model_dump() for message in item.history])
        rewritten = route_question(item.question, history)["question"]
        # route_question returns the raw question when the LLM fails: it is cached too, check the file
        cache[item.id] = {"question": item.question, "rewritten": rewritten}
        updated = True
        print(f"  réécrite : {item.id} -> {rewritten}")

    if updated:
        RESULTS_DIR.mkdir(exist_ok=True)
        REWRITES_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return {item_id: entry["rewritten"] for item_id, entry in cache.items()}


# ---------- Palier 5: run and aggregate ----------

def question_metrics(retrieved, relevant):
    """The metrics of one question: Hit@k for every k of KS, and the reciprocal rank."""
    metrics = {f"hit@{k}": hit_at_k(retrieved, relevant, k) for k in KS}
    metrics["mrr"] = reciprocal_rank(retrieved, relevant)
    return metrics


def mean_metrics(rows):
    """Mean of every metric over a list of rows, plus the number of questions."""
    names = [f"hit@{k}" for k in KS] + ["mrr"]
    means = {name: round(sum(row["metrics"][name] for row in rows) / len(rows), 3) for name in names}
    means["n"] = len(rows)
    return means


def group_by(rows, key):
    """Mean metrics per value of key(row); key may return several values (a question with 2 documents)."""
    groups = defaultdict(list)
    for row in rows:
        for value in key(row):
            groups[value].append(row)
    return {value: mean_metrics(group) for value, group in sorted(groups.items())}


def run_config(name, config, items, rewrites):
    """
    Retrieve the pages of every question with one configuration.

    Returns:
        list[dict]: One row per question (the follow-ups get a second row with the raw question).
    """
    # Warm-up: the first call loads the models, it must not be counted in the latency
    retrieve("warm-up", config)

    rows = []
    for item in items:
        relevant = relevant_pages(item)
        # The follow-ups are searched with their rewritten question (like in answer()) AND with the raw one
        variants = [("reecrite", rewrites.get(item.id, item.question)), ("brute", item.question)] if item.history \
            else [("directe", item.question)]
        for variant, query in variants:
            start = time.perf_counter()
            retrieved = retrieve(query, config)
            seconds = time.perf_counter() - start
            rows.append({
                "id": item.id,
                "variant": variant,
                "category": item.category,
                "difficulty": item.difficulty,
                "documents": sorted({source.file for source in item.expected_sources}),
                "query": query,
                "expected": sorted([list(page) for page in relevant]),
                "retrieved": [list(page) for page in retrieved],
                "metrics": question_metrics(retrieved, relevant),
                "seconds": round(seconds, 3),
            })
    print(f"  {name:16} {len(rows)} recherches")
    return rows


def aggregate(rows):
    """Global, per category, per document and per difficulty metrics, plus the follow-up comparison and latency."""
    # The raw follow-ups are only an experiment: the main metrics use what answer() really searches
    main = [row for row in rows if row["variant"] != "brute"]
    followups = [row for row in rows if row["variant"] != "directe"]
    return {
        "global": mean_metrics(main),
        "par_categorie": group_by(main, lambda row: [row["category"]]),
        "par_document": group_by(main, lambda row: row["documents"]),
        "par_difficulte": group_by(main, lambda row: [row["difficulty"]]),
        "suivi_brute_vs_reecrite": group_by(followups, lambda row: [row["variant"]]) if followups else {},
        "latence_moyenne_s": round(sum(row["seconds"] for row in main) / len(main), 3),
    }


# ---------- Palier 6: save and print ----------

def git_commit():
    """Hash of the current commit (+ "-dirty" if there are uncommitted changes), or None outside git."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def markdown_table(metrics_by_config):
    """Summary table, ready to paste in the README."""
    header = "| Configuration | " + " | ".join(f"Hit@{k}" for k in KS) + " | MRR | Latence (s) |"
    lines = [header, "|" + "---|" * (len(KS) + 3)]
    for name, metrics in metrics_by_config.items():
        values = [f"{metrics['global'][f'hit@{k}']:.2f}" for k in KS]
        lines.append(f"| {name} | " + " | ".join(values)
                     + f" | {metrics['global']['mrr']:.2f} | {metrics['latence_moyenne_s']:.2f} |")
    return "\n".join(lines)


def print_breakdown(metrics_by_config, section, metric="hit@5"):
    """One metric per configuration (columns) and per group of a section (rows)."""
    names = list(metrics_by_config)
    groups = sorted({group for metrics in metrics_by_config.values() for group in metrics[section]})
    if not groups:
        return
    print(f"\n=== {metric} {section.replace('_', ' ')}")
    print(f"  {'':38}" + "".join(f"{name[:16]:>17}" for name in names))
    for group in groups:
        n = metrics_by_config[names[0]][section][group]["n"]
        values = "".join(f"{metrics_by_config[name][section][group][metric]:>17.2f}" for name in names)
        print(f"  {group[:32]:32} n={n:<3}{values}")


def evaluate(label, split="dev", config_names=None, refresh_rewrites=False):
    """Run every configuration on the selected split, print the tables and save the run."""
    config_names = config_names or list(CONFIGS)
    items = load_eval_items(split)
    print(f"{len(items)} questions avec sources (split={split or 'tous'})")

    rewrites = load_rewrites(items, refresh=refresh_rewrites)

    rows_by_config, metrics_by_config = {}, {}
    for name in config_names:
        rows_by_config[name] = run_config(name, CONFIGS[name], items, rewrites)
        metrics_by_config[name] = aggregate(rows_by_config[name])

    print("\n=== Résumé (questions de suivi : version réécrite)\n")
    print(markdown_table(metrics_by_config))
    for section in ["par_categorie", "par_difficulte", "par_document", "suivi_brute_vs_reecrite"]:
        print_breakdown(metrics_by_config, section)

    run = {
        "label": label,
        "date": datetime.now().isoformat(timespec="seconds"),
        "context": {
            "dataset": DEFAULT_DATASET.name,
            "split": split or "tous",
            "questions": len(items),
            "git_commit": git_commit(),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "reranker_model": RERANKER_MODEL_NAME,
            "max_k": MAX_K,
            "configs": {name: CONFIGS[name] for name in config_names},
        },
        "metrics": metrics_by_config,
        "rows": rows_by_config,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"retriever_{label}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats enregistrés dans {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run", help="Name of the run (e.g. baseline, sans-sommaire)")
    parser.add_argument("--split", choices=["dev", "test"], default="dev", help="dev to tune, test only at the end")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), help="Only these configurations (default: all)")
    parser.add_argument("--refresh-rewrites", action="store_true", help="Rewrite the follow-ups again with the LLM")
    args = parser.parse_args()
    evaluate(args.label, split=args.split, config_names=args.configs, refresh_rewrites=args.refresh_rewrites)
