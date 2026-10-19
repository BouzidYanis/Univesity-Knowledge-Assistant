"""
Assign a dev / test split to the questions that don't have one yet, stratified by category.

- dev  (~70 %): used to tune the system (prompts, top_k, candidates, thresholds...)
- test (~30 %): only used at the end, for the reported score

Idempotent: questions that already have a split are never moved (a frozen test set stays frozen).

Run from the project root:
    python -m evaluation.assign_splits
"""
import random
from collections import defaultdict

from evaluation.dataset import load_dataset, save_dataset

TEST_SHARE = 0.30
SEED = 42


def assign_splits(items, test_share=TEST_SHARE, seed=SEED):
    """Set item.split for the items without one, so that each category has ~test_share of test items."""
    rng = random.Random(seed)
    by_category = defaultdict(list)
    for item in items:
        by_category[item.category].append(item)

    for category in sorted(by_category):
        group = by_category[category]
        target_test = round(len(group) * test_share)
        if len(group) >= 2:
            target_test = max(1, target_test)  # every category is represented in the test set

        missing_test = target_test - sum(item.split == "test" for item in group)
        unassigned = sorted((item for item in group if item.split is None), key=lambda item: item.id)
        rng.shuffle(unassigned)
        for index, item in enumerate(unassigned):
            item.split = "test" if index < missing_test else "dev"
    return items


if __name__ == "__main__":
    items = assign_splits(load_dataset())
    save_dataset(items)
    by_category = defaultdict(lambda: {"dev": 0, "test": 0})
    for item in items:
        by_category[item.category][item.split] += 1
    print(f"{'catégorie':14} {'dev':>4} {'test':>5}")
    for category, counts in sorted(by_category.items()):
        print(f"{category:14} {counts['dev']:>4} {counts['test']:>5}")
    print(f"{'total':14} {sum(c['dev'] for c in by_category.values()):>4} {sum(c['test'] for c in by_category.values()):>5}")
