"""
Evaluation dataset: schema of an entry (Pydantic) and loading with validation.

Every evaluation script loads the questions with load_dataset(): one source of truth, one format.
"""
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DATASET_DIR = Path(__file__).parent / "dataset"
DEFAULT_DATASET = DATASET_DIR / "eval_dataset_v1.jsonl"

Category = Literal[
    "factuelle",      # answer in a single passage
    "mots_cles",      # article numbers, figures, acronyms: what BM25 must find
    "reformulee",     # without the words of the document: what the dense search must find
    "multi_sources",  # answer spread over several passages or documents
    "partielle",      # one part is in the documents, the other is not
    "non_couverte",   # about the university, but not in the indexed documents
    "hors_domaine",   # nothing to do with the university
    "conversation",   # greetings, thanks, questions about the assistant
    "suivi",          # follow-up question: needs the history
    "detournement",   # attempts to bypass the rules
]


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ExpectedSource(BaseModel):
    """A passage that answers the question: file name + pages numbered FROM 1 (as in a PDF reader).

    File + page and not chunk ids: chunk ids change at every re-indexing, PDF pages do not.
    """
    file: str
    pages: list[int] = Field(min_length=1)


class EvalItem(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    question: str = Field(min_length=1)
    history: list[HistoryMessage] = Field(default_factory=list)
    category: Category
    expected_behavior: Literal["reponse", "reponse_partielle", "refus", "conversation"]
    expected_type: Literal["documentation", "conversation", "hors_domaine"]
    expected_sources: list[ExpectedSource] = Field(default_factory=list)
    reference_answer: str = ""
    # Short facts the answer must contain. Alternatives are separated by "|": "4 ans|quatre ans"
    key_facts: list[str] = Field(default_factory=list)
    difficulty: Literal["facile", "moyenne", "difficile"] = "moyenne"
    origin: Literal["manuelle", "synthetique"] = "manuelle"
    split: Literal["dev", "test"] | None = None
    notes: str = ""


def load_dataset(path=DEFAULT_DATASET, split=None, categories=None):
    """
    Load and validate the evaluation dataset (one JSON object per line).

    Args:
        path (str | Path): The JSONL file.
        split (str | None): Keep only "dev" or "test" items (None: all).
        categories (iterable | None): Keep only these categories (None: all).

    Returns:
        list[EvalItem]: The validated items, in file order.

    Raises:
        ValueError: A line is not valid (the line number is given).
    """
    items = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(EvalItem.model_validate_json(line))
        except ValueError as e:
            raise ValueError(f"{path}, ligne {line_number} : {e}") from e

    if split is not None:
        items = [item for item in items if item.split == split]
    if categories is not None:
        items = [item for item in items if item.category in set(categories)]
    return items


def save_dataset(items, path=DEFAULT_DATASET):
    """Write the items as JSONL (one object per line, accents kept)."""
    lines = [item.model_dump_json(exclude_defaults=False) for item in items]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
