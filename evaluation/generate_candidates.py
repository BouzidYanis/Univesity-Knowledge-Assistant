"""
Synthetic generation of CANDIDATE questions from random chunks, for a human review.

The candidates are written to evaluation/dataset/candidates.jsonl, never directly to the dataset:
each one must be read, then kept (with corrections) or rejected, and marked origin="synthetique".

Run from the project root (1 LLM call per chunk):
    python -m evaluation.generate_candidates --file STATUTS --n 8 --seed 42
"""
import argparse
import json
import random

from evaluation.dataset import DATASET_DIR
from src.embedding import client
from src.llm import generate

COLLECTION_NAME = "knowledge_base"
CANDIDATES_PATH = DATASET_DIR / "candidates.jsonl"
MIN_CHUNK_CHARS = 400  # short chunks (titles, signatures) make trivial questions

GENERATION_PROMPT = """Tu aides à construire un jeu d'évaluation pour un assistant qui répond aux questions des étudiants sur les documents officiels de leur université.

On te donne un extrait d'un document. Écris UNE question qu'un étudiant ou un membre du personnel pourrait réellement poser, et dont la réponse se trouve dans cet extrait.

Règles :
1. Formule la question avec tes propres mots, comme un étudiant qui n'a pas lu le document : ne reprends pas les expressions de l'extrait.
2. La question doit avoir du sens sans l'extrait sous les yeux (pas de « que dit ce paragraphe ? »).
3. La réponse doit être entièrement contenue dans l'extrait, sans interprétation.
4. Donne 1 à 3 faits clés courts et vérifiables (un nombre, un nom, une règle), copiés de l'extrait.

Réponds uniquement avec un objet JSON sur une ligne :
{"question": "...", "reference_answer": "...", "key_facts": ["...", "..."]}"""


def load_chunks(file_filter):
    """All the chunks whose file name contains file_filter, long enough to make a real question."""
    chunks, offset = [], None
    while True:
        points, offset = client.scroll(COLLECTION_NAME, limit=256, offset=offset, with_payload=True)
        chunks += [p for p in points
                   if file_filter.lower() in p.payload["source_file"].lower() and len(p.payload["text"]) >= MIN_CHUNK_CHARS]
        if offset is None:
            break
    return sorted(chunks, key=lambda p: p.id)


def generate_candidate(chunk):
    """Ask the LLM for a question about one chunk. Returns the parsed dict, or None if the output is not valid JSON."""
    messages = [{"role": "system", "content": GENERATION_PROMPT},
                {"role": "user", "content": f"Extrait :\n{chunk.payload['text']}"}]
    output = generate(messages)
    start, end = output.find("{"), output.rfind("}")
    try:
        parsed = json.loads(output[start:end + 1])
    except ValueError:
        print(f"  sortie invalide pour le chunk {chunk.id} : {output[:100]!r}")
        return None
    return parsed if isinstance(parsed, dict) and parsed.get("question") else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Part of the file name, e.g. STATUTS, Guide")
    parser.add_argument("--n", type=int, default=8, help="Number of chunks to sample")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed: the same command gives the same chunks")
    args = parser.parse_args()

    chunks = load_chunks(args.file)
    sample = random.Random(args.seed).sample(chunks, min(args.n, len(chunks)))
    print(f"{len(chunks)} chunks éligibles, {len(sample)} tirés au hasard (seed={args.seed})")

    with CANDIDATES_PATH.open("a", encoding="utf-8") as f:
        for chunk in sample:
            candidate = generate_candidate(chunk)
            if candidate is None:
                continue
            record = {
                **candidate,
                # Where the question comes from: to check the answer by hand
                "source_file": chunk.payload["source_file"],
                "page": chunk.payload["page"] + 1,  # numbered from 1, like the dataset
                "chunk_id": chunk.id,
                "chunk_text": chunk.payload["text"],
                "status": "a_relire",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  [p.{record['page']}] {candidate['question']}")

    print(f"\nCandidats ajoutés à {CANDIDATES_PATH} : à relire avant tout ajout au dataset.")


if __name__ == "__main__":
    main()
