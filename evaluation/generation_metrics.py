"""
Automatic metrics of the generated answers (no LLM judge): key facts and citations.

The behaviour (refusal / answer / conversation) is measured with is_refusal of out_of_scope_questions.
"""
import re
import unicodedata

# "[1]", "[2, 3]", "[1; 4]"
CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def normalize(text):
    """Lowercase, no accents, typographic apostrophes, hyphens and non-breaking spaces replaced, single spaces."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("’", "'").replace("‘", "'").replace("*", "")
    # Hyphens: the LLM often writes "vice‑président" with a non-breaking hyphen (U+2011, U+2010 after NFKD)
    text = text.replace("\u2010", "-").replace("\u2011", "-")
    return " ".join(text.split())  # split() also splits on non-breaking spaces (\xa0,  )


def cited_numbers(answer):
    """The excerpt numbers cited in the answer, in order of first appearance: "[1] ... [2, 1]" -> [1, 2]."""
    numbers = []
    for group in CITATION_PATTERN.findall(answer):
        for number in re.split(r"\s*[,;]\s*", group):
            if int(number) not in numbers:
                numbers.append(int(number))
    return numbers


def contains_fact(answer, fact):
    """
    True if the answer contains one of the alternatives of the fact ("4 ans|quatre ans").

    The citations are removed first ("[2]" must not count as the fact "2"). An alternative must start at a
    word boundary; it must also end at one only if it ends with a digit: "2" must not match "20",
    but "interdit" must match "interdite".
    """
    text = normalize(CITATION_PATTERN.sub(" ", answer))
    for alternative in fact.split("|"):
        alternative = normalize(alternative)
        if not alternative:
            continue
        pattern = r"(?<!\w)" + re.escape(alternative) + (r"(?!\d)" if alternative[-1].isdigit() else "")
        if re.search(pattern, text):
            return True
    return False


def key_fact_coverage(answer, key_facts):
    """Share of the key facts found in the answer (None if the question has no key fact)."""
    if not key_facts:
        return None
    return sum(contains_fact(answer, fact) for fact in key_facts) / len(key_facts)


def citation_metrics(answer, sources, relevant_pages):
    """
    Citations of an answer that uses excerpts.

    Args:
        answer (str): The generated answer.
        sources (list[dict]): The excerpts given to the LLM ({"number", "source_file", "page", ...}).
        relevant_pages (set[tuple[str, int]]): The expected (file, page) pairs.

    Returns:
        dict: has_citation (bool), invalid (number of cited numbers without excerpt),
            on_expected_page (share of the valid citations pointing to an expected page, None if no valid citation).
    """
    by_number = {source["number"]: (source["source_file"], source["page"]) for source in sources}
    cited = cited_numbers(answer)
    valid = [number for number in cited if number in by_number]
    return {
        "has_citation": bool(cited),
        "invalid": len(cited) - len(valid),
        "on_expected_page": sum(by_number[number] in relevant_pages for number in valid) / len(valid) if valid else None,
    }
