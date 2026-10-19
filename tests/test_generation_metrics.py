"""
Tests of the automatic metrics of the generated answers.

Run from the project root:
    python -m pytest tests/test_generation_metrics.py
"""
import pytest

from evaluation.generation_metrics import citation_metrics, cited_numbers, contains_fact, key_fact_coverage


@pytest.mark.parametrize("answer, fact, expected", [
    ("Le mandat est de quatre ans [1].", "4 ans|quatre ans", True),       # second alternative
    ("Le mandat est de 4 ans.", "4 ans|quatre ans", True),
    ("Il est élu pour un mandat.", "4 ans|quatre ans", False),
    ("Il faut un ETUDIANT INSCRIT.", "étudiant inscrit", True),            # case
    ("Un etudiant inscrit à l'université", "étudiant inscrit", True),      # accents
    ("Il est strictement personnel.", "strictement personnels", False),    # singular: not the same text
    ("La consommation est interdite.", "interdit", True),                  # word continued on the right
    ("C'est réinterdit.", "interdit", False),                              # but must start a word
    ("Au moins 20 % de produits bio.", "20 %|20%", True),
    ("Au moins 20 % de produits bio.", "20 %|20%", True),             # narrow non-breaking space
    ("Une amende de 7\xa0500 euros.", "7500|7 500", True),                 # non-breaking space
    ("Voir l'article 20.", "deux|2", False),                               # "2" must not match "20"
    ("Selon l'extrait [2], il faut sortir.", "deux|2", False),             # nor a citation
    ("Il faut deux sorties [1].", "deux|2", True),
    ("La procédure de l’université", "l'université", True),                # typographic apostrophe
    ("Le (la) Vice\u2011Président(e) étudiant(e) [1].", "vice-président(e) étudiant(e)", True),  # non-breaking hyphen
])
def test_contains_fact(answer, fact, expected):
    assert contains_fact(answer, fact) is expected


def test_key_fact_coverage():
    answer = "Le président est élu pour 4 ans [1]."
    assert key_fact_coverage(answer, ["4 ans|quatre ans", "renouvelable une fois"]) == 0.5
    assert key_fact_coverage(answer, []) is None


def test_cited_numbers():
    assert cited_numbers("A [1]. B [2, 3]. C [1; 4] et [2].") == [1, 2, 3, 4]
    assert cited_numbers("Aucune citation.") == []


SOURCES = [
    {"number": 1, "source_file": "RI.pdf", "page": 12},
    {"number": 2, "source_file": "RI.pdf", "page": 13},
    {"number": 3, "source_file": "Statuts.pdf", "page": 4},
]


def test_citation_metrics():
    metrics = citation_metrics("A [1]. B [3]. C [6].", SOURCES, {("RI.pdf", 12)})
    assert metrics == {"has_citation": True, "invalid": 1, "on_expected_page": 0.5}


def test_citation_metrics_without_citation():
    metrics = citation_metrics("Pas de citation.", SOURCES, {("RI.pdf", 12)})
    assert metrics == {"has_citation": False, "invalid": 0, "on_expected_page": None}
