"""
Tests of the API, without Qdrant, models or LLM: fast, free and deterministic.

Run from the project root:
    python -m pytest
"""
import pytest
from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import ResponseHandlingException

import api.main
from src.rag import NOT_FOUND_ANSWER, OUT_OF_SCOPE_ANSWER

# No "with TestClient(app)": without the context manager, the lifespan (model loading, Qdrant check) is not run
client = TestClient(api.main.app)

SOURCE = {"number": 1, "source_file": "Charte_des_Associations.pdf", "page": 1, "score": 0.8, "rerank_score": 1.2}


@pytest.fixture
def fake_answer(monkeypatch):
    """Replace answer() where it is USED (api.main), and record the calls."""
    calls = []
    response = {"answer": "Un étudiant inscrit [1].", "sources": [SOURCE],
                "rewritten_question": "question réécrite", "type": "documentation"}

    def _answer(question, **kwargs):
        calls.append({"question": question, **kwargs})
        return response

    monkeypatch.setattr(api.main, "answer", _answer)
    return calls, response


def test_ask_returns_the_answer(fake_answer):
    response = client.post("/ask", json={"question": "Qui peut présider une association étudiante ?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Un étudiant inscrit [1]."
    assert body["type"] == "documentation"
    assert body["refused"] is False
    assert body["rewritten_question"] == "question réécrite"
    assert body["sources"] == [SOURCE]


def test_ask_passes_the_options(fake_answer):
    calls, _ = fake_answer
    client.post("/ask", json={"question": "Q ?", "top_k": 3, "mode": "bm25", "use_reranker": False})

    assert calls[0]["top_k"] == 3
    assert calls[0]["mode"] == "bm25"
    assert calls[0]["use_reranker"] is False


def test_ask_cleans_the_history_with_format_history(fake_answer):
    calls, _ = fake_answer
    history = [
        {"role": "user", "content": "Qui peut présider une association ?"},
        {"role": "assistant", "content": "Un étudiant inscrit [1]."},
        {"role": "user", "content": "Météo ?"},
        {"role": "assistant", "content": NOT_FOUND_ANSWER},
    ]
    client.post("/ask", json={"question": "Et le bureau ?", "history": history})

    # Citations removed, "I don't know" answer skipped
    assert calls[0]["history"] == [
        {"role": "user", "content": "Qui peut présider une association ?"},
        {"role": "assistant", "content": "Un étudiant inscrit."},
        {"role": "user", "content": "Météo ?"},
    ]


@pytest.mark.parametrize("answer_text", [NOT_FOUND_ANSWER, OUT_OF_SCOPE_ANSWER])
def test_ask_marks_refusals(fake_answer, answer_text):
    _, response = fake_answer
    response.update(answer=answer_text, sources=[])

    assert client.post("/ask", json={"question": "Recette des crêpes ?"}).json()["refused"] is True


@pytest.mark.parametrize("payload", [
    {"question": ""},                                            # empty question
    {"question": "x" * 1001},                                    # too long
    {"question": "Q ?", "mode": "foo"},                          # unknown mode
    {"question": "Q ?", "top_k": 0},                             # top_k out of bounds
    {"question": "Q ?", "history": [{"role": "system", "content": "..."}]},  # forbidden role
    {},                                                          # no question
])
def test_ask_rejects_invalid_requests(fake_answer, payload):
    calls, _ = fake_answer
    response = client.post("/ask", json=payload)

    assert response.status_code == 422
    assert calls == []  # answer() is never called with invalid data


def test_ask_returns_503_when_a_dependency_is_down(monkeypatch):
    def _answer(question, **kwargs):
        raise ResponseHandlingException("Connection refused to http://localhost:6333")

    monkeypatch.setattr(api.main, "answer", _answer)
    response = client.post("/ask", json={"question": "Q ?"})

    assert response.status_code == 503
    assert "localhost" not in response.text  # the error details are not sent to the client


def test_ask_returns_500_on_unexpected_error(monkeypatch):
    def _answer(question, **kwargs):
        raise KeyError("bug")

    monkeypatch.setattr(api.main, "answer", _answer)
    assert client.post("/ask", json={"question": "Q ?"}).status_code == 500


class FakeQdrant:
    def __init__(self, count=None, error=None):
        self._count, self._error = count, error

    def count(self, collection_name):
        if self._error:
            raise self._error
        return type("CountResult", (), {"count": self._count})()


def test_health_ok(monkeypatch):
    monkeypatch.setattr(api.main, "client", FakeQdrant(count=235))
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "qdrant": "ok", "points": 235}


def test_health_returns_503_when_qdrant_is_down(monkeypatch):
    monkeypatch.setattr(api.main, "client", FakeQdrant(error=ConnectionError("refused")))
    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "error"


def test_health_returns_503_when_the_collection_is_empty(monkeypatch):
    monkeypatch.setattr(api.main, "client", FakeQdrant(count=0))
    assert client.get("/health").status_code == 503
