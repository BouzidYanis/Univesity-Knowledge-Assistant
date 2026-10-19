"""
Pydantic schemas: the "contract" of the API (validation of the requests, documentation in /docs).
"""
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """One message of the conversation history."""
    role: Literal["user", "assistant"]
    content: str = Field(max_length=5000)


class AskRequest(BaseModel):
    """A question, with the previous messages of the conversation (the API is stateless)."""
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="La question de l'utilisateur.",
        examples=["Qui peut présider une association étudiante ?"],
    )
    history: list[Message] = Field(
        default_factory=list,
        max_length=50,
        description="Les messages précédents, du plus ancien au plus récent. "
                    "Le serveur ne garde que les derniers échanges.",
    )
    top_k: int = Field(default=5, ge=1, le=10, description="Nombre d'extraits transmis au LLM.")
    mode: Literal["hybrid", "dense", "bm25"] = Field(default="hybrid", description="Mode de recherche.")
    use_reranker: bool = Field(default=True, description="Reclasser les candidats avec le cross-encoder.")


class Source(BaseModel):
    """An excerpt used to answer, cited as [number] in the answer."""
    number: int
    source_file: str
    page: int | str
    score: float | None = None
    rerank_score: float | None = None


class AskResponse(BaseModel):
    """The answer of the assistant."""
    answer: str
    type: Literal["documentation", "conversation", "hors_domaine"] = Field(
        description="Classification de la question par le routeur.")
    refused: bool = Field(
        description="True si l'assistant n'a pas pu répondre (information absente des documents ou question hors domaine).")
    rewritten_question: str = Field(description="Question réellement utilisée pour la recherche.")
    sources: list[Source]


class HealthResponse(BaseModel):
    """State of the API and of its dependencies."""
    status: Literal["ok", "error"]
    qdrant: str
    points: int | None = None
