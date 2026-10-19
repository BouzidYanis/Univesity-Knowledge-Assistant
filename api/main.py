"""
REST API of the University Knowledge Assistant.

Run from the project root (1 worker: each worker loads its own models):
    uvicorn api.main:app --reload

Interactive documentation: http://localhost:8000/docs
"""
import logging
from contextlib import asynccontextmanager

import httpx
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from huggingface_hub.errors import HfHubHTTPError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from api.schemas import AskRequest, AskResponse, HealthResponse
from src.embedding import client, get_embedding_model, get_sparse_model
from src.rag import OUT_OF_SCOPE_ANSWER, answer, format_history, is_not_found
from src.reranker import get_reranker

COLLECTION_NAME = "knowledge_base"

# Errors of an external dependency (Qdrant, Hugging Face LLM): 503 instead of 500
DEPENDENCY_ERRORS = (
    ConnectionError,
    TimeoutError,
    ResponseHandlingException,
    UnexpectedResponse,
    HfHubHTTPError,
    httpx.HTTPError,
    requests.exceptions.RequestException,
)

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# One log line per HTTP request to Qdrant / Hugging Face is too verbose
logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: check Qdrant and load the models once, so that the first question is not slower than the others.
    An API that starts but cannot answer is worse than an API that refuses to start.
    """
    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(f"La collection Qdrant '{COLLECTION_NAME}' n'existe pas : indexer les documents d'abord.")

    logger.info("Chargement des modèles (embedding e5, BM25, reranker)...")
    get_embedding_model()
    get_sparse_model()
    get_reranker()
    logger.info("Modèles chargés.")
    yield


app = FastAPI(
    title="University Knowledge Assistant API",
    description="Questions-réponses sur les documents officiels d'Avignon Université (RAG).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
def health():
    """Check that the API and Qdrant are working (collection present and not empty)."""
    try:
        points = client.count(COLLECTION_NAME).count
    except Exception as e:
        logger.error("Health check: Qdrant indisponible: %s", e)
        return JSONResponse(status_code=503, content={"status": "error", "qdrant": "indisponible"})

    if points == 0:
        return JSONResponse(status_code=503, content={"status": "error", "qdrant": "collection vide", "points": 0})

    return {"status": "ok", "qdrant": "ok", "points": points}


# def (not async def): answer() is blocking (torch, Qdrant, HTTP call to the LLM). FastAPI runs a def
# endpoint in a separate thread, so the server stays available during the few seconds of an answer.
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Answer a question from the official documents, using the conversation history for follow-up questions."""
    # The server limits and cleans the history itself: never trust the client for this
    history = format_history([message.model_dump() for message in request.history])

    try:
        result = answer(
            request.question,
            top_k=request.top_k,
            mode=request.mode,
            use_reranker=request.use_reranker,
            history=history,
        )
    except DEPENDENCY_ERRORS as e:
        # The details go to the logs, never to the client (paths, URLs, keys...)
        logger.exception("Dépendance indisponible pendant /ask: %s", e)
        raise HTTPException(status_code=503, detail="Service temporairement indisponible (base de documents ou LLM).")
    except Exception as e:
        logger.exception("Erreur inattendue pendant /ask: %s", e)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")

    return {
        **result,
        "refused": is_not_found(result["answer"]) or result["answer"] == OUT_OF_SCOPE_ANSWER,
    }
