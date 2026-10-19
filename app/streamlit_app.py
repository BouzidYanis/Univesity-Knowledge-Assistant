import os

import requests
import streamlit as st
from dotenv import load_dotenv

# The interface only talks to the API: no model, no Qdrant, no LLM token here
load_dotenv()
API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_TIMEOUT_SECONDS = 120  # without a timeout, the interface could wait forever

st.set_page_config(page_title="University Knowledge Assistant", page_icon=":mortar_board:", layout="wide")

ERROR_ANSWER = "Désolé, une erreur est survenue pendant la recherche ou la génération de la réponse. Réessayez dans un instant."
API_DOWN_ANSWER = f"Impossible de joindre l'API ({API_URL}). Vérifiez qu'elle est lancée : uvicorn api.main:app"

# Questions verified to be answered by the documents, suggested after an out-of-domain question
EXAMPLE_QUESTIONS = [
    "Qui peut présider une association étudiante ?",
    "Est-ce que je peux prêter mon mot de passe à un ami ?",
    "Quelle part de produits bio doit être servie dans les repas ?",
]


def show_answer(message):
    """
    Display an assistant answer: refusals in an info box (a limit, not an answer), the rest as text.

    Args:
        message (dict): An assistant message with "content" and optional "type" / "refused" / "error".
    """
    if message.get("error"):
        st.error(message["content"])
    elif message.get("type") == "hors_domaine":
        st.info(message["content"])
        st.markdown("Exemples de questions :\n" + "\n".join(f"- {q}" for q in EXAMPLE_QUESTIONS))
    elif message.get("refused"):
        st.info(message["content"])
    else:
        st.markdown(message["content"])


def ask_api(question, messages):
    """
    Send the question and the previous messages to the API.

    Args:
        question (str): The user's question.
        messages (list): The previous chat messages (the API cleans and limits the history itself).

    Returns:
        tuple: (result, error) - result is the JSON answer of the API, or a dict with an error message.
    """
    # Raw history (role + content), without the error messages; the last 20 messages are enough
    history = [{"role": m["role"], "content": m["content"]} for m in messages if not m.get("error")][-20:]
    try:
        response = requests.post(
            f"{API_URL}/ask", json={"question": question, "history": history}, timeout=API_TIMEOUT_SECONDS
        )
    except requests.exceptions.RequestException as e:
        print(f"API unreachable: {e}")
        return {"answer": API_DOWN_ANSWER, "sources": []}, True

    if response.status_code != 200:
        print(f"API error {response.status_code}: {response.text}")
        # 503: the API explains which dependency is down (message safe to display)
        detail = response.json().get("detail") if response.status_code == 503 else None
        return {"answer": detail if isinstance(detail, str) else ERROR_ANSWER, "sources": []}, True

    return response.json(), False


def show_sources(sources):
    """
    Display the sources under an answer (nothing if the list is empty).

    Args:
        sources (list): The sources returned by answer().
    """
    if not sources:
        return
    st.markdown("**Sources :**")
    for source in sources:
        st.markdown(f"- [{source['number']}] {source['source_file']} (page {source['page']})")


def show_rewritten_question(message):
    """
    Show the question actually searched when it differs from the one asked (follow-up questions).

    Args:
        message (dict): An assistant message with an optional "rewritten_question" and the "question" asked.
    """
    rewritten = message.get("rewritten_question")
    if rewritten and rewritten != message.get("question"):
        st.caption(f"🔎 Recherche effectuée pour : {rewritten}")


st.title("Welcome to My Streamlit App")

# The script is re-run on every interaction: st.session_state keeps the conversation between runs
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display the previous messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            show_answer(message)
        else:
            st.markdown(message["content"])
        show_sources(message["sources"])
        show_rewritten_question(message)

user_input = st.chat_input("Ask a question about your university:", key="user_input")

if user_input:
    # PREVIOUS messages: copied before adding the new question,
    # otherwise the question would be in its own history
    previous_messages = list(st.session_state["messages"])

    # Question in a "user" chat bubble
    st.session_state["messages"].append({"role": "user", "content": user_input, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Answer in an "assistant" chat bubble
    with st.chat_message("assistant"):
        # The spinner is shown while the RAG pipeline runs (retrieval + LLM can take several seconds)
        with st.spinner("Recherche dans les documents..."):
            result, error = ask_api(user_input, previous_messages)

        assistant_message = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "question": user_input,
            "rewritten_question": result.get("rewritten_question"),
            "type": result.get("type"),
            "refused": result.get("refused", False),
            "error": error,  # error messages are not sent back to the API as history
        }
        show_answer(assistant_message)
        show_sources(assistant_message["sources"])
        show_rewritten_question(assistant_message)

    st.session_state["messages"].append(assistant_message)
