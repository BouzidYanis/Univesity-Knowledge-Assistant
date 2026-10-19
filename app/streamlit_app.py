import streamlit as st
from src.rag import answer, format_history, is_not_found

st.set_page_config(page_title="University Knowledge Assistant", page_icon=":mortar_board:", layout="wide")

ERROR_ANSWER = "Désolé, une erreur est survenue pendant la recherche ou la génération de la réponse. Réessayez dans un instant."

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
        message (dict): An assistant message with "content" and an optional "type" / "error".
    """
    if message.get("error"):
        st.error(message["content"])
    elif message.get("type") == "hors_domaine":
        st.info(message["content"])
        st.markdown("Exemples de questions :\n" + "\n".join(f"- {q}" for q in EXAMPLE_QUESTIONS))
    elif is_not_found(message["content"]):
        st.info(message["content"])
    else:
        st.markdown(message["content"])


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
    # History of the PREVIOUS messages: built before adding the new question,
    # otherwise the question would be in its own history
    history = format_history(st.session_state["messages"])

    # Question in a "user" chat bubble
    st.session_state["messages"].append({"role": "user", "content": user_input, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Answer in an "assistant" chat bubble
    with st.chat_message("assistant"):
        # The spinner is shown while the RAG pipeline runs (retrieval + LLM can take several seconds)
        with st.spinner("Recherche dans les documents..."):
            error = False
            try:
                result = answer(user_input, history=history)
            except Exception as e:
                # The details go to the console, the user gets a readable message
                print(f"Error during retrieval or generation: {e}")
                error = True
                result = {"answer": ERROR_ANSWER, "sources": []}

        assistant_message = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "question": user_input,
            "rewritten_question": result.get("rewritten_question"),
            "type": result.get("type"),
            "error": error,  # format_history skips error messages
        }
        show_answer(assistant_message)
        show_sources(assistant_message["sources"])
        show_rewritten_question(assistant_message)

    st.session_state["messages"].append(assistant_message)
