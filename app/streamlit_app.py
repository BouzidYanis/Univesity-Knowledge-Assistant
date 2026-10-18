import streamlit as st
from src.rag import answer

st.set_page_config(page_title="University Knowledge Assistant", page_icon=":mortar_board:", layout="wide")


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


st.title("Welcome to My Streamlit App")

# The script is re-run on every interaction: st.session_state keeps the conversation between runs
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display the previous messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message["sources"])

user_input = st.chat_input("Ask a question about your university:", key="user_input")

if user_input:
    # Question in a "user" chat bubble
    st.session_state["messages"].append({"role": "user", "content": user_input, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Answer in an "assistant" chat bubble
    with st.chat_message("assistant"):
        # The spinner is shown while the RAG pipeline runs (retrieval + LLM can take several seconds)
        with st.spinner("Recherche dans les documents..."):
            try 
                result = answer(user_input)
            except Exception as e:
                st.error(f"Error during retrieval or generation: {e}")
                result = {"answer": "Sorry, I couldn't process your request.", "sources": []}

        st.markdown(result["answer"])
        show_sources(result["sources"])

    st.session_state["messages"].append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
