from src.embedding import search_in_qdrant
from src.llm import generate

NOT_FOUND_ANSWER = "Je ne trouve pas cette information dans les documents de l'université."

SYSTEM_PROMPT = f"""Tu es l'assistant de connaissances de l'université. Tu aides les étudiants et le personnel à trouver des informations dans les documents officiels de l'université.

Règles :
1. Réponds UNIQUEMENT à partir des extraits fournis dans le contexte. N'utilise jamais tes connaissances générales.
2. Si le contexte ne contient pas la réponse, réponds exactement : "{NOT_FOUND_ANSWER}" N'invente rien.
3. Cite tes sources avec le numéro de l'extrait entre crochets, juste après l'information concernée, par exemple [1] ou [2][3].
4. Si les extraits se contredisent, signale-le et cite les deux sources.
5. Réponds dans la langue de la question, de façon claire et concise."""

USER_PROMPT_TEMPLATE = """Contexte :
{context}

Question :
{question}"""


def build_messages(context, question):
    """
    Build the chat messages sent to the LLM.

    Args:
        context (str): The numbered excerpts returned by build_context.
        question (str): The user's question.

    Returns:
        list: The messages in the {"role", "content"} format (Ollama / OpenAI / Mistral).
              For Claude, pass SYSTEM_PROMPT in the `system=` parameter and keep only the user message.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(context=context, question=question)},
    ]


def build_context(results):
    """
    Turn the search results into a single numbered text block for the prompt.

    Args:
        results (list): The results returned by search_in_qdrant.

    Returns:
        tuple: (context, sources)
            context (str): The numbered excerpts, e.g. "[1] (file.pdf, p.3)\n<text>".
            sources (list): One dict per excerpt (number, file, page, score) to display the citations.
    """
    excerpts = []
    sources = []

    for number, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        source_file = metadata.get("source_file", "inconnu")

        # PyMuPDF counts pages from 0, humans count from 1
        page = metadata.get("page")
        page = page + 1 if isinstance(page, int) else "?"

        excerpts.append(f"[{number}] ({source_file}, p.{page})\n{result.get('text', '').strip()}")
        sources.append({
            "number": number,
            "source_file": source_file,
            "page": page,
            "score": result.get("score"),
        })

    context = "\n\n---\n\n".join(excerpts)
    return context, sources


def answer(question, top_k=5):
    """
    Full RAG pipeline: retrieve the relevant chunks, build the prompt and generate the answer.

    Args:
        question (str): The user's question.
        top_k (int): The number of chunks to retrieve from Qdrant.

    Returns:
        dict: {"answer": str, "sources": list} - sources is empty when nothing was found.
    """
    question = question.strip()
    if not question:
        return {"answer": "Pose une question.", "sources": []}

    # 1. Retrieval
    results = search_in_qdrant(question, top_k=top_k)

    # No chunk found (empty collection, Qdrant down...): no need to call the LLM
    if not results:
        return {"answer": NOT_FOUND_ANSWER, "sources": []}

    # 2. Context + 3. Prompt
    context, sources = build_context(results)
    messages = build_messages(context, question)

    # 4. Generation
    response = generate(messages).strip()

    # The LLM said it doesn't know: the excerpts were not useful, don't show them as sources
    if response == NOT_FOUND_ANSWER:
        sources = []

    return {"answer": response, "sources": sources}
