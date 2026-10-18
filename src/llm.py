import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load HUGGINGFACEHUB_API_TOKEN, HF_MODEL_ID, LLM_PROVIDER... from the .env file
load_dotenv()

_hf_chat_model = None

def get_hf_chat_model():
    """
    Load the Hugging Face chat model once and reuse it.

    Returns:
        ChatHuggingFace: The chat model (runs on Hugging Face servers, not locally).
    """
    global _hf_chat_model
    if _hf_chat_model is None:
        # HuggingFaceEndpoint only completes raw text: ChatHuggingFace adds the
        # system/user messages support on top of it
        endpoint = HuggingFaceEndpoint(
            repo_id=os.environ.get("HF_MODEL_ID", "openai/gpt-oss-20b"),
            max_new_tokens=2048,  # reasoning models use part of it to think
            temperature=0.1,      # low temperature -> answers faithful to the documents
        )
        _hf_chat_model = ChatHuggingFace(llm=endpoint)
    return _hf_chat_model


def _generate_hf(messages):
    """
    Send the messages to a Hugging Face model through the Inference Providers.

    Args:
        messages (list): The messages returned by build_messages.

    Returns:
        str: The text of the answer.
    """
    response = get_hf_chat_model().invoke(messages)
    return response.content


# Add "ollama" / "claude" here later: generate() won't change
PROVIDERS = {
    "hf": _generate_hf,
}


def generate(messages):
    """
    Single entry point to the LLM: the provider is chosen with LLM_PROVIDER in the .env file.

    Args:
        messages (list): The messages returned by build_messages.

    Returns:
        str: The text of the answer.
    """
    provider = os.environ.get("LLM_PROVIDER", "hf")
    if provider not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER inconnu : '{provider}'. Valeurs possibles : {list(PROVIDERS)}")

    return PROVIDERS[provider](messages)
