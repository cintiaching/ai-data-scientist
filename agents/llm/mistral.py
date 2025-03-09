import os
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_mistralai import ChatMistralAI

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)


def _build_mistral() -> ChatMistralAI:
    """Initialize the Mistral chat model"""
    print("Initializing LLM: Mistral")

    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key is None:
        raise ValueError("MISTRAL_API_KEY is not set")

    return ChatMistralAI(
        model=os.getenv("MODEL_NAME"),
        temperature=0,
    )


def get_mistral_client():
    """Set up llm client for Mistral"""
    api_key = os.getenv("MISTRAL_API_KEY")
    client = Mistral(api_key=api_key)
    return client
