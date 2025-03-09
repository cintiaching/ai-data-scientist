import os
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from vanna.hf import Hf

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)


def _build_huggingface_model() -> ChatHuggingFace:
    """Initialize the HuggingFace chat model"""
    print("Initializing LLM: HuggingFace")

    if (model := os.getenv("HUGGINGFACE_MODEL")) is None:
        raise ValueError("HUGGINGFACE_MODEL is not set")

    if os.getenv("HUGGINGFACEHUB_API_TOKEN") is None:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set")

    llm = HuggingFaceEndpoint(
        repo_id=model,
        task="text-generation",
    )
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model


def get_huggingface_client():
    """Set up llm client for vanna.ai"""
    config = {
        "model_name": os.getenv("HUGGINGFACE_MODEL")
    }
    huggingface_client = Hf(config=config)
    return huggingface_client
