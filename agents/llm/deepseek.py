import os

from langchain_deepseek import ChatDeepSeek
from openai import OpenAI
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)


def _build_deepseek() -> ChatDeepSeek:
    """Initialize the DeepSeek chat model"""
    print("Initializing LLM: ChatDeepSeek")

    if (DEEPSEEK_API_KEY := os.getenv("DEEPSEEK_API_KEY")) is None:
        raise ValueError("DEEPSEEK_API_KEY is not set")

    return ChatDeepSeek(
        model=os.getenv("MODEL_NAME"),
        model_kwargs={"parallel_tool_calls": False}
    )


def get_deepseek_client():
    """Set up llm client for vanna.ai"""
    if (DEEPSEEK_API_KEY := os.getenv("DEEPSEEK_API_KEY")) is None:
        raise ValueError("DEEPSEEK_API_KEY is not set")
    deepseek_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )
    return deepseek_client
