import os

from langchain_deepseek import ChatDeepSeek
from openai import OpenAI
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)


def _build_deepseek() -> ChatDeepSeek:
    print("Initializing LLM: ChatDeepSeek")

    if (DEEPSEEK_API_KEY := os.getenv("DEEPSEEK_API_KEY")) is None:
        raise ValueError("DEEPSEEK_API_KEY is not set")

    return ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_retries=2,
    )


def get_deepseek_client():
    if (DEEPSEEK_API_KEY := os.getenv("DEEPSEEK_API_KEY")) is None:
        raise ValueError("DEEPSEEK_API_KEY is not set")
    deepseek_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )
    return deepseek_client
