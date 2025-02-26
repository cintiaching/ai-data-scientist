import os

from langchain_core.language_models import BaseChatModel

from agents.llm.azure_openai import _build_azure_openai
from agents.llm.deepseek import _build_deepseek


def build_llm() -> BaseChatModel:
    """langchain llm object"""
    llm_type = os.getenv("LLM_TYPE")
    if llm_type == "azure_openai":
        return _build_azure_openai()
    if llm_type == "deepseek":
        return _build_deepseek()
    raise ValueError(f"Unknown LLM type: {llm_type}. Only 'azure_openai' and 'deepseek' are currently supported.")
