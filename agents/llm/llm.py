import os

from langchain_core.language_models import BaseChatModel

from agents.llm.azure_openai import _build_azure_openai, get_azure_openai_client
from agents.llm.deepseek import _build_deepseek, get_deepseek_client
from agents.llm.mistral import get_mistral_client, _build_mistral


def build_llm() -> BaseChatModel:
    """langchain llm object"""
    llm_type = os.getenv("LLM_TYPE")
    if llm_type == "azure_openai":
        return _build_azure_openai()
    if llm_type == "deepseek":
        return _build_deepseek()
    if llm_type == "mistral":
        return _build_mistral()
    raise ValueError(f"Unknown LLM type: {llm_type}. Only 'azure_openai' and 'deepseek' are currently supported.")


def get_llm_client():
    """Set up llm client for vanna.ai"""
    llm_type = os.getenv("LLM_TYPE")
    if llm_type == "azure_openai":
        return get_azure_openai_client()
    if llm_type == "deepseek":
        return get_deepseek_client()
    if llm_type == "mistral":
        return get_mistral_client()
    raise ValueError(f"Unknown LLM type: {llm_type}. Only 'azure_openai' and 'deepseek' are currently supported.")
