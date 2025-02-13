import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI


def _build_azure_openai() -> AzureChatOpenAI:
    print("Initializing LLM: AzureChatOpenAI")

    if (azure_openai_deployment_id := os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")) is None:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT_ID is not set")

    return AzureChatOpenAI(
        deployment_name=azure_openai_deployment_id,
        # model details used for tracing and token counting
        model=os.getenv("AZURE_OPENAI_MODEL_NAME"),
        model_version=os.getenv("OPENAI_API_VERSION"),
    )


def build_llm() -> BaseChatModel:
    llm_type = os.getenv("LLM_TYPE")
    if llm_type == "azure_openai":
        return _build_azure_openai()
    raise ValueError(f"Unknown LLM type: {llm_type}. Only 'azure_openai' is currently supported.")
