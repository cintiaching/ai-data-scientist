import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)


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
    """langchain llm object"""
    llm_type = os.getenv("LLM_TYPE")
    if llm_type == "azure_openai":
        return _build_azure_openai()
    raise ValueError(f"Unknown LLM type: {llm_type}. Only 'azure_openai' is currently supported.")


def get_azure_openai_client():
    azure_openai_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    return azure_openai_client
