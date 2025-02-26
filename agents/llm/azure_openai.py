import os
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)


def _build_azure_openai() -> AzureChatOpenAI:
    """Initialize the Azure OpenAI chat model"""
    print("Initializing LLM: AzureChatOpenAI")

    if (azure_openai_deployment_id := os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")) is None:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT_ID is not set")

    return AzureChatOpenAI(
        deployment_name=azure_openai_deployment_id,
        # model details used for tracing and token counting
        model=os.getenv("MODEL_NAME"),
        model_version=os.getenv("OPENAI_API_VERSION"),
    )


def get_azure_openai_client():
    """Set up llm client for vanna.ai"""
    azure_openai_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    return azure_openai_client
