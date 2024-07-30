
from typing import List, Optional, Tuple
import os

from flask import Request
from src.constants import OPENAI_API_KEY
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings

def parse_api_key(bearToken: str) -> str:
    if not bearToken:
        return ""
    bearToken = bearToken.strip()
    bearToken = bearToken.replace("Bearer ", "")
    return bearToken

def get_rag_agent_prompts() -> List[str]:
    return [
        "The user has provided additional background information from scientific "
        "articles.",
        "Take the following statements into account and specifically comment on "
        "consistencies and inconsistencies with all other information available to "
        "you: {statements}",
    ]

def get_auth(authorization: str):
    # If OPENAI_API_KEY is provided by server, we will use it
    if OPENAI_API_KEY in os.environ and os.environ[OPENAI_API_KEY]:
        return os.environ[OPENAI_API_KEY]
    
    # Otherwise, we will parse it from request
    auth = authorization
    auth = auth if auth is not None and len(auth) > 0 else ""
    return parse_api_key(auth)

def get_azure_embedding_deployment() -> Tuple[bool, str, str]:
    is_azure = os.environ.get("OPENAI_API_TYPE", "") == "azure"
    deployment = os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME", "")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    return (is_azure, deployment, endpoint)

def get_embedding_function(
    is_azure: Optional[bool] = False,
    api_key: Optional[str] = None,
    model: Optional[str] = "text-embedding-ada-002",
    azure_deployment: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
):
    return (
        OpenAIEmbeddings(api_key=api_key, model=model)
        if not is_azure else
        AzureOpenAIEmbeddings(
            api_key=api_key,
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            model=model,
        )
    )

