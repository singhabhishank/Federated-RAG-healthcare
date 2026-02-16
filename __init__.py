"""
Privacy-Preserving Federated RAG System for Healthcare
Retrieval-Augmented Generation with Differential Privacy
"""

from .federated_rag import (
    FederatedRAGClient,
    FederatedRAGCoordinator,
    create_federated_rag_clients_from_csv,
)
from .llm_providers import LLMProvider, create_llm_provider

__all__ = [
    "FederatedRAGClient",
    "FederatedRAGCoordinator",
    "create_federated_rag_clients_from_csv",
    "LLMProvider",
    "create_llm_provider",
]

__version__ = '1.0.0'
