#!/usr/bin/env python3
"""
Entry point for the Privacy-Preserving Federated RAG System.

This script:
  - Creates 3 FederatedRAGClient instances (simulated hospitals).
  - Each client builds its own private ChromaDB vector store from a shard
    of the global medical literature CSV (extracted_data.csv).
  - A FederatedRAGCoordinator performs federated retrieval with
    differentially private embeddings and calls an LLM (Qwen API) to
    generate an answer.

IMPORTANT:
  - No raw document text is shared between clients.
  - Only DP-noised embeddings + light metadata (title/journal/year) are
    used by the coordinator and LLM.
"""

import os
# Fix tokenizers warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
from pathlib import Path

import yaml

# Add src directory to path
SRC_PATH = str(Path(__file__).parent / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.federated_rag import (
    create_federated_rag_clients_from_csv,
    FederatedRAGCoordinator,
)
from src.llm_providers import create_llm_provider


def main() -> None:
    # Load config (for paths, LLM, privacy params)
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found in project root.")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    rag_cfg = config["rag"]
    llm_cfg = config["llm"]
    privacy_cfg = config["privacy"]

    csv_path = data_cfg.get("medical_literature_path", "./extracted_data.csv")
    num_clients = 3
    max_docs_per_client = 8000  # as requested

    print("=" * 60)
    print("Privacy-Preserving Federated RAG System (Healthcare)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1) Build federated RAG clients (private vector DB per client)
    # ------------------------------------------------------------------
    clients, embedder = create_federated_rag_clients_from_csv(
        csv_path=csv_path,
        num_clients=num_clients,
        max_docs_per_client=max_docs_per_client,
        embedding_model=rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        vector_db_root="./federated_vector_db",
    )

    # ------------------------------------------------------------------
    # 2) Initialize LLM provider (Qwen API via OpenRouter or similar)
    # ------------------------------------------------------------------
    provider_name = llm_cfg.get("provider", "qwen_api")
    model_name = llm_cfg.get("model", "qwen/qwen3-vl-235b-a22b-instruct")
    use_api = llm_cfg.get("use_api", True)
    
    # Get API key from environment or config
    # Check provider-specific environment variables
    if provider_name.lower() == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        base_url = None  # Groq uses its own base URL
    else:
        api_key = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = llm_cfg.get("api_base_url", "https://openrouter.ai/api/v1")
    
    if not api_key and "api_key" in llm_cfg:
        api_key = llm_cfg["api_key"]

    # Build kwargs for provider
    provider_kwargs = {
        "provider": provider_name,
        "model": model_name,
        "use_api": use_api,
        "api_key": api_key,
        "temperature": llm_cfg.get("temperature", 0.3),
        "max_tokens": llm_cfg.get("max_tokens", 1000),
    }
    # Only add base_url if it's not None (Groq doesn't need it)
    if base_url is not None:
        provider_kwargs["base_url"] = base_url
    
    llm_provider = create_llm_provider(**provider_kwargs)

    print(f"\nLLM provider: {provider_name} - {model_name}")

    # ------------------------------------------------------------------
    # 3) Create coordinator with DP parameters
    # ------------------------------------------------------------------
    epsilon = float(privacy_cfg.get("epsilon", 1.0))
    delta_raw = privacy_cfg.get("delta", 1e-5)
    # Convert delta to float (YAML may read 1e-5 as string)
    delta = float(delta_raw)

    coordinator = FederatedRAGCoordinator(
        clients=clients,
        epsilon=epsilon,
        delta=delta,
        llm_provider=llm_provider,
    )

    # ------------------------------------------------------------------
    # 4) Example federated query
    # ------------------------------------------------------------------
    question = "What are the cardiovascular risk factors?"
    print("\n" + "=" * 60)
    print("Federated RAG Example Query")
    print("=" * 60)
    print(f"\nQuestion: {question}\n")

    result = coordinator.generate_answer(question, top_k_per_client=5)

    print("\nAnswer:")
    print(result["answer"])

    print(f"\nNumber of federated citations used: {result['num_references']}")
    print(f"Clients contributing: {result.get('clients_contributed', [])}")

    print("\nExample citations (metadata only, no raw text):")
    for meta in result["citations"][:5]:
        print(
            f"  [Client {meta.get('client_id', '?')}] "
            f"\"{meta.get('title', '[No title]')}\" "
            f"({meta.get('journal', 'Unknown journal')}, {meta.get('year', 'Unknown year')})"
        )

    print("\n" + "=" * 60)
    print("Federated RAG run complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

