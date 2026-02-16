#!/usr/bin/env python3
"""
FastAPI backend for the Federated RAG web UI.

Run from project root:
    uvicorn api.server:app --reload --host 127.0.0.1 --port 8000

Endpoints:
    POST /api/query     - Run federated RAG (question, top_k, epsilon, provider)
    GET  /api/clients   - List client status (doc counts)
    GET  /api/evaluation - Last evaluation results (from evaluation_results.json)
    POST /api/evaluate  - Run evaluation (optional, num_questions)
"""

from __future__ import annotations

import os
import sys
import json
import subprocess
from pathlib import Path
import shutil
from typing import Any, Optional

# Run from project root
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables from .env (API keys, etc.)
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=ROOT / ".env", override=False)
except Exception:
    # If python-dotenv isn't installed or .env missing, we'll rely on the process environment.
    pass

# Lazy-loaded state
_coordinator: Any = None
_clients: list = []
_config: dict = {}
# Privacy budget accounting (composition): total epsilon consumed across queries
_privacy_budget_used: float = 0.0
_privacy_budget_cap: Optional[float] = None  # from config privacy.budget_cap


def _get_config() -> dict:
    global _config
    if not _config:
        config_path = ROOT / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError("config.yaml not found")
        with open(config_path, "r") as f:
            _config = yaml.safe_load(f)
    return _config


def _init_rag():
    """Build coordinator and clients from config (lazy)."""
    global _coordinator, _clients
    if _coordinator is not None:
        return

    from src.federated_rag import (
        create_federated_rag_clients_from_csv,
        FederatedRAGCoordinator,
    )
    from src.llm_providers import create_llm_provider

    cfg = _get_config()
    data_cfg = cfg["data"]
    rag_cfg = cfg["rag"]
    llm_cfg = cfg["llm"]
    privacy_cfg = cfg["privacy"]

    csv_path = data_cfg.get("medical_literature_path", "./extracted_data.csv")
    num_clients = 3
    max_docs_per_client = 8000

    include_abstract = bool(rag_cfg.get("include_abstract_in_context", False))
    _clients, _ = create_federated_rag_clients_from_csv(
        csv_path=str(ROOT / csv_path) if not os.path.isabs(csv_path) else csv_path,
        num_clients=num_clients,
        max_docs_per_client=max_docs_per_client,
        embedding_model=rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        vector_db_root=str(ROOT / "federated_vector_db"),
        include_abstract_in_metadata=include_abstract,
    )

    provider_name = llm_cfg.get("provider", "openrouter")
    model_name = llm_cfg.get("model", "openai/gpt-4o-mini")
    use_api = llm_cfg.get("use_api", True)

    provider_lc = str(provider_name).lower()
    base_url = None

    if provider_lc == "groq":
        api_key = os.getenv("GROQ_API_KEY")
    elif provider_lc == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = llm_cfg.get("api_base_url", "https://openrouter.ai/api/v1")
    elif provider_lc == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        # DeepSeek base URL is handled inside the provider (OpenAI-compatible)
    elif provider_lc == "qwen_api":
        api_key = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = llm_cfg.get("api_base_url", os.getenv("QWEN_BASE_URL") or "https://openrouter.ai/api/v1")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and "api_key" in llm_cfg:
        api_key = llm_cfg["api_key"]

    provider_kwargs = {
        "provider": provider_name,
        "model": model_name,
        "use_api": use_api,
        "api_key": api_key,
        "temperature": llm_cfg.get("temperature", 0.3),
        "max_tokens": llm_cfg.get("max_tokens", 1500),
    }
    if base_url is not None:
        provider_kwargs["base_url"] = base_url

    llm_provider = create_llm_provider(**provider_kwargs)
    epsilon = float(privacy_cfg.get("epsilon", 1.0))
    delta_raw = privacy_cfg.get("delta", 1e-5)
    delta = float(delta_raw)
    global _privacy_budget_cap
    _privacy_budget_cap = privacy_cfg.get("budget_cap")
    if _privacy_budget_cap is not None:
        _privacy_budget_cap = float(_privacy_budget_cap)

    _coordinator = FederatedRAGCoordinator(
        clients=_clients,
        epsilon=epsilon,
        delta=delta,
        llm_provider=llm_provider,
    )


# --- Pydantic models ---

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k_per_client: int = Field(5, ge=1, le=50)
    epsilon: Optional[float] = Field(None, ge=0.01, le=10.0)
    year_min: Optional[int] = Field(None, description="Filter citations by year (inclusive)")
    year_max: Optional[int] = Field(None, description="Filter citations by year (inclusive)")


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    num_references: int
    clients_contributed: list[int]
    model_used: str
    abstracts_included: bool = False  # True if LLM received document abstracts (grounded answers)


class ClientInfo(BaseModel):
    id: int
    name: str
    status: str
    docs: int
    latency: Optional[str] = None
    version: str = "v2.1.0"


# --- App ---

app = FastAPI(
    title="Federated RAG API",
    description="Backend for Privacy-Preserving Federated RAG (Healthcare)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/query", response_model=QueryResponse)
def api_query(req: QueryRequest) -> QueryResponse:
    """Run federated RAG for a single question."""
    global _privacy_budget_used
    try:
        _init_rag()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RAG init failed: {str(e)}")

    top_k = req.top_k_per_client
    result = _coordinator.generate_answer(
        req.question,
        top_k_per_client=top_k,
        year_min=req.year_min,
        year_max=req.year_max,
    )
    # Account epsilon used (coordinator's epsilon is what's actually applied to DP)
    _privacy_budget_used += _coordinator.epsilon

    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        num_references=result["num_references"],
        clients_contributed=result.get("clients_contributed", []),
        model_used=result.get("model_used", "unknown"),
        abstracts_included=result.get("abstracts_included", False),
    )


@app.get("/api/clients", response_model=list[ClientInfo])
def api_clients() -> list[ClientInfo]:
    """Return status of each federated client (doc count, etc.)."""
    try:
        _init_rag()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RAG init failed: {str(e)}")

    names = ["General Hospital", "Memorial Center", "Research Institute"]
    out = []
    for i, client in enumerate(_clients):
        doc_count = client.collection.count()
        out.append(
            ClientInfo(
                id=client.client_id,
                name=names[i] if i < len(names) else f"Client {i}",
                status="online",
                docs=doc_count,
                latency=None,
                version="v2.1.0",
            )
        )
    return out


@app.get("/api/privacy-budget")
def api_privacy_budget() -> dict:
    """
    Return privacy budget accounting (composition).
    total_used: cumulative epsilon consumed by queries in this process.
    budget_cap: from config (null if not set).
    remaining: cap - total_used if cap is set, else null.
    """
    cap = _privacy_budget_cap
    remaining = (cap - _privacy_budget_used) if cap is not None else None
    return {
        "total_epsilon_used": round(_privacy_budget_used, 4),
        "budget_cap": round(cap, 4) if cap is not None else None,
        "remaining": round(remaining, 4) if remaining is not None else None,
        "delta": _get_config().get("privacy", {}).get("delta", 1e-5),
    }


@app.get("/api/evaluation")
def api_evaluation() -> dict:
    """Return last evaluation results from evaluation_results.json."""
    path = ROOT / "evaluation_results.json"
    if not path.exists():
        return {"statistics": {}, "results": [], "message": "No evaluation run yet."}
    with open(path, "r") as f:
        return json.load(f)


class EvaluateRequest(BaseModel):
    num_questions: int = Field(10, ge=1, le=100)


@app.post("/api/evaluate")
def api_evaluate(req: EvaluateRequest) -> dict:
    """Run evaluation script and return results."""
    path = ROOT / "evaluate_federated_rag.py"
    if not path.exists():
        raise HTTPException(status_code=501, detail="evaluate_federated_rag.py not found")
    try:
        subprocess.run(
            [sys.executable, str(path), "--num-questions", str(req.num_questions)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Evaluation timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return api_evaluation()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/rebuild-index")
def api_rebuild_index() -> dict:
    """
    Delete the federated vector DB and reset coordinator/clients so the next
    query rebuilds the index from CSV. Use this after adding DOI/PMC_ID
    support so citations include document identifiers from your dataset.
    """
    global _coordinator, _clients, _privacy_budget_used
    _coordinator = None
    _clients = []
    _privacy_budget_used = 0.0
    vector_db_path = ROOT / "federated_vector_db"
    if vector_db_path.exists():
        shutil.rmtree(vector_db_path)
    return {
        "status": "ok",
        "message": "Vector index cleared. Next query will rebuild from CSV (with DOI/PMC_ID in metadata).",
    }
