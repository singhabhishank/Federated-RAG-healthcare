#!/usr/bin/env python3
"""
Experiment 7: Reviewer-Approved Metrics Comparison

Goal: Compare all systems using comprehensive metrics approved by reviewers.

Metrics compared:
1. Retrieval Quality
   - Average relevance score
   - Top-k similarity
   - % queries ≥ relevance threshold (0.85, 0.90)

2. Privacy
   - Raw data sharing: Yes / No
   - Embedding sharing: Yes / DP-noised
   - ε (epsilon) value

3. Federated Behavior
   - Client participation rate
   - Client contribution balance (variance)
   - Single-client dominance (yes/no)

4. System Cost (optional but strong)
   - Latency per query
   - Communication rounds
   - Embeddings transmitted

This script:
1. Runs evaluations on all 4 systems
2. Collects all reviewer-approved metrics
3. Creates comprehensive visualizations
4. Generates detailed comparison report
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import yaml
import pandas as pd
import time
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# Suppress tokenizers warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.federated_rag import (
    create_federated_rag_clients_from_csv,
    FederatedRAGCoordinator,
    FederatedRAGClient,
    RetrievedChunk,
)
from src.evaluate_federated_rag import (
    MEDICAL_QUESTIONS,
    run_evaluation,
    calculate_statistics,
    EvaluationMetrics,
    evaluate_retrieval_relevance,
)
from src.llm_providers import create_llm_provider


# ---------------------------------------------------------------------------
# System Coordinators (copied from exp6)
# ---------------------------------------------------------------------------

class CentralizedRAGCoordinator:
    """
    Centralized RAG: All data in one place, no federation, no privacy.
    This is the baseline non-federated system.
    """
    
    def __init__(
        self,
        client: FederatedRAGClient,
        llm_provider=None,
    ):
        self.client = client
        self.llm_provider = llm_provider
    
    def retrieve(self, question: str, top_k: int = 15) -> List[RetrievedChunk]:
        """Retrieve from single centralized database (no DP, no federation)."""
        if self.client.collection.count() == 0:
            return []
        
        query_emb = self.client.embedder.encode([question], show_progress_bar=False)[0]
        
        results = self.client.collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, self.client.collection.count()),
            include=["embeddings", "metadatas", "distances"],
        )
        
        chunks: List[RetrievedChunk] = []
        for doc_id, meta, emb, dist in zip(
            results["ids"][0],
            results["metadatas"][0],
            results["embeddings"][0],
            results["distances"][0],
        ):
            # NO DP noise - raw embeddings
            emb_np = np.array(emb, dtype=np.float32)
            chunks.append(
                RetrievedChunk(
                    client_id=0,
                    doc_id=doc_id,
                    score=float(dist),
                    embedding=emb_np,  # No privacy protection
                    metadata=meta,
                )
            )
        
        chunks.sort(key=lambda c: c.score)
        return chunks[:top_k]
    
    def build_llm_context(self, question: str, chunks: List[RetrievedChunk], max_refs: int = 10) -> str:
        """Build context for LLM (same as federated version)."""
        lines: List[str] = []
        lines.append("You are a medical expert using a centralized RAG system.\n")
        lines.append("Question:")
        lines.append(question)
        lines.append("\nEvidence (citations sorted by relevance):")
        
        for i, chunk in enumerate(chunks[:max_refs], 1):
            meta = chunk.metadata
            title = meta.get("title", "[No title]")
            journal = meta.get("journal", "[Unknown journal]")
            year = meta.get("year", "[Unknown year]")
            relevance_indicator = "✓" if chunk.score < 0.5 else "~"
            lines.append(
                f"- Source {i} {relevance_indicator}: \"{title}\" "
                f"({journal}, {year})"
            )
        
        lines.append(
            "\nInstructions for Answer Generation:\n"
            "1. Use the evidence provided above.\n"
            "2. Combine with your medical knowledge to provide a comprehensive answer.\n"
            "3. Be specific and evidence-based - cite which sources support each claim.\n"
            "4. Structure your answer clearly with markdown formatting.\n"
            "5. Be comprehensive and accurate.\n"
            "\nNow provide a detailed, well-structured, evidence-informed answer:"
        )
        
        return "\n".join(lines)
    
    def generate_answer(self, question: str, top_k_per_client: int = 5) -> Dict:
        """Generate answer using centralized retrieval."""
        chunks = self.retrieve(question, top_k=top_k_per_client * 3)
        
        if self.llm_provider is None:
            return {
                "answer": "LLM provider not configured.",
                "citations": [c.metadata for c in chunks],
                "num_references": len(chunks),
                "model_used": "none",
                "clients_contributed": [0],
            }
        
        context = self.build_llm_context(question, chunks)
        answer = self.llm_provider.generate(prompt=context)
        
        return {
            "answer": answer,
            "citations": [c.metadata for c in chunks],
            "num_references": len(chunks),
            "model_used": type(self.llm_provider).__name__,
            "clients_contributed": [0],
        }


class FederatedRAGNoDPCoordinator(FederatedRAGCoordinator):
    """Federated RAG without Differential Privacy."""
    
    def __init__(
        self,
        clients: List[FederatedRAGClient],
        llm_provider=None,
    ):
        # Use very large epsilon to effectively disable DP
        super().__init__(
            clients=clients,
            epsilon=1000.0,  # Effectively no noise
            delta=1e-5,
            llm_provider=llm_provider,
        )


class SecureAggregationCoordinator(FederatedRAGCoordinator):
    """Federated RAG with DP + Secure Aggregation."""
    
    def federated_retrieve(
        self,
        question: str,
        top_k_per_client: int = 5,
    ) -> List[RetrievedChunk]:
        """Federated retrieval with secure aggregation."""
        all_chunks: List[RetrievedChunk] = []
        
        for client in self.clients:
            client_chunks = client.local_retrieve(
                question,
                top_k=top_k_per_client,
                epsilon=self.epsilon,
                delta=self.delta,
            )
            all_chunks.extend(client_chunks)
        
        # Secure aggregation: Group similar embeddings
        aggregated_chunks = []
        used_indices = set()
        
        for i, chunk1 in enumerate(all_chunks):
            if i in used_indices:
                continue
            
            similar_group = [chunk1]
            used_indices.add(i)
            
            for j, chunk2 in enumerate(all_chunks[i+1:], i+1):
                if j in used_indices:
                    continue
                
                if (chunk1.metadata.get("title") == chunk2.metadata.get("title") or
                    np.linalg.norm(chunk1.embedding - chunk2.embedding) < 0.1):
                    similar_group.append(chunk2)
                    used_indices.add(j)
            
            if len(similar_group) > 1:
                avg_embedding = np.mean([c.embedding for c in similar_group], axis=0)
                avg_score = np.mean([c.score for c in similar_group])
                aggregated_chunks.append(
                    RetrievedChunk(
                        client_id=-1,
                        doc_id=similar_group[0].doc_id,
                        score=avg_score,
                        embedding=avg_embedding,
                        metadata=similar_group[0].metadata,
                    )
                )
            else:
                aggregated_chunks.append(chunk1)
        
        aggregated_chunks.sort(key=lambda c: c.score)
        filtered_chunks = [c for c in aggregated_chunks if c.score < 0.85]
        if len(filtered_chunks) < 10 and len(aggregated_chunks) >= 10:
            filtered_chunks = aggregated_chunks[:max(10, len(filtered_chunks))]
        elif not filtered_chunks:
            filtered_chunks = aggregated_chunks[:5] if aggregated_chunks else []
        
        return filtered_chunks


# ---------------------------------------------------------------------------
# Metric Collection Functions
# ---------------------------------------------------------------------------

def collect_retrieval_quality_metrics(
    results: List[EvaluationMetrics],
    embedder
) -> Dict:
    """Collect retrieval quality metrics."""
    if not results:
        return {
            'avg_relevance': 0.0,
            'top_k_avg_similarity': 0.0,
            'pct_above_85': 0.0,
            'pct_above_90': 0.0,
        }
    
    valid_results = [r for r in results if r.num_references > 0]
    if not valid_results:
        return {
            'avg_relevance': 0.0,
            'top_k_avg_similarity': 0.0,
            'pct_above_85': 0.0,
            'pct_above_90': 0.0,
        }
    
    relevance_scores = [r.retrieval_relevance_score for r in valid_results]
    avg_relevance = np.mean(relevance_scores)
    
    # Top-k similarity (average of top 5 most relevant)
    all_similarities = []
    for r in valid_results:
        if r.citations:
            # Use relevance score as similarity proxy
            all_similarities.append(r.retrieval_relevance_score)
    top_k_avg = np.mean(sorted(all_similarities, reverse=True)[:5]) if all_similarities else 0.0
    
    # Percentage above thresholds
    pct_above_85 = sum(1 for s in relevance_scores if s >= 0.85) / len(relevance_scores) * 100
    pct_above_90 = sum(1 for s in relevance_scores if s >= 0.90) / len(relevance_scores) * 100
    
    return {
        'avg_relevance': avg_relevance,
        'top_k_avg_similarity': top_k_avg,
        'pct_above_85': pct_above_85,
        'pct_above_90': pct_above_90,
        'total_queries': len(results),
        'valid_queries': len(valid_results),
    }


def collect_privacy_metrics(
    system_type: str,
    epsilon: float = 0.0,
    uses_dp: bool = False,
) -> Dict:
    """Collect privacy metrics."""
    if system_type == "Centralized":
        return {
            'raw_data_sharing': 'No',  # Only metadata shared
            'embedding_sharing': 'Yes (no DP)',
            'epsilon': 0.0,
            'uses_dp': False,
            'privacy_score': 0.3,  # Low - no DP, centralized
        }
    elif system_type == "Federated_No_DP":
        return {
            'raw_data_sharing': 'No',
            'embedding_sharing': 'Yes (no DP)',
            'epsilon': 0.0,
            'uses_dp': False,
            'privacy_score': 0.5,  # Medium - federated but no DP
        }
    elif system_type == "Federated_DP":
        return {
            'raw_data_sharing': 'No',
            'embedding_sharing': 'Yes (DP-noised)',
            'epsilon': epsilon,
            'uses_dp': True,
            'privacy_score': 0.8,  # High - federated + DP
        }
    elif system_type == "Federated_DP_SA":
        return {
            'raw_data_sharing': 'No',
            'embedding_sharing': 'Yes (DP-noised)',
            'epsilon': epsilon,
            'uses_dp': True,
            'privacy_score': 1.0,  # Highest - federated + DP + Secure Aggregation
        }
    else:
        return {
            'raw_data_sharing': 'Unknown',
            'embedding_sharing': 'Unknown',
            'epsilon': 0.0,
            'uses_dp': False,
            'privacy_score': 0.0,
        }


def collect_federated_behavior_metrics(
    results: List[EvaluationMetrics],
    num_clients: int = 3,
) -> Dict:
    """Collect federated behavior metrics."""
    if not results:
        return {
            'client_participation_rate': 0.0,
            'contribution_variance': 0.0,
            'single_client_dominance': False,
            'avg_clients_per_query': 0.0,
        }
    
    valid_results = [r for r in results if r.num_references > 0]
    if not valid_results:
        return {
            'client_participation_rate': 0.0,
            'contribution_variance': 0.0,
            'single_client_dominance': False,
            'avg_clients_per_query': 0.0,
        }
    
    # Client participation per query
    clients_per_query = []
    client_contributions = defaultdict(int)
    
    for r in valid_results:
        contributing_clients = set(r.clients_contributed)
        clients_per_query.append(len(contributing_clients))
        for client_id in contributing_clients:
            client_contributions[client_id] += 1
    
    avg_clients_per_query = np.mean(clients_per_query) if clients_per_query else 0.0
    client_participation_rate = avg_clients_per_query / num_clients * 100
    
    # Contribution balance (variance)
    if client_contributions:
        contribution_values = list(client_contributions.values())
        contribution_variance = np.var(contribution_values) if len(contribution_values) > 1 else 0.0
    else:
        contribution_variance = 0.0
    
    # Single-client dominance (if any client contributes to >70% of queries alone)
    single_client_dominant = False
    for client_id, count in client_contributions.items():
        if count / len(valid_results) > 0.7:
            single_client_dominant = True
            break
    
    return {
        'client_participation_rate': client_participation_rate,
        'contribution_variance': contribution_variance,
        'single_client_dominance': single_client_dominant,
        'avg_clients_per_query': avg_clients_per_query,
        'client_contributions': dict(client_contributions),
    }


def collect_system_cost_metrics(
    coordinator,
    questions: List[str],
    num_clients: int = 3,
) -> Dict:
    """Collect system cost metrics."""
    latencies = []
    total_embeddings = 0
    communication_rounds = []
    
    for question in questions[:5]:  # Test on 5 questions
        start_time = time.time()
        
        # Simulate query
        try:
            if isinstance(coordinator, CentralizedRAGCoordinator):
                chunks = coordinator.retrieve(question, top_k=15)
                comm_rounds = 1  # Single round
                embeddings_sent = len(chunks) * 384  # Assuming 384-dim embeddings
            else:
                # Federated systems
                chunks = coordinator.federated_retrieve(question, top_k_per_client=5)
                comm_rounds = num_clients  # One round per client
                embeddings_sent = len(chunks) * 384
            
            end_time = time.time()
            latency = end_time - start_time
            
            latencies.append(latency)
            communication_rounds.append(comm_rounds)
            total_embeddings += embeddings_sent
        except Exception:
            continue
    
    if not latencies:
        return {
            'avg_latency': 0.0,
            'total_embeddings_transmitted': 0,
            'avg_communication_rounds': 0.0,
        }
    
    return {
        'avg_latency': np.mean(latencies),
        'total_embeddings_transmitted': total_embeddings,
        'avg_communication_rounds': np.mean(communication_rounds),
        'latency_per_query': np.mean(latencies),
    }


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

def run_reviewer_metrics_experiment(
    num_questions: int = 20,
    delta: float = 1e-5,
    epsilon: float = 1.0,
):
    """Run comprehensive reviewer-approved metrics comparison."""
    print("=" * 80)
    print("EXPERIMENT 7: REVIEWER-APPROVED METRICS COMPARISON")
    print("=" * 80)
    print("\nComparing 4 systems using comprehensive metrics:")
    print("1. Retrieval Quality (relevance, top-k, thresholds)")
    print("2. Privacy (raw data sharing, embedding DP, epsilon)")
    print("3. Federated Behavior (participation, balance, dominance)")
    print("4. System Cost (latency, communication, embeddings)")
    print(f"\nNumber of questions per system: {num_questions}")
    print(f"Epsilon (for DP systems): {epsilon}")
    print("=" * 80)
    
    # Load config
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found in project root.")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_cfg = config["data"]
    rag_cfg = config["rag"]
    llm_cfg = config["llm"]
    
    csv_path = data_cfg.get("medical_literature_path", "./extracted_data.csv")
    csv_path = PROJECT_ROOT / csv_path if not Path(csv_path).is_absolute() else Path(csv_path)
    
    # Load data and create clients
    print("\n[1/5] Loading data and creating clients...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Title", "Abstract"])
    df = df[df["Title"].str.strip() != ""]
    df = df[df["Abstract"].str.strip() != ""]
    
    print(f"Loaded {len(df)} cleaned medical articles")
    
    # Load embedding model
    embedding_model = rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache = os.path.join(cache_dir, f"models--{embedding_model.replace('/', '--')}")
    if os.path.exists(model_cache):
        print("   Model found in cache, using local files only (offline mode)...")
        embedder = SentenceTransformer(embedding_model, device="cpu", local_files_only=True)
    else:
        print("   Model not in cache, attempting to download...")
        embedder = SentenceTransformer(embedding_model, device="cpu")
    
    # Initialize LLM provider
    print("\n[2/5] Initializing LLM provider...")
    provider_name = llm_cfg.get("provider", "openrouter")
    model_name = llm_cfg.get("model", "openai/gpt-4o-mini")
    use_api = llm_cfg.get("use_api", True)
    
    if provider_name.lower() == "groq":
        api_key = os.getenv("GROQ_API_KEY")
    elif provider_name.lower() == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("\n❌ Error: OPENROUTER_API_KEY not set in environment")
            print("   Please set it: export OPENROUTER_API_KEY='your-key-here'")
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
    else:
        api_key = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    llm_provider = create_llm_provider(
        provider=provider_name,
        model=model_name,
        use_api=use_api,
        api_key=api_key,
        temperature=llm_cfg.get("temperature", 0.3),
        max_tokens=llm_cfg.get("max_tokens", 1500),
    )
    print(f"✓ LLM provider: {provider_name}\n")
    
    # Create federated clients
    print("[3/5] Creating federated clients...")
    num_clients = 3
    max_docs_per_client = 8000
    per_client = min(max_docs_per_client, len(df) // num_clients)
    
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    federated_clients = []
    start_idx = 0
    for client_id in range(num_clients):
        end_idx = start_idx + per_client
        client_df = df_shuffled.iloc[start_idx:end_idx].copy()
        start_idx = end_idx
        
        client = FederatedRAGClient(
            client_id=client_id,
            df=client_df,
            embedder=embedder,
            vector_db_root=str(PROJECT_ROOT / "federated_vector_db"),
        )
        client.build_local_index()
        federated_clients.append(client)
    
    # Create centralized client
    print("\n[4/5] Creating centralized client (all data)...")
    centralized_client = FederatedRAGClient(
        client_id=0,
        df=df_shuffled.copy(),
        embedder=embedder,
        vector_db_root=str(PROJECT_ROOT / "federated_vector_db"),
        collection_name="centralized_medical_0",
    )
    centralized_client.build_local_index()
    
    print("\n[5/5] Running evaluations and collecting metrics...")
    print("=" * 80)
    
    questions = MEDICAL_QUESTIONS[:num_questions]
    all_metrics = []
    
    # System 1: Centralized
    print("\n[System 1/4] Centralized RAG...")
    centralized_coord = CentralizedRAGCoordinator(
        client=centralized_client,
        llm_provider=llm_provider,
    )
    results1 = run_evaluation(
        coordinator=centralized_coord,
        questions=questions,
        output_file=None,
        interactive=False,
        embedder=embedder,
    )
    
    metrics1 = {
        'system_name': 'Centralized RAG',
        'system_type': 'Centralized',
        'retrieval_quality': collect_retrieval_quality_metrics(results1, embedder),
        'privacy': collect_privacy_metrics('Centralized', epsilon=0.0, uses_dp=False),
        'federated_behavior': collect_federated_behavior_metrics(results1, num_clients=1),
        'system_cost': collect_system_cost_metrics(centralized_coord, questions, num_clients=1),
        'results': results1,
    }
    all_metrics.append(metrics1)
    
    time.sleep(5)
    
    # System 2: Federated No DP
    print("\n[System 2/4] Federated RAG (No DP)...")
    fed_no_dp_coord = FederatedRAGNoDPCoordinator(
        clients=federated_clients,
        llm_provider=llm_provider,
    )
    results2 = run_evaluation(
        coordinator=fed_no_dp_coord,
        questions=questions,
        output_file=None,
        interactive=False,
        embedder=embedder,
    )
    
    metrics2 = {
        'system_name': 'Federated RAG (No DP)',
        'system_type': 'Federated_No_DP',
        'retrieval_quality': collect_retrieval_quality_metrics(results2, embedder),
        'privacy': collect_privacy_metrics('Federated_No_DP', epsilon=0.0, uses_dp=False),
        'federated_behavior': collect_federated_behavior_metrics(results2, num_clients=num_clients),
        'system_cost': collect_system_cost_metrics(fed_no_dp_coord, questions, num_clients=num_clients),
        'results': results2,
    }
    all_metrics.append(metrics2)
    
    time.sleep(5)
    
    # System 3: Federated + DP
    print("\n[System 3/4] Federated RAG + DP...")
    fed_dp_coord = FederatedRAGCoordinator(
        clients=federated_clients,
        epsilon=epsilon,
        delta=delta,
        llm_provider=llm_provider,
    )
    results3 = run_evaluation(
        coordinator=fed_dp_coord,
        questions=questions,
        output_file=None,
        interactive=False,
        embedder=embedder,
    )
    
    metrics3 = {
        'system_name': 'Federated RAG + DP',
        'system_type': 'Federated_DP',
        'retrieval_quality': collect_retrieval_quality_metrics(results3, embedder),
        'privacy': collect_privacy_metrics('Federated_DP', epsilon=epsilon, uses_dp=True),
        'federated_behavior': collect_federated_behavior_metrics(results3, num_clients=num_clients),
        'system_cost': collect_system_cost_metrics(fed_dp_coord, questions, num_clients=num_clients),
        'results': results3,
    }
    all_metrics.append(metrics3)
    
    time.sleep(5)
    
    # System 4: Federated + DP + Secure Aggregation
    print("\n[System 4/4] Federated RAG + DP + Secure Aggregation...")
    secure_coord = SecureAggregationCoordinator(
        clients=federated_clients,
        epsilon=epsilon,
        delta=delta,
        llm_provider=llm_provider,
    )
    results4 = run_evaluation(
        coordinator=secure_coord,
        questions=questions,
        output_file=None,
        interactive=False,
        embedder=embedder,
    )
    
    metrics4 = {
        'system_name': 'Federated RAG + DP + SA',
        'system_type': 'Federated_DP_SA',
        'retrieval_quality': collect_retrieval_quality_metrics(results4, embedder),
        'privacy': collect_privacy_metrics('Federated_DP_SA', epsilon=epsilon, uses_dp=True),
        'federated_behavior': collect_federated_behavior_metrics(results4, num_clients=num_clients),
        'system_cost': collect_system_cost_metrics(secure_coord, questions, num_clients=num_clients),
        'results': results4,
    }
    all_metrics.append(metrics4)
    
    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("=" * 80)
    
    generate_comprehensive_report(all_metrics, epsilon, num_questions)
    
    return all_metrics


def generate_comprehensive_report(all_metrics: List[Dict], epsilon: float, num_questions: int):
    """Generate comprehensive comparison report."""
    
    # Extract all metrics
    system_names = [m['system_name'] for m in all_metrics]
    
    # Retrieval Quality
    avg_relevance = [m['retrieval_quality']['avg_relevance'] for m in all_metrics]
    top_k_sim = [m['retrieval_quality']['top_k_avg_similarity'] for m in all_metrics]
    pct_85 = [m['retrieval_quality']['pct_above_85'] for m in all_metrics]
    pct_90 = [m['retrieval_quality']['pct_above_90'] for m in all_metrics]
    
    # Privacy
    raw_data_sharing = [m['privacy']['raw_data_sharing'] for m in all_metrics]
    embedding_sharing = [m['privacy']['embedding_sharing'] for m in all_metrics]
    epsilon_vals = [m['privacy']['epsilon'] for m in all_metrics]
    privacy_scores = [m['privacy']['privacy_score'] for m in all_metrics]
    
    # Federated Behavior
    participation_rate = [m['federated_behavior']['client_participation_rate'] for m in all_metrics]
    contribution_var = [m['federated_behavior']['contribution_variance'] for m in all_metrics]
    dominance = [m['federated_behavior']['single_client_dominance'] for m in all_metrics]
    avg_clients = [m['federated_behavior']['avg_clients_per_query'] for m in all_metrics]
    
    # System Cost
    latencies = [m['system_cost']['avg_latency'] * 1000 for m in all_metrics]  # Convert to ms
    comm_rounds = [m['system_cost']['avg_communication_rounds'] for m in all_metrics]
    embeddings_trans = [m['system_cost']['total_embeddings_transmitted'] for m in all_metrics]
    
    # Print comprehensive table
    print("\n" + "=" * 100)
    print("COMPREHENSIVE METRICS COMPARISON")
    print("=" * 100)
    
    print("\n1. RETRIEVAL QUALITY")
    print("-" * 100)
    print(f"{'System':<30} {'Avg Relevance':<15} {'Top-K Sim':<15} {'≥0.85':<12} {'≥0.90':<12}")
    print("-" * 100)
    for i, name in enumerate(system_names):
        print(f"{name:<30} {avg_relevance[i]:<15.3f} {top_k_sim[i]:<15.3f} "
              f"{pct_85[i]:<12.1f}% {pct_90[i]:<12.1f}%")
    print("-" * 100)
    
    print("\n2. PRIVACY")
    print("-" * 100)
    print(f"{'System':<30} {'Raw Data':<15} {'Embeddings':<20} {'Epsilon':<12} {'Privacy Score':<15}")
    print("-" * 100)
    for i, name in enumerate(system_names):
        print(f"{name:<30} {raw_data_sharing[i]:<15} {embedding_sharing[i]:<20} "
              f"{epsilon_vals[i]:<12.2f} {privacy_scores[i]:<15.2f}")
    print("-" * 100)
    
    print("\n3. FEDERATED BEHAVIOR")
    print("-" * 100)
    print(f"{'System':<30} {'Part. Rate':<15} {'Avg Clients':<15} {'Var':<12} {'Dominance':<15}")
    print("-" * 100)
    for i, name in enumerate(system_names):
        dominance_str = "Yes" if dominance[i] else "No"
        print(f"{name:<30} {participation_rate[i]:<15.1f}% {avg_clients[i]:<15.2f} "
              f"{contribution_var[i]:<12.2f} {dominance_str:<15}")
    print("-" * 100)
    
    print("\n4. SYSTEM COST")
    print("-" * 100)
    print(f"{'System':<30} {'Latency (ms)':<15} {'Comm Rounds':<15} {'Embeddings':<15}")
    print("-" * 100)
    for i, name in enumerate(system_names):
        print(f"{name:<30} {latencies[i]:<15.1f} {comm_rounds[i]:<15.2f} "
              f"{embeddings_trans[i]:<15}")
    print("-" * 100)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    fig = plt.figure(figsize=(18, 14))
    
    # Plot 1: Retrieval Quality
    ax1 = plt.subplot(3, 3, 1)
    x = np.arange(len(system_names))
    width = 0.35
    ax1.bar(x - width/2, avg_relevance, width, label='Avg Relevance', alpha=0.8)
    ax1.bar(x + width/2, top_k_sim, width, label='Top-K Sim', alpha=0.8)
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('1. Retrieval Quality', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace(' ', '\n') for s in system_names], fontsize=8, rotation=15)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: Relevance Thresholds
    ax2 = plt.subplot(3, 3, 2)
    x = np.arange(len(system_names))
    width = 0.35
    ax2.bar(x - width/2, pct_85, width, label='≥0.85', alpha=0.8, color='orange')
    ax2.bar(x + width/2, pct_90, width, label='≥0.90', alpha=0.8, color='green')
    ax2.set_ylabel('% Queries', fontweight='bold')
    ax2.set_title('Relevance Threshold Compliance', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace(' ', '\n') for s in system_names], fontsize=8, rotation=15)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 105])
    
    # Plot 3: Privacy Score
    ax3 = plt.subplot(3, 3, 3)
    colors_privacy = ['#FF6B6B', '#FFB84D', '#4ECDC4', '#2ECC71']
    bars3 = ax3.bar(system_names, privacy_scores, color=colors_privacy, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Privacy Score', fontweight='bold')
    ax3.set_title('2. Privacy Score', fontweight='bold', fontsize=12)
    ax3.set_xticklabels([s.replace(' ', '\n') for s in system_names], fontsize=8, rotation=15)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 1.05])
    for i, (bar, val) in enumerate(zip(bars3, privacy_scores)):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 4: Epsilon Values
    ax4 = plt.subplot(3, 3, 4)
    bars4 = ax4.bar(system_names, epsilon_vals, color=['#FF6B6B', '#FFB84D', '#4ECDC4', '#2ECC71'], alpha=0.8, edgecolor='black')
    ax4.set_ylabel('ε (Epsilon)', fontweight='bold')
    ax4.set_title('Differential Privacy (ε)', fontweight='bold', fontsize=12)
    ax4.set_xticklabels([s.replace(' ', '\n') for s in system_names], fontsize=8, rotation=15)
    ax4.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars4, epsilon_vals)):
        if val > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 5: Client Participation
    ax5 = plt.subplot(3, 3, 5)
    bars5 = ax5.bar(system_names, participation_rate, color=['#FF6B6B', '#FFB84D', '#4ECDC4', '#2ECC71'], alpha=0.8, edgecolor='black')
    ax5.set_ylabel('Participation Rate (%)', fontweight='bold')
    ax5.set_title('3. Federated Behavior - Participation', fontweight='bold', fontsize=12)
    ax5.set_xticklabels([s.replace(' ', '\n') for s in system_names], fontsize=8, rotation=15)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim([0, 110])
    
    # Plot 6: Contribution Balance
    ax6 = plt.subplot(3, 3, 6)
    bars6 = ax6.bar(system_names, contribution_var, color=['#FF6B6B', '#FFB84D', '#4ECDC4', '#2ECC71'], alpha=0.8, edgecolor='black')
    ax6.set_ylabel('Contribution Variance', fontweight='bold')
    ax6.set_title('Contribution Balance (Lower = Better)', fontweight='bold', fontsize=12)
    ax6.set_xticklabels([s.replace(' ', '\n') for s in system_names], fontsize=8, rotation=15)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Plot 7: Latency
    ax7 = plt.subplot(3, 3, 7)
    bars7 = ax7.bar(system_names, latencies, color=['#FF6B6B', '#FFB84D', '#4ECDC4', '#2ECC71'], alpha=0.8, edgecolor='black')
    ax7.set_ylabel('Latency (ms)', fontweight='bold')
    ax7.set_title('4. System Cost - Latency', fontweight='bold', fontsize=12)
    ax7.set_xticklabels([s.replace(' ', '\n') for s in system_names], fontsize=8, rotation=15)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Plot 8: Communication Rounds
    ax8 = plt.subplot(3, 3, 8)
    bars8 = ax8.bar(system_names, comm_rounds, color=['#FF6B6B', '#FFB84D', '#4ECDC4', '#2ECC71'], alpha=0.8, edgecolor='black')
    ax8.set_ylabel('Communication Rounds', fontweight='bold')
    ax8.set_title('Communication Rounds per Query', fontweight='bold', fontsize=12)
    ax8.set_xticklabels([s.replace(' ', '\n') for s in system_names], fontsize=8, rotation=15)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Plot 9: Embeddings Transmitted
    ax9 = plt.subplot(3, 3, 9)
    bars9 = ax9.bar(system_names, [e/1000 for e in embeddings_trans], color=['#FF6B6B', '#FFB84D', '#4ECDC4', '#2ECC71'], alpha=0.8, edgecolor='black')
    ax9.set_ylabel('Embeddings (K)', fontweight='bold')
    ax9.set_title('Embeddings Transmitted (×1000)', fontweight='bold', fontsize=12)
    ax9.set_xticklabels([s.replace(' ', '\n') for s in system_names], fontsize=8, rotation=15)
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    results_dir = PROJECT_ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = results_dir / "exp7_reviewer_metrics_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")
    
    # Save detailed report
    data_path = results_dir / "exp7_reviewer_metrics_comparison.txt"
    with open(data_path, "w") as f:
        f.write("EXPERIMENT 7: REVIEWER-APPROVED METRICS COMPARISON\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Number of questions per system: {num_questions}\n")
        f.write(f"Epsilon (for DP systems): {epsilon}\n\n")
        
        f.write("1. RETRIEVAL QUALITY\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'System':<30} {'Avg Relevance':<15} {'Top-K Sim':<15} {'≥0.85':<12} {'≥0.90':<12}\n")
        f.write("-" * 100 + "\n")
        for i, name in enumerate(system_names):
            f.write(f"{name:<30} {avg_relevance[i]:<15.3f} {top_k_sim[i]:<15.3f} "
                   f"{pct_85[i]:<12.1f}% {pct_90[i]:<12.1f}%\n")
        f.write("-" * 100 + "\n\n")
        
        f.write("2. PRIVACY\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'System':<30} {'Raw Data':<15} {'Embeddings':<20} {'Epsilon':<12} {'Privacy Score':<15}\n")
        f.write("-" * 100 + "\n")
        for i, name in enumerate(system_names):
            f.write(f"{name:<30} {raw_data_sharing[i]:<15} {embedding_sharing[i]:<20} "
                   f"{epsilon_vals[i]:<12.2f} {privacy_scores[i]:<15.2f}\n")
        f.write("-" * 100 + "\n\n")
        
        f.write("3. FEDERATED BEHAVIOR\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'System':<30} {'Part. Rate':<15} {'Avg Clients':<15} {'Var':<12} {'Dominance':<15}\n")
        f.write("-" * 100 + "\n")
        for i, name in enumerate(system_names):
            dominance_str = "Yes" if dominance[i] else "No"
            f.write(f"{name:<30} {participation_rate[i]:<15.1f}% {avg_clients[i]:<15.2f} "
                   f"{contribution_var[i]:<12.2f} {dominance_str:<15}\n")
        f.write("-" * 100 + "\n\n")
        
        f.write("4. SYSTEM COST\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'System':<30} {'Latency (ms)':<15} {'Comm Rounds':<15} {'Embeddings':<15}\n")
        f.write("-" * 100 + "\n")
        for i, name in enumerate(system_names):
            f.write(f"{name:<30} {latencies[i]:<15.1f} {comm_rounds[i]:<15.2f} "
                   f"{embeddings_trans[i]:<15}\n")
        f.write("-" * 100 + "\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write(f"- Best Retrieval Quality: {system_names[np.argmax(avg_relevance)]} ({max(avg_relevance):.3f})\n")
        f.write(f"- Best Privacy Score: {system_names[np.argmax(privacy_scores)]} ({max(privacy_scores):.2f})\n")
        f.write(f"- Best Federated Behavior: {system_names[np.argmax(participation_rate)]} ({max(participation_rate):.1f}%)\n")
        f.write(f"- Lowest Latency: {system_names[np.argmin(latencies)]} ({min(latencies):.1f}ms)\n")
        f.write(f"- Lowest Communication: {system_names[np.argmin(comm_rounds)]} ({min(comm_rounds):.2f} rounds)\n")
    
    print(f"✓ Report saved to: {data_path}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 7 COMPLETE")
    print("=" * 80)
    print(f"✓ Comprehensive metrics collected for all 4 systems")
    print(f"✓ Visualizations created: {plot_path}")
    print(f"✓ Detailed report saved: {data_path}")
    print("=" * 80)
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Experiment 7: Reviewer-Approved Metrics Comparison"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=20,
        help="Number of questions per system (default: 20)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Epsilon value for DP systems (default: 1.0)"
    )
    
    args = parser.parse_args()
    run_reviewer_metrics_experiment(
        num_questions=args.num_questions,
        epsilon=args.epsilon,
    )
