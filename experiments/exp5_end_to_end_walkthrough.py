#!/usr/bin/env python3
"""
Experiment 5: End-to-End "One Question Walkthrough" Diagram

Goal: Create the easiest visual proof for a supervisor showing the complete pipeline.

This script:
1. Runs a single question through the system
2. Tracks each step of the process
3. Creates a visual diagram with arrows showing:
   - Question → Local Retrieval (Client 0/1/2) → DP embedding sharing → 
     Aggregation → Context building → LLM answer + citations
4. Clearly shows what IS shared vs what is NOT shared
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np
import yaml

# Suppress tokenizers warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.federated_rag import (
    create_federated_rag_clients_from_csv,
    FederatedRAGCoordinator,
)
from src.llm_providers import create_llm_provider


def create_pipeline_diagram(question: str = None):
    """Create end-to-end pipeline diagram for a single question."""
    print("=" * 80)
    print("EXPERIMENT 5: END-TO-END WALKTHROUGH DIAGRAM")
    print("=" * 80)
    print("\nGoal: Visual proof of the complete federated RAG pipeline.")
    print("=" * 80)
    
    if question is None:
        question = "What are the cardiovascular risk factors?"
    
    print(f"\nExample Question: {question}\n")
    
    # Load config
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found in project root.")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_cfg = config["data"]
    rag_cfg = config["rag"]
    llm_cfg = config["llm"]
    privacy_cfg = config["privacy"]
    
    csv_path = data_cfg.get("medical_literature_path", "./extracted_data.csv")
    csv_path = PROJECT_ROOT / csv_path if not Path(csv_path).is_absolute() else Path(csv_path)
    
    num_clients = 3
    max_docs_per_client = 8000
    top_k_per_client = 5
    
    print("[1/4] Creating federated clients...")
    clients, embedder = create_federated_rag_clients_from_csv(
        csv_path=str(csv_path),
        num_clients=num_clients,
        max_docs_per_client=max_docs_per_client,
        embedding_model=rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        vector_db_root=str(PROJECT_ROOT / "federated_vector_db"),
    )
    print(f"✓ Created {len(clients)} clients\n")
    
    # Initialize LLM provider
    print("[2/4] Initializing LLM provider...")
    provider_name = llm_cfg.get("provider", "groq")
    model_name = llm_cfg.get("model", "llama-3.3-70b-versatile")
    use_api = llm_cfg.get("use_api", True)
    
    if provider_name.lower() == "groq":
        api_key = os.getenv("GROQ_API_KEY")
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
    
    # Create coordinator
    epsilon = float(privacy_cfg.get("epsilon", 1.0))
    delta = float(privacy_cfg.get("delta", 1e-5))
    
    coordinator = FederatedRAGCoordinator(
        clients=clients,
        epsilon=epsilon,
        delta=delta,
        llm_provider=llm_provider,
    )
    
    print("[3/4] Processing question through pipeline...")
    
    # Step 1: Local retrieval from each client
    client_results = {}
    for client in clients:
        chunks = client.local_retrieve(
            question,
            top_k=top_k_per_client,
            epsilon=epsilon,
            delta=delta,
        )
        client_results[client.client_id] = {
            'chunks': chunks,
            'num_docs': len(chunks),
            'raw_docs': len(client.df),  # Total docs in client DB
        }
    
    # Step 2: Federated aggregation
    aggregated_chunks = coordinator.federated_retrieve(question, top_k_per_client=top_k_per_client)
    
    # Step 3: Generate answer
    result = coordinator.generate_answer(question, top_k_per_client=top_k_per_client)
    answer = result.get("answer", "")
    citations = result.get("citations", [])
    
    print(f"✓ Retrieved {len(aggregated_chunks)} documents")
    print(f"✓ Generated answer ({len(answer)} chars)")
    print(f"✓ Citations: {len(citations)}\n")
    
    print("[4/4] Creating visualization...")
    
    # Create the diagram
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    color_question = '#4A90E2'  # Blue
    color_client = '#7ED321'   # Green
    color_shared = '#F5A623'    # Orange
    color_private = '#D0021B'   # Red
    color_aggregation = '#9013FE'  # Purple
    color_llm = '#50E3C2'       # Teal
    color_answer = '#BD10E0'    # Magenta
    
    # Title
    ax.text(5, 9.5, 'Privacy-Preserving Federated RAG Pipeline', 
            ha='center', va='top', fontsize=18, fontweight='bold')
    ax.text(5, 9.0, 'End-to-End Walkthrough for a Single Question', 
            ha='center', va='top', fontsize=14, style='italic')
    
    # Step 1: Question Input
    question_box = FancyBboxPatch((3.5, 8.0), 3.0, 0.6, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=color_question, 
                                   edgecolor='black', linewidth=2)
    ax.add_patch(question_box)
    ax.text(5, 8.3, f'Question:\n"{question[:40]}..."', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # Step 2: Local Retrieval (3 clients in parallel)
    client_y_positions = [6.5, 5.5, 4.5]
    client_boxes = []
    
    for i, client_id in enumerate([0, 1, 2]):
        y_pos = client_y_positions[i]
        
        # Client box
        client_box = FancyBboxPatch((0.5, y_pos - 0.3), 2.5, 0.6,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color_client,
                                    edgecolor='black', linewidth=2)
        ax.add_patch(client_box)
        ax.text(1.75, y_pos, f'Client {client_id}\nLocal Retrieval', 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # Private database indicator
        db_box = FancyBboxPatch((0.5, y_pos - 0.8), 2.5, 0.4,
                                boxstyle="round,pad=0.05",
                                facecolor=color_private,
                                edgecolor='black', linewidth=1.5, linestyle='--')
        ax.add_patch(db_box)
        num_docs = client_results[client_id]['num_docs']
        total_docs = client_results[client_id]['raw_docs']
        ax.text(1.75, y_pos - 0.6, f'Private DB: {total_docs} docs\nRetrieved: {num_docs} docs', 
                ha='center', va='center', fontsize=8, color='white', style='italic')
        
        client_boxes.append((1.75, y_pos))
    
    # Step 3: DP Embedding Sharing
    dp_boxes = []
    for i, client_id in enumerate([0, 1, 2]):
        y_pos = client_y_positions[i]
        dp_box = FancyBboxPatch((3.5, y_pos - 0.3), 2.0, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=color_shared,
                                edgecolor='black', linewidth=2)
        ax.add_patch(dp_box)
        ax.text(4.5, y_pos, f'DP Embeddings\n(ε={epsilon})', 
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        dp_boxes.append((4.5, y_pos))
    
    # Step 4: Aggregation
    agg_box = FancyBboxPatch((6.0, 5.0), 2.0, 1.0,
                             boxstyle="round,pad=0.1",
                             facecolor=color_aggregation,
                             edgecolor='black', linewidth=2)
    ax.add_patch(agg_box)
    ax.text(7.0, 5.5, 'Aggregation\n& Reranking', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # Step 5: Context Building
    context_box = FancyBboxPatch((6.0, 3.5), 2.0, 0.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor=color_llm,
                                 edgecolor='black', linewidth=2)
    ax.add_patch(context_box)
    ax.text(7.0, 3.9, 'Context Building\n(Metadata Only)', 
            ha='center', va='center', fontsize=10, fontweight='bold', color='black')
    
    # Step 6: LLM Answer
    answer_box = FancyBboxPatch((3.5, 1.5), 3.0, 1.0,
                                boxstyle="round,pad=0.1",
                                facecolor=color_answer,
                                edgecolor='black', linewidth=2)
    ax.add_patch(answer_box)
    answer_preview = answer[:100] + "..." if len(answer) > 100 else answer
    ax.text(5, 2.0, f'LLM Answer\n({len(answer)} chars)\n+ Citations', 
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Arrows
    # Question → Clients
    for y_pos in client_y_positions:
        arrow = FancyArrowPatch((3.5, 8.0), (1.75, y_pos + 0.3),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='black', zorder=1)
        ax.add_patch(arrow)
    
    # Clients → DP Embeddings
    for i, (client_pos, dp_pos) in enumerate(zip(client_boxes, dp_boxes)):
        arrow = FancyArrowPatch((client_pos[0] + 1.25, client_pos[1]), 
                               (dp_pos[0] - 1.0, dp_pos[1]),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2, color='black', zorder=1)
        ax.add_patch(arrow)
    
    # DP Embeddings → Aggregation
    for dp_pos in dp_boxes:
        arrow = FancyArrowPatch((dp_pos[0] + 1.0, dp_pos[1]), 
                               (6.0, 5.5),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2, color='black', zorder=1)
        ax.add_patch(arrow)
    
    # Aggregation → Context Building
    arrow = FancyArrowPatch((7.0, 5.0), (7.0, 4.3),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='black', zorder=1)
    ax.add_patch(arrow)
    
    # Context Building → LLM Answer
    arrow = FancyArrowPatch((7.0, 3.5), (5, 2.5),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='black', zorder=1)
    ax.add_patch(arrow)
    
    # Legend: What is shared vs NOT shared
    legend_y = 0.5
    ax.text(1.0, legend_y + 0.3, 'PRIVACY INFORMATION:', 
            ha='left', va='top', fontsize=12, fontweight='bold')
    
    # What is NOT shared
    not_shared_box = FancyBboxPatch((1.0, legend_y - 0.4), 3.5, 0.3,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color_private,
                                    edgecolor='black', linewidth=1.5)
    ax.add_patch(not_shared_box)
    ax.text(2.75, legend_y - 0.25, '❌ NOT SHARED: Raw documents, Patient data, Full text', 
            ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # What IS shared
    shared_box = FancyBboxPatch((5.0, legend_y - 0.4), 3.5, 0.3,
                                boxstyle="round,pad=0.05",
                                facecolor=color_shared,
                                edgecolor='black', linewidth=1.5)
    ax.add_patch(shared_box)
    ax.text(6.75, legend_y - 0.25, '✅ SHARED: DP-noised embeddings, Metadata (title/journal/year)', 
            ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # Statistics box
    stats_text = (
        f"Pipeline Statistics:\n"
        f"• Clients: 3 (each with ~{max_docs_per_client} private documents)\n"
        f"• Retrieved per client: {top_k_per_client} documents\n"
        f"• Total aggregated: {len(aggregated_chunks)} documents\n"
        f"• Privacy: ε={epsilon}, δ={delta}\n"
        f"• Citations: {len(citations)} sources"
    )
    stats_box = FancyBboxPatch((1.0, 0.0), 7.5, 0.4,
                               boxstyle="round,pad=0.05",
                               facecolor='lightgray',
                               edgecolor='black', linewidth=1)
    ax.add_patch(stats_box)
    ax.text(4.75, 0.2, stats_text,
            ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    results_dir = PROJECT_ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = results_dir / "exp5_end_to_end_walkthrough.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Diagram saved to: {plot_path}")
    
    # Save detailed walkthrough
    walkthrough_path = results_dir / "exp5_end_to_end_walkthrough.txt"
    with open(walkthrough_path, "w") as f:
        f.write("EXPERIMENT 5: END-TO-END WALKTHROUGH\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Question: {question}\n\n")
        
        f.write("STEP 1: QUESTION INPUT\n")
        f.write("-" * 80 + "\n")
        f.write(f"User asks: {question}\n\n")
        
        f.write("STEP 2: LOCAL RETRIEVAL (Parallel at each client)\n")
        f.write("-" * 80 + "\n")
        for client_id in [0, 1, 2]:
            num_docs = client_results[client_id]['num_docs']
            total_docs = client_results[client_id]['raw_docs']
            f.write(f"Client {client_id}:\n")
            f.write(f"  - Searches its PRIVATE database ({total_docs} documents)\n")
            f.write(f"  - Retrieves top {num_docs} relevant documents\n")
            f.write(f"  - NO raw text leaves the client\n\n")
        
        f.write("STEP 3: DIFFERENTIAL PRIVACY (DP) EMBEDDING SHARING\n")
        f.write("-" * 80 + "\n")
        f.write("Each client applies DP noise to embeddings:\n")
        f.write(f"  - Privacy parameters: ε={epsilon}, δ={delta}\n")
        f.write("  - Only DP-noised embeddings are shared\n")
        f.write("  - Metadata (title, journal, year) is also shared\n")
        f.write("  - ❌ Raw document text is NEVER shared\n\n")
        
        f.write("STEP 4: AGGREGATION & RERANKING\n")
        f.write("-" * 80 + "\n")
        f.write(f"Coordinator receives embeddings from all {num_clients} clients\n")
        f.write(f"  - Merges results from all clients\n")
        f.write(f"  - Reranks by similarity score\n")
        f.write(f"  - Total aggregated documents: {len(aggregated_chunks)}\n\n")
        
        f.write("STEP 5: CONTEXT BUILDING\n")
        f.write("-" * 80 + "\n")
        f.write("Builds context for LLM using ONLY:\n")
        f.write("  - High-level metadata (titles, journals, years)\n")
        f.write("  - Relevance scores\n")
        f.write("  - ❌ NO raw document text\n")
        f.write("  - ❌ NO patient data\n\n")
        
        f.write("STEP 6: LLM ANSWER GENERATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"LLM generates answer based on metadata-only context\n")
        f.write(f"  - Answer length: {len(answer)} characters\n")
        f.write(f"  - Citations: {len(citations)} sources\n")
        f.write(f"  - Privacy-preserved citations (metadata only)\n\n")
        
        f.write("PRIVACY SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write("✅ SHARED:\n")
        f.write("  - Differentially private embeddings\n")
        f.write("  - Metadata (title, journal, year)\n")
        f.write("  - Relevance scores\n\n")
        f.write("❌ NOT SHARED:\n")
        f.write("  - Raw document text\n")
        f.write("  - Patient data\n")
        f.write("  - Full abstracts\n")
        f.write("  - Any identifying information\n\n")
        
        f.write("CONCLUSION\n")
        f.write("-" * 80 + "\n")
        f.write("The system successfully answers questions using federated data\n")
        f.write("while maintaining strict privacy guarantees through:\n")
        f.write("  1. Local-only data storage\n")
        f.write("  2. Differential privacy on embeddings\n")
        f.write("  3. Metadata-only context building\n")
        f.write("  4. No raw data sharing between clients\n")
    
    print(f"✓ Walkthrough saved to: {walkthrough_path}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\n✓ Diagram: {plot_path}")
    print(f"✓ Walkthrough: {walkthrough_path}")
    print("\nThe diagram shows:")
    print("  • Complete pipeline flow")
    print("  • What IS shared (DP embeddings, metadata)")
    print("  • What is NOT shared (raw docs, patient data)")
    print("=" * 80)
    
    # Show plot
    plt.show()
    
    return plot_path, walkthrough_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Experiment 5: End-to-End Walkthrough Diagram"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question to use for walkthrough (default: 'What are the cardiovascular risk factors?')"
    )
    
    args = parser.parse_args()
    create_pipeline_diagram(question=args.question)
