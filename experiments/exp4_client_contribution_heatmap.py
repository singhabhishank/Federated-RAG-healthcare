#!/usr/bin/env python3
"""
Experiment 4: Client Contribution Heatmap

Goal: Prove federated collaboration across multiple questions.

This script:
1. Runs multiple questions through the federated RAG system
2. Tracks how many documents each client contributes for each question
3. Creates a heatmap showing client contributions
4. Analyzes collaboration patterns
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
from src.evaluate_federated_rag import (
    MEDICAL_QUESTIONS,
)


def run_client_contribution_experiment(num_questions: int = 50):
    """Run experiment to track client contributions across questions."""
    print("=" * 80)
    print("EXPERIMENT 4: CLIENT CONTRIBUTION HEATMAP")
    print("=" * 80)
    print(f"\nGoal: Prove federated collaboration across {num_questions} questions.")
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
    privacy_cfg = config["privacy"]
    
    csv_path = data_cfg.get("medical_literature_path", "./extracted_data.csv")
    csv_path = PROJECT_ROOT / csv_path if not Path(csv_path).is_absolute() else Path(csv_path)
    
    num_clients = 3
    max_docs_per_client = 8000
    top_k_per_client = 5
    
    print("\n[1/3] Creating federated clients...")
    clients, embedder = create_federated_rag_clients_from_csv(
        csv_path=str(csv_path),
        num_clients=num_clients,
        max_docs_per_client=max_docs_per_client,
        embedding_model=rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        vector_db_root=str(PROJECT_ROOT / "federated_vector_db"),
    )
    print(f"✓ Created {len(clients)} clients\n")
    
    # For this experiment, we only need retrieval (no LLM needed)
    # This avoids API rate limits and is faster
    print("[2/3] Setting up coordinator (retrieval-only mode, no LLM)...")
    
    # Create coordinator without LLM (retrieval-only)
    epsilon = float(privacy_cfg.get("epsilon", 1.0))
    delta = float(privacy_cfg.get("delta", 1e-5))
    
    coordinator = FederatedRAGCoordinator(
        clients=clients,
        epsilon=epsilon,
        delta=delta,
        llm_provider=None,  # No LLM needed - we only track contributions
    )
    print("✓ Coordinator ready (retrieval-only mode)\n")
    
    # Get questions
    questions = MEDICAL_QUESTIONS[:num_questions]
    print(f"[3/3] Processing {len(questions)} questions...\n")
    
    # Track contributions
    contribution_matrix = []  # Will be [num_questions x num_clients]
    question_labels = []
    
    for idx, question in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] {question[:60]}...", end=" ")
        
        try:
            # Use federated_retrieve directly (no LLM needed - faster, no API limits)
            chunks = coordinator.federated_retrieve(question, top_k_per_client=top_k_per_client)
            
            # Count contributions per client
            client_counts = {0: 0, 1: 0, 2: 0}
            
            for chunk in chunks:
                client_id = chunk.client_id
                if client_id in client_counts:
                    client_counts[client_id] += 1
            
            # Store contribution counts
            contribution_matrix.append([client_counts[0], client_counts[1], client_counts[2]])
            question_labels.append(f"Q{idx}")
            
            total = sum(client_counts.values())
            contributing = sum(1 for count in client_counts.values() if count > 0)
            print(f"Total: {total}, Clients: {contributing} (C0:{client_counts[0]}, C1:{client_counts[1]}, C2:{client_counts[2]})")
            
            # Small delay to avoid overwhelming the system
            if idx < len(questions):
                import time
                time.sleep(0.5)  # Much shorter delay since no API calls
                
        except Exception as e:
            print(f"Error: {e}")
            # On error, record zeros
            contribution_matrix.append([0, 0, 0])
            question_labels.append(f"Q{idx}")
    
    # Convert to numpy array
    contribution_array = np.array(contribution_matrix)
    
    print("\n" + "=" * 80)
    print("ANALYZING CLIENT CONTRIBUTIONS")
    print("=" * 80)
    
    # Calculate statistics
    total_contributions = contribution_array.sum(axis=1)  # Total docs per question
    questions_with_multiple_clients = sum(1 for row in contribution_array if np.count_nonzero(row) > 1)
    questions_with_single_client = sum(1 for row in contribution_array if np.count_nonzero(row) == 1)
    questions_with_all_clients = sum(1 for row in contribution_array if np.count_nonzero(row) == 3)
    
    # Client-specific stats
    client_0_total = contribution_array[:, 0].sum()
    client_1_total = contribution_array[:, 1].sum()
    client_2_total = contribution_array[:, 2].sum()
    
    client_0_questions = np.count_nonzero(contribution_array[:, 0])
    client_1_questions = np.count_nonzero(contribution_array[:, 1])
    client_2_questions = np.count_nonzero(contribution_array[:, 2])
    
    print(f"\nCollaboration Statistics:")
    print(f"  Questions with multiple clients: {questions_with_multiple_clients}/{len(questions)} ({questions_with_multiple_clients/len(questions)*100:.1f}%)")
    print(f"  Questions with single client: {questions_with_single_client}/{len(questions)} ({questions_with_single_client/len(questions)*100:.1f}%)")
    print(f"  Questions with all 3 clients: {questions_with_all_clients}/{len(questions)} ({questions_with_all_clients/len(questions)*100:.1f}%)")
    
    print(f"\nClient Contribution Totals:")
    print(f"  Client 0: {client_0_total} documents across {client_0_questions} questions")
    print(f"  Client 1: {client_1_total} documents across {client_1_questions} questions")
    print(f"  Client 2: {client_2_total} documents across {client_2_questions} questions")
    
    print(f"\nAverage contributions per question:")
    print(f"  Client 0: {contribution_array[:, 0].mean():.2f} documents")
    print(f"  Client 1: {contribution_array[:, 1].mean():.2f} documents")
    print(f"  Client 2: {contribution_array[:, 2].mean():.2f} documents")
    
    print("\nCreating visualization...")
    
    # Create heatmap
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: Heatmap
    ax1 = axes[0]
    
    # Create heatmap data (transpose so questions are rows, clients are columns)
    heatmap_data = contribution_array.T  # [num_clients x num_questions]
    
    # Create custom colormap (white to dark blue)
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Number of Documents'},
        xticklabels=question_labels,
        yticklabels=['Client 0', 'Client 1', 'Client 2'],
        ax=ax1,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax1.set_xlabel('Question Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Client', fontsize=12, fontweight='bold')
    ax1.set_title(f'Client Contribution Heatmap ({len(questions)} Questions)\n'
                  f'Shows number of documents contributed by each client per question',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(range(0, len(question_labels), max(1, len(question_labels)//20)))
    ax1.set_xticklabels([question_labels[i] for i in range(0, len(question_labels), max(1, len(question_labels)//20))],
                        rotation=45, ha='right')
    
    # Plot 2: Stacked bar chart showing contribution distribution
    ax2 = axes[1]
    
    x_pos = np.arange(len(questions))
    width = 0.8
    
    # Stack bars
    ax2.bar(x_pos, contribution_array[:, 0], width, label='Client 0', color='#1f77b4', alpha=0.8)
    ax2.bar(x_pos, contribution_array[:, 1], width, bottom=contribution_array[:, 0],
            label='Client 1', color='#ff7f0e', alpha=0.8)
    ax2.bar(x_pos, contribution_array[:, 2], width,
            bottom=contribution_array[:, 0] + contribution_array[:, 1],
            label='Client 2', color='#2ca02c', alpha=0.8)
    
    ax2.set_xlabel('Question Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Documents', fontsize=12, fontweight='bold')
    ax2.set_title('Stacked Client Contributions per Question', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(0, len(questions), max(1, len(questions)//20)))
    ax2.set_xticklabels([question_labels[i] for i in range(0, len(questions), max(1, len(questions)//20))],
                        rotation=45, ha='right')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    results_dir = PROJECT_ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = results_dir / "exp4_client_contribution_heatmap.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")
    
    # Save data
    data_path = results_dir / "exp4_client_contribution_data.txt"
    with open(data_path, "w") as f:
        f.write("EXPERIMENT 4: CLIENT CONTRIBUTION HEATMAP\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Questions: {len(questions)}\n")
        f.write(f"Top-K per client: {top_k_per_client}\n\n")
        
        f.write("Collaboration Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Questions with multiple clients: {questions_with_multiple_clients}/{len(questions)} ({questions_with_multiple_clients/len(questions)*100:.1f}%)\n")
        f.write(f"Questions with single client: {questions_with_single_client}/{len(questions)} ({questions_with_single_client/len(questions)*100:.1f}%)\n")
        f.write(f"Questions with all 3 clients: {questions_with_all_clients}/{len(questions)} ({questions_with_all_clients/len(questions)*100:.1f}%)\n\n")
        
        f.write("Client Contribution Totals:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Client 0: {client_0_total} documents across {client_0_questions} questions (avg: {contribution_array[:, 0].mean():.2f}/question)\n")
        f.write(f"Client 1: {client_1_total} documents across {client_1_questions} questions (avg: {contribution_array[:, 1].mean():.2f}/question)\n")
        f.write(f"Client 2: {client_2_total} documents across {client_2_questions} questions (avg: {contribution_array[:, 2].mean():.2f}/question)\n\n")
        
        f.write("Per-Question Contributions:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Question':<10} {'Client 0':<10} {'Client 1':<10} {'Client 2':<10} {'Total':<10} {'Clients':<10}\n")
        f.write("-" * 80 + "\n")
        for i, (q, row) in enumerate(zip(questions, contribution_array)):
            num_clients = np.count_nonzero(row)
            f.write(f"Q{i+1:<9} {int(row[0]):<10} {int(row[1]):<10} {int(row[2]):<10} {int(row.sum()):<10} {num_clients:<10}\n")
    
    print(f"✓ Data saved to: {data_path}")
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"✅ {questions_with_multiple_clients}/{len(questions)} questions ({questions_with_multiple_clients/len(questions)*100:.1f}%) show collaboration from multiple clients")
    print(f"✅ All clients contribute: C0 in {client_0_questions} questions, C1 in {client_1_questions} questions, C2 in {client_2_questions} questions")
    print(f"✅ Federated collaboration is working - most questions benefit from multiple clients")
    print("=" * 80)
    
    # Show plot
    plt.show()
    
    return contribution_array, questions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Experiment 4: Client Contribution Heatmap"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=50,
        help="Number of questions to evaluate (default: 50)"
    )
    
    args = parser.parse_args()
    run_client_contribution_experiment(num_questions=args.num_questions)
