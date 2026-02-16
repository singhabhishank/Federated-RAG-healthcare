#!/usr/bin/env python3
"""
Experiment 2: Retrieval Relevance Plot (Quality curve)

Goal: Show the retrieval quality is consistently high across multiple questions.

This script:
1. Runs 50 medical questions through the federated RAG system
2. Calculates retrieval relevance score for each question
3. Creates a bar chart/line plot showing relevance per question
4. Adds a horizontal threshold line (0.85)
5. Shows statistics and saves the plot
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np

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
    evaluate_retrieval_relevance,
)
import yaml


def run_relevance_experiment(num_questions: int = 50):
    """Run retrieval relevance experiment across multiple questions."""
    print("=" * 80)
    print("EXPERIMENT 2: RETRIEVAL RELEVANCE PLOT")
    print("=" * 80)
    print(f"\nGoal: Show retrieval quality is consistently high across {num_questions} questions.")
    print("=" * 80)
    
    # Load config
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found in project root.")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_cfg = config["data"]
    rag_cfg = config["rag"]
    privacy_cfg = config["privacy"]
    
    csv_path = data_cfg.get("medical_literature_path", "./extracted_data.csv")
    csv_path = PROJECT_ROOT / csv_path if not Path(csv_path).is_absolute() else Path(csv_path)
    
    num_clients = 3
    max_docs_per_client = 8000
    top_k_per_client = 5
    
    print("\n[1/4] Creating federated clients...")
    clients, embedder = create_federated_rag_clients_from_csv(
        csv_path=str(csv_path),
        num_clients=num_clients,
        max_docs_per_client=max_docs_per_client,
        embedding_model=rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        vector_db_root=str(PROJECT_ROOT / "federated_vector_db"),
    )
    print(f"✓ Created {len(clients)} clients\n")
    
    # Create coordinator (no LLM needed for retrieval-only)
    epsilon = float(privacy_cfg.get("epsilon", 1.0))
    delta = float(privacy_cfg.get("delta", 1e-5))
    
    coordinator = FederatedRAGCoordinator(
        clients=clients,
        epsilon=epsilon,
        delta=delta,
        llm_provider=None,  # Retrieval only
    )
    
    # Get questions
    questions = MEDICAL_QUESTIONS[:num_questions]
    print(f"[2/4] Evaluating {len(questions)} questions...\n")
    
    # Evaluate each question
    relevance_scores = []
    question_labels = []
    
    for idx, question in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] {question[:60]}...", end=" ")
        
        try:
            # Retrieve documents
            chunks = coordinator.federated_retrieve(
                question,
                top_k_per_client=top_k_per_client,
            )
            
            # Get citations
            citations = [c.metadata for c in chunks]
            
            # Calculate relevance score
            relevance = evaluate_retrieval_relevance(
                question,
                citations,
                embedder=embedder
            )
            
            relevance_scores.append(relevance)
            question_labels.append(f"Q{idx}")
            
            print(f"Relevance: {relevance:.3f}")
            
        except Exception as e:
            print(f"Error: {e}")
            relevance_scores.append(0.0)
            question_labels.append(f"Q{idx}")
    
    print(f"\n[3/4] Calculating statistics...")
    
    # Calculate statistics
    avg_relevance = np.mean(relevance_scores)
    median_relevance = np.median(relevance_scores)
    min_relevance = np.min(relevance_scores)
    max_relevance = np.max(relevance_scores)
    std_relevance = np.std(relevance_scores)
    
    # Count questions above thresholds
    above_085 = sum(1 for r in relevance_scores if r >= 0.85)
    above_090 = sum(1 for r in relevance_scores if r >= 0.90)
    above_095 = sum(1 for r in relevance_scores if r >= 0.95)
    
    print(f"\nStatistics:")
    print(f"  Average Relevance: {avg_relevance:.3f}")
    print(f"  Median Relevance: {median_relevance:.3f}")
    print(f"  Min Relevance: {min_relevance:.3f}")
    print(f"  Max Relevance: {max_relevance:.3f}")
    print(f"  Std Deviation: {std_relevance:.3f}")
    print(f"\nQuestions above threshold:")
    print(f"  ≥ 0.85: {above_085}/{len(questions)} ({above_085/len(questions)*100:.1f}%)")
    print(f"  ≥ 0.90: {above_090}/{len(questions)} ({above_090/len(questions)*100:.1f}%)")
    print(f"  ≥ 0.95: {above_095}/{len(questions)} ({above_095/len(questions)*100:.1f}%)")
    
    print(f"\n[4/4] Creating visualization...")
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Bar chart
    colors = ['green' if r >= 0.90 else 'orange' if r >= 0.85 else 'red' for r in relevance_scores]
    bars = ax1.bar(range(len(questions)), relevance_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=0.85, color='blue', linestyle='--', linewidth=2, label='Target Threshold (0.85)')
    ax1.axhline(y=0.90, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='High Quality (0.90)')
    ax1.set_xlabel('Question Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Retrieval Relevance Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'Retrieval Relevance Scores Across {len(questions)} Questions\n(Average: {avg_relevance:.3f})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(range(0, len(questions), max(1, len(questions)//20)))
    ax1.set_xticklabels([question_labels[i] for i in range(0, len(questions), max(1, len(questions)//20))], 
                        rotation=45, ha='right')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(loc='upper right')
    
    # Add value labels on bars (every 5th bar to avoid clutter)
    for i in range(0, len(bars), max(1, len(questions)//20)):
        height = bars[i].get_height()
        ax1.text(bars[i].get_x() + bars[i].get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Line plot with moving average
    ax2.plot(range(len(questions)), relevance_scores, 'o-', color='steelblue', 
             markersize=4, linewidth=1.5, alpha=0.7, label='Relevance Score')
    
    # Moving average (window of 5)
    if len(relevance_scores) >= 5:
        window = 5
        moving_avg = np.convolve(relevance_scores, np.ones(window)/window, mode='valid')
        moving_avg_x = range(window-1, len(relevance_scores))
        ax2.plot(moving_avg_x, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window})')
    
    ax2.axhline(y=0.85, color='blue', linestyle='--', linewidth=2, label='Target Threshold (0.85)')
    ax2.axhline(y=0.90, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='High Quality (0.90)')
    ax2.axhline(y=avg_relevance, color='orange', linestyle=':', linewidth=2, label=f'Average ({avg_relevance:.3f})')
    ax2.set_xlabel('Question Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Retrieval Relevance Score', fontsize=12, fontweight='bold')
    ax2.set_title('Retrieval Relevance Trend', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(0, len(questions), max(1, len(questions)//20)))
    ax2.set_xticklabels([question_labels[i] for i in range(0, len(questions), max(1, len(questions)//20))], 
                        rotation=45, ha='right')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save plot
    results_dir = PROJECT_ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = results_dir / "exp2_retrieval_relevance_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")
    
    # Save data
    data_path = results_dir / "exp2_retrieval_relevance_data.txt"
    with open(data_path, "w") as f:
        f.write("EXPERIMENT 2: RETRIEVAL RELEVANCE PLOT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Questions: {len(questions)}\n")
        f.write(f"Average Relevance: {avg_relevance:.3f}\n")
        f.write(f"Median Relevance: {median_relevance:.3f}\n")
        f.write(f"Min Relevance: {min_relevance:.3f}\n")
        f.write(f"Max Relevance: {max_relevance:.3f}\n")
        f.write(f"Std Deviation: {std_relevance:.3f}\n\n")
        f.write(f"Questions ≥ 0.85: {above_085}/{len(questions)} ({above_085/len(questions)*100:.1f}%)\n")
        f.write(f"Questions ≥ 0.90: {above_090}/{len(questions)} ({above_090/len(questions)*100:.1f}%)\n")
        f.write(f"Questions ≥ 0.95: {above_095}/{len(questions)} ({above_095/len(questions)*100:.1f}%)\n\n")
        f.write("\nPer-Question Scores:\n")
        f.write("-" * 80 + "\n")
        for i, (q, score) in enumerate(zip(questions, relevance_scores), 1):
            f.write(f"Q{i:2d}: {score:.3f} - {q}\n")
    
    print(f"✓ Data saved to: {data_path}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    
    # Show plot
    plt.show()
    
    return relevance_scores, avg_relevance


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Experiment 2: Retrieval Relevance Plot"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=50,
        help="Number of questions to evaluate (default: 50)"
    )
    
    args = parser.parse_args()
    run_relevance_experiment(num_questions=args.num_questions)
