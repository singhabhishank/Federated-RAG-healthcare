#!/usr/bin/env python3
"""
Experiment 10: Confusion Matrices

Goal: Generate confusion matrices comparing the 4 systems across different metrics.

Confusion matrices created:
1. Retrieval Relevance Confusion Matrix (High vs Low relevance)
2. Answer Quality Confusion Matrix (High vs Low quality)
3. Medical Accuracy Confusion Matrix (High vs Low accuracy)

Each confusion matrix compares:
- Baseline (Centralized RAG) as "Expected/Actual"
- Other systems as "Predicted"
- Shows True Positives, False Positives, True Negatives, False Negatives

This script:
1. Runs evaluations on all 4 systems
2. Classifies results as High/Low based on thresholds
3. Creates confusion matrices comparing each system to baseline
4. Generates comprehensive visualizations
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import yaml
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sentence_transformers import SentenceTransformer
import time

# Suppress tokenizers warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

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
)
from src.llm_providers import create_llm_provider

# Import system coordinators from exp6
from experiments.exp6_system_comparison import (
    CentralizedRAGCoordinator,
    FederatedRAGNoDPCoordinator,
    SecureAggregationCoordinator,
)


def classify_performance(score: float, threshold: float) -> str:
    """Classify performance as 'High' or 'Low' based on threshold."""
    return "High" if score >= threshold else "Low"

def classify_performance_three_way(score: float, low_threshold: float, high_threshold: float) -> str:
    """Classify performance as 'Low', 'Medium', or 'High' based on thresholds."""
    if score >= high_threshold:
        return "High"
    elif score >= low_threshold:
        return "Medium"
    else:
        return "Low"


def create_confusion_matrix_plot(
    y_true: List[str],
    y_pred: List[str],
    title: str,
    labels: List[str] = None,
    ax=None,
) -> None:
    """Create a confusion matrix plot showing agreement between baseline and system."""
    if labels is None:
        labels = ["Low", "High"]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Check if we have any data
    if cm.sum() == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Normalize to get proportions (by row - what proportion of each baseline class)
    row_sums = cm.sum(axis=1)
    cm_normalized = cm.astype('float') / row_sums[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    # Create heatmap
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={'label': 'Proportion'},
        vmin=0,
        vmax=1,
    )
    
    ax.set_xlabel('System Classification', fontsize=12, fontweight='bold')
    ax.set_ylabel('Baseline Classification', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Add counts to cells (always show count if > 0, or proportion if count = 0 but we want to show it)
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = cm[i, j]
            proportion = cm_normalized[i, j]
            # Show count in gray below proportion
            if count > 0:
                ax.text(
                    j + 0.5,
                    i + 0.75,
                    f'\n({count})',
                    ha='center',
                    va='top',
                    fontsize=9,
                    color='gray',
                    weight='normal',
                )
            # If no count but we want to indicate it's a valid cell
            elif row_sums[i] > 0:  # Only show 0 if there's data in that row
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    '0',
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='lightgray',
                )


def run_confusion_matrix_experiment(
    num_questions: int = 30,
    delta: float = 1e-5,
    epsilon: float = 1.0,
    retrieval_only: bool = False,
):
    """Run confusion matrix experiment."""
    print("=" * 80)
    print("EXPERIMENT 10: CONFUSION MATRICES")
    print("=" * 80)
    print("\nGoal: Generate confusion matrices comparing 4 systems across metrics.")
    print(f"Number of questions per system: {num_questions}")
    print(f"Epsilon (for DP systems): {epsilon}")
    print(f"Retrieval-only mode: {retrieval_only}")
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
    
    # Load data (we'll load it again for centralized client, but function also loads it)
    print("\n[1/6] Loading data and creating clients...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Title", "Abstract"])
    df = df[df["Title"].str.strip() != ""]
    df = df[df["Abstract"].str.strip() != ""]
    
    print(f"Loaded {len(df)} cleaned medical articles")
    
    # Get embedding model name (will be loaded by create_federated_rag_clients_from_csv)
    embedding_model = rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize LLM provider (if not retrieval-only)
    llm_provider = None
    if not retrieval_only:
        print("\n[2/6] Initializing LLM provider...")
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
    else:
        print("\n[2/6] Skipping LLM provider (retrieval-only mode)\n")
    
    # Create federated clients
    print("[3/6] Creating federated clients...")
    num_clients = 3
    max_docs_per_client = 8000
    
    clients, embedder_from_func = create_federated_rag_clients_from_csv(
        csv_path=str(csv_path),
        num_clients=num_clients,
        max_docs_per_client=max_docs_per_client,
        embedding_model=embedding_model,
        vector_db_root=str(PROJECT_ROOT / "federated_vector_db"),
    )
    # Use the embedder from the function (it's the same model)
    embedder = embedder_from_func
    print(f"✓ Created {len(clients)} federated clients")
    
    # Create centralized client (all data in one client)
    print("[4/6] Creating centralized client...")
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    centralized_client = FederatedRAGClient(
        client_id=0,
        df=df_shuffled,
        embedder=embedder,
        vector_db_root=str(PROJECT_ROOT / "federated_vector_db"),
        collection_name="centralized_rag",
    )
    centralized_client.build_local_index()
    
    print(f"✓ Created centralized client with {centralized_client.collection.count()} documents")
    
    # Select questions
    questions = MEDICAL_QUESTIONS[:num_questions]
    print(f"\n[5/6] Running evaluations on {len(questions)} questions...")
    
    # Define systems
    systems = {
        "Centralized RAG": CentralizedRAGCoordinator(centralized_client, llm_provider),
        "Federated RAG (No DP)": FederatedRAGNoDPCoordinator(clients, llm_provider),
        "Federated RAG + DP": FederatedRAGCoordinator(clients, epsilon=epsilon, delta=delta, llm_provider=llm_provider),
        "Federated RAG + DP + SA": SecureAggregationCoordinator(clients, epsilon=epsilon, delta=delta, llm_provider=llm_provider),
    }
    
    # Run evaluations
    all_results = {}
    for system_name, coordinator in systems.items():
        print(f"\n  Evaluating: {system_name}...")
        results = run_evaluation(
            coordinator=coordinator,
            questions=questions,
            embedder=embedder,
        )
        all_results[system_name] = results
        
        # Add delay between systems to avoid rate limits
        if system_name != list(systems.keys())[-1]:
            time.sleep(5)
    
    print("\n[6/6] Generating confusion matrices...")
    
    # Define thresholds
    relevance_threshold = 0.85
    quality_threshold = 0.7
    accuracy_threshold = 0.7
    
    # Print actual score distributions for debugging
    print("\n" + "=" * 80)
    print("SCORE DISTRIBUTIONS (for debugging):")
    print("=" * 80)
    for system_name, results in all_results.items():
        relevance_scores = [r.retrieval_relevance_score for r in results if hasattr(r, 'retrieval_relevance_score')]
        if relevance_scores:
            print(f"\n{system_name}:")
            print(f"  Relevance scores: min={min(relevance_scores):.3f}, max={max(relevance_scores):.3f}, "
                  f"mean={np.mean(relevance_scores):.3f}, median={np.median(relevance_scores):.3f}")
            print(f"  Scores >= {relevance_threshold}: {sum(1 for s in relevance_scores if s >= relevance_threshold)}/{len(relevance_scores)}")
            print(f"  Scores < {relevance_threshold}: {sum(1 for s in relevance_scores if s < relevance_threshold)}/{len(relevance_scores)}")
    print("=" * 80 + "\n")
    
    # Extract metrics - define baseline first
    baseline_name = "Centralized RAG"
    baseline_results = all_results[baseline_name]
    
    # Always use adaptive thresholds based on actual score distribution
    baseline_scores = [r.retrieval_relevance_score for r in baseline_results if hasattr(r, 'retrieval_relevance_score')]
    use_three_way = True  # Always use three-way classification for better discrimination
    if baseline_scores:
        median_score = np.median(baseline_scores)
        q25 = np.percentile(baseline_scores, 25)
        q75 = np.percentile(baseline_scores, 75)
        min_score = np.min(baseline_scores)
        max_score = np.max(baseline_scores)
        
        # Use quartile-based thresholds for better discrimination
        # Low: below Q25, Medium: Q25 to Q75, High: above Q75
        low_threshold = q25
        high_threshold = q75
        
        print(f"📊 Score Distribution Analysis:")
        print(f"   Min={min_score:.3f}, Q25={q25:.3f}, Median={median_score:.3f}, Q75={q75:.3f}, Max={max_score:.3f}")
        print(f"   Using adaptive thresholds: Low < {low_threshold:.3f}, Medium: {low_threshold:.3f}-{high_threshold:.3f}, High > {high_threshold:.3f}\n")
    
    # Prepare data for confusion matrices
    metrics_data = {}
    for system_name, results in all_results.items():
        metrics_data[system_name] = {
            "relevance": [r.retrieval_relevance_score for r in results if hasattr(r, 'retrieval_relevance_score')],
            "quality": [r.answer_quality_score for r in results if hasattr(r, 'answer_quality_score')],
            "accuracy": [r.medical_accuracy_score for r in results if hasattr(r, 'medical_accuracy_score')],
        }
    
    # Prepare baseline classifications (will be updated if using three-way)
    baseline_relevance = [classify_performance(r.retrieval_relevance_score, relevance_threshold) 
                         for r in baseline_results if hasattr(r, 'retrieval_relevance_score')]
    baseline_quality = [classify_performance(r.answer_quality_score, quality_threshold) 
                       for r in baseline_results if hasattr(r, 'answer_quality_score')] if not retrieval_only else []
    baseline_accuracy = [classify_performance(r.medical_accuracy_score, accuracy_threshold) 
                        for r in baseline_results if hasattr(r, 'medical_accuracy_score')] if not retrieval_only else []
    
    # Update baseline_relevance if using three-way classification
    if use_three_way:
        baseline_relevance = [classify_performance_three_way(r.retrieval_relevance_score, low_threshold, high_threshold) 
                            for r in baseline_results if hasattr(r, 'retrieval_relevance_score')]
    
    # Create confusion matrices for each system vs baseline
    num_metrics = 3 if not retrieval_only else 1
    fig, axes = plt.subplots(len(systems) - 1, num_metrics, figsize=(6 * num_metrics, 5 * (len(systems) - 1)))
    if num_metrics == 1:
        axes = axes.reshape(-1, 1)
    if len(systems) - 1 == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle("Confusion Matrices: System Performance vs Baseline", fontsize=16, fontweight='bold', y=0.995)
    
    row_idx = 0
    for system_name, system_results in all_results.items():
        if system_name == baseline_name:
            continue
        
        # Retrieval Relevance
        if use_three_way:
            system_relevance = [classify_performance_three_way(r.retrieval_relevance_score, low_threshold, high_threshold) 
                               for r in system_results if hasattr(r, 'retrieval_relevance_score')]
            baseline_relevance_3way = [classify_performance_three_way(r.retrieval_relevance_score, low_threshold, high_threshold) 
                                      for r in baseline_results if hasattr(r, 'retrieval_relevance_score')]
            min_len = min(len(baseline_relevance_3way), len(system_relevance))
            if min_len > 0:
                create_confusion_matrix_plot(
                    baseline_relevance_3way[:min_len],
                    system_relevance[:min_len],
                    f"Retrieval Relevance\n{system_name}",
                    labels=["Low", "Medium", "High"],
                    ax=axes[row_idx, 0],
                )
        else:
            system_relevance = [classify_performance(r.retrieval_relevance_score, relevance_threshold) 
                               for r in system_results if hasattr(r, 'retrieval_relevance_score')]
            min_len = min(len(baseline_relevance), len(system_relevance))
            if min_len > 0:
                create_confusion_matrix_plot(
                    baseline_relevance[:min_len],
                    system_relevance[:min_len],
                    f"Retrieval Relevance\n{system_name}",
                    labels=["Low", "High"],
                    ax=axes[row_idx, 0],
                )
        
        # Answer Quality (if not retrieval-only)
        if not retrieval_only:
            system_quality = [classify_performance(r.answer_quality_score, quality_threshold) 
                             for r in system_results if hasattr(r, 'answer_quality_score')]
            min_len = min(len(baseline_quality), len(system_quality))
            if min_len > 0:
                create_confusion_matrix_plot(
                    baseline_quality[:min_len],
                    system_quality[:min_len],
                    f"Answer Quality\n{system_name}",
                    labels=["Low", "High"],
                    ax=axes[row_idx, 1],
                )
            
            # Medical Accuracy
            system_accuracy = [classify_performance(r.medical_accuracy_score, accuracy_threshold) 
                             for r in system_results if hasattr(r, 'medical_accuracy_score')]
            min_len = min(len(baseline_accuracy), len(system_accuracy))
            if min_len > 0:
                create_confusion_matrix_plot(
                    baseline_accuracy[:min_len],
                    system_accuracy[:min_len],
                    f"Medical Accuracy\n{system_name}",
                    labels=["Low", "High"],
                    ax=axes[row_idx, 2],
                )
        
        row_idx += 1
    
    plt.tight_layout()
    
    # Save figure
    results_dir = PROJECT_ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(results_dir / "exp10_confusion_matrices.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved confusion matrices to: {results_dir / 'exp10_confusion_matrices.png'}")
    
    # Create an additional visualization showing actual score differences
    print("\n[7/6] Creating score difference heatmap...")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    # Prepare score differences
    baseline_scores_array = np.array([r.retrieval_relevance_score for r in baseline_results if hasattr(r, 'retrieval_relevance_score')])
    score_diffs = {}
    for system_name, system_results in all_results.items():
        if system_name == baseline_name:
            continue
        system_scores_array = np.array([r.retrieval_relevance_score for r in system_results if hasattr(r, 'retrieval_relevance_score')])
        min_len = min(len(baseline_scores_array), len(system_scores_array))
        if min_len > 0:
            score_diffs[system_name] = system_scores_array[:min_len] - baseline_scores_array[:min_len]
    
    # Create heatmap of score differences
    if score_diffs:
        # Prepare data for heatmap
        max_len = max(len(diff) for diff in score_diffs.values())
        diff_matrix = np.zeros((len(score_diffs), max_len))
        system_names_list = list(score_diffs.keys())
        
        for i, (system_name, diffs) in enumerate(score_diffs.items()):
            diff_matrix[i, :len(diffs)] = diffs
        
        # Create heatmap
        im = ax2.imshow(diff_matrix, aspect='auto', cmap='RdYlGn', vmin=-0.1, vmax=0.1)
        ax2.set_yticks(range(len(system_names_list)))
        ax2.set_yticklabels(system_names_list)
        ax2.set_xlabel('Question Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('System', fontsize=12, fontweight='bold')
        ax2.set_title('Score Differences: System vs Baseline (Green = Better, Red = Worse)', 
                     fontsize=14, fontweight='bold', pad=15)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Score Difference (System - Baseline)', fontsize=10)
        
        # Add grid
        ax2.set_xticks(range(0, max_len, max(1, max_len // 10)))
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig(results_dir / "exp10_score_differences.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved score difference heatmap to: {results_dir / 'exp10_score_differences.png'}")
    
    # Generate classification reports
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("EXPERIMENT 10: CONFUSION MATRIX RESULTS")
    report_lines.append("=" * 80)
    report_lines.append(f"\nThresholds:")
    if use_three_way:
        report_lines.append(f"  - Retrieval Relevance: Adaptive (Low < {low_threshold:.3f}, Medium: {low_threshold:.3f}-{high_threshold:.3f}, High > {high_threshold:.3f})")
    else:
        report_lines.append(f"  - Retrieval Relevance: {relevance_threshold}")
    if not retrieval_only:
        report_lines.append(f"  - Answer Quality: {quality_threshold}")
        report_lines.append(f"  - Medical Accuracy: {accuracy_threshold}")
    report_lines.append("\n" + "=" * 80)
    
    for system_name, results in all_results.items():
        if system_name == baseline_name:
            continue
        
        report_lines.append(f"\n{system_name} vs Baseline:")
        report_lines.append("-" * 80)
        
        # Retrieval Relevance - use same classification as matrices
        if use_three_way:
            baseline_relevance = [classify_performance_three_way(r.retrieval_relevance_score, low_threshold, high_threshold) 
                                 for r in baseline_results if hasattr(r, 'retrieval_relevance_score')]
            system_relevance = [classify_performance_three_way(r.retrieval_relevance_score, low_threshold, high_threshold) 
                               for r in results if hasattr(r, 'retrieval_relevance_score')]
            labels = ["Low", "Medium", "High"]
        else:
            baseline_relevance = [classify_performance(r.retrieval_relevance_score, relevance_threshold) 
                                 for r in baseline_results if hasattr(r, 'retrieval_relevance_score')]
            system_relevance = [classify_performance(r.retrieval_relevance_score, relevance_threshold) 
                               for r in results if hasattr(r, 'retrieval_relevance_score')]
            labels = ["Low", "High"]
        
        min_len = min(len(baseline_relevance), len(system_relevance))
        if min_len > 0:
            report = classification_report(
                baseline_relevance[:min_len],
                system_relevance[:min_len],
                labels=labels,
                output_dict=True,
                zero_division=0.0,
            )
            report_lines.append("\nRetrieval Relevance Classification Report:")
            # Overall accuracy
            if 'accuracy' in report:
                report_lines.append(f"  Overall Accuracy: {report['accuracy']:.3f} ({report['accuracy']*100:.1f}%)")
            report_lines.append("  Per-Class Metrics:")
            for label in labels:
                if label in report:
                    report_lines.append(f"    {label}:")
                    report_lines.append(f"      Precision: {report[label]['precision']:.3f}")
                    report_lines.append(f"      Recall: {report[label]['recall']:.3f}")
                    report_lines.append(f"      F1-Score: {report[label]['f1-score']:.3f}")
        
        if not retrieval_only:
            # Answer Quality
            baseline_quality = [classify_performance(r.answer_quality_score, quality_threshold) 
                               for r in baseline_results if hasattr(r, 'answer_quality_score')]
            system_quality = [classify_performance(r.answer_quality_score, quality_threshold) 
                             for r in results if hasattr(r, 'answer_quality_score')]
            
            min_len = min(len(baseline_quality), len(system_quality))
            if min_len > 0:
                report = classification_report(
                    baseline_quality[:min_len],
                    system_quality[:min_len],
                    labels=["Low", "High"],
                    output_dict=True,
                )
                report_lines.append("\nAnswer Quality Classification Report:")
                report_lines.append(f"  Precision (High): {report['High']['precision']:.3f}")
                report_lines.append(f"  Recall (High): {report['High']['recall']:.3f}")
                report_lines.append(f"  F1-Score (High): {report['High']['f1-score']:.3f}")
            
            # Medical Accuracy
            baseline_accuracy = [classify_performance(r.medical_accuracy_score, accuracy_threshold) 
                               for r in baseline_results if hasattr(r, 'medical_accuracy_score')]
            system_accuracy = [classify_performance(r.medical_accuracy_score, accuracy_threshold) 
                             for r in results if hasattr(r, 'medical_accuracy_score')]
            
            min_len = min(len(baseline_accuracy), len(system_accuracy))
            if min_len > 0:
                report = classification_report(
                    baseline_accuracy[:min_len],
                    system_accuracy[:min_len],
                    labels=["Low", "High"],
                    output_dict=True,
                )
                report_lines.append("\nMedical Accuracy Classification Report:")
                report_lines.append(f"  Precision (High): {report['High']['precision']:.3f}")
                report_lines.append(f"  Recall (High): {report['High']['recall']:.3f}")
                report_lines.append(f"  F1-Score (High): {report['High']['f1-score']:.3f}")
    
    # Save report
    report_text = "\n".join(report_lines)
    report_path = results_dir / "exp10_confusion_matrices_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    
    print(f"✓ Saved classification report to: {report_path}")
    print("\n" + "=" * 80)
    print("EXPERIMENT 10 COMPLETE!")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment 10: Confusion Matrices")
    parser.add_argument("--num-questions", type=int, default=30, help="Number of questions to evaluate")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon value for DP")
    parser.add_argument("--delta", type=float, default=1e-5, help="Delta value for DP")
    parser.add_argument("--retrieval-only", action="store_true", help="Run in retrieval-only mode (no LLM calls)")
    
    args = parser.parse_args()
    
    run_confusion_matrix_experiment(
        num_questions=args.num_questions,
        epsilon=args.epsilon,
        delta=args.delta,
        retrieval_only=args.retrieval_only,
    )
