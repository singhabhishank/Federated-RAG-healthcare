#!/usr/bin/env python3
"""
Experiment 1: Federated Retrieval Proof (Client-by-client view)

Goal: Prove each client only searches its own private DB.

This script:
1. Takes a medical question
2. Retrieves top-5 documents from each client separately
3. Shows individual client results in a formatted table
4. Shows the final aggregated top-15 list
5. Demonstrates that each client searches only its own database
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd

# Simple table formatter (replaces tabulate)
def format_table(data, headers, max_widths=None):
    """Simple table formatter without external dependencies."""
    if not data:
        return ""
    
    # Truncate cells to max widths
    if max_widths:
        for row in data:
            for i, cell in enumerate(row):
                if i < len(max_widths) and max_widths[i] and len(str(cell)) > max_widths[i]:
                    row[i] = str(cell)[:max_widths[i]-3] + "..."
    
    # Convert all to strings
    data_str = [[str(cell) for cell in row] for row in data]
    headers_str = [str(h) for h in headers]
    
    # Calculate column widths
    all_rows = [headers_str] + data_str
    num_cols = len(headers_str)
    col_widths = [max(len(str(row[i])) for row in all_rows) for i in range(num_cols)]
    
    # Build table
    lines = []
    # Top border
    top_border = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    lines.append(top_border)
    
    # Header
    header_row = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers_str)) + " |"
    lines.append(header_row)
    lines.append(top_border)
    
    # Data rows
    for row in data_str:
        data_row = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
        lines.append(data_row)
    
    # Bottom border
    lines.append(top_border)
    
    return "\n".join(lines)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
# Add project root to path so we can import from src
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.federated_rag import (
    create_federated_rag_clients_from_csv,
    FederatedRAGCoordinator,
    RetrievedChunk,
)
import yaml


def get_document_snippet(client, doc_id: str, max_length: int = 150) -> str:
    """
    Extract a snippet from the original document text.
    Returns first part of abstract or title if abstract not available.
    """
    try:
        # Extract index from doc_id (format: client{id}_doc{idx})
        # Example: "client0_doc123" -> idx = 123
        if "_doc" in doc_id:
            idx_str = doc_id.split("_doc")[-1]
            idx = int(idx_str)
            
            if 0 <= idx < len(client.df):
                row = client.df.iloc[idx]
                abstract = str(row.get("Abstract", "") or "")
                title = str(row.get("Title", "") or "")
                
                # Prefer abstract, fallback to title
                text = abstract if abstract.strip() else title
                
                if text.strip():
                    if len(text) > max_length:
                        return text[:max_length] + "..."
                    return text
    except (ValueError, IndexError, KeyError, AttributeError) as e:
        # Silently fail and return placeholder
        pass
    
    return "[Snippet not available]"


def format_similarity_score(score: float) -> str:
    """Format similarity score (distance) for display."""
    # ChromaDB uses cosine distance: lower = more similar
    # Convert to similarity: 1 - distance (clamped to [0, 1])
    similarity = max(0.0, min(1.0, 1.0 - score))
    return f"{similarity:.3f}"


def display_client_results(
    client_id: int,
    chunks: List[RetrievedChunk],
    client,
    question: str,
) -> None:
    """Display top-5 results from a single client in a formatted table."""
    print(f"\n{'='*80}")
    print(f"CLIENT {client_id} - Top 5 Documents")
    print(f"{'='*80}")
    print(f"Question: {question}\n")
    
    if not chunks:
        print("No documents found in this client's database.")
        return
    
    table_data = []
    for rank, chunk in enumerate(chunks[:5], 1):
        meta = chunk.metadata
        title = meta.get("title", "[No title]")
        journal = meta.get("journal", "[Unknown journal]")
        year = meta.get("year", "[Unknown year]")
        similarity = format_similarity_score(chunk.score)
        snippet = get_document_snippet(client, chunk.doc_id, max_length=120)
        
        table_data.append([
            rank,
            title[:60] + "..." if len(title) > 60 else title,
            journal[:30] + "..." if len(journal) > 30 else journal,
            year,
            similarity,
            snippet,
        ])
    
    headers = ["Rank", "Title", "Journal", "Year", "Similarity", "Snippet"]
    print(format_table(table_data, headers, max_widths=[5, 60, 30, 6, 10, 40]))
    print(f"\nTotal documents in Client {client_id} database: {client.collection.count()}")


def display_aggregated_results(
    all_chunks: List[RetrievedChunk],
    clients: List,
    question: str,
) -> None:
    """Display the final aggregated top-15 list."""
    print(f"\n{'='*80}")
    print("AGGREGATED TOP-15 RESULTS (After Federated Retrieval & Merging)")
    print(f"{'='*80}")
    print(f"Question: {question}\n")
    
    if not all_chunks:
        print("No documents found across all clients.")
        return
    
    # Sort by score (distance) - lower is better
    sorted_chunks = sorted(all_chunks, key=lambda c: c.score)[:15]
    
    # Create a mapping from client_id to client object for snippet extraction
    client_map = {client.client_id: client for client in clients}
    
    table_data = []
    contributing_clients = set()
    
    for rank, chunk in enumerate(sorted_chunks, 1):
        meta = chunk.metadata
        title = meta.get("title", "[No title]")
        journal = meta.get("journal", "[Unknown journal]")
        year = meta.get("year", "[Unknown year]")
        client_id = chunk.client_id
        similarity = format_similarity_score(chunk.score)
        snippet = get_document_snippet(client_map[client_id], chunk.doc_id, max_length=100)
        
        contributing_clients.add(client_id)
        
        table_data.append([
            rank,
            f"Client {client_id}",
            title[:55] + "..." if len(title) > 55 else title,
            journal[:25] + "..." if len(journal) > 25 else journal,
            year,
            similarity,
            snippet,
        ])
    
    headers = ["Rank", "Client", "Title", "Journal", "Year", "Similarity", "Snippet"]
    print(format_table(table_data, headers, max_widths=[5, 8, 55, 25, 6, 10, 40]))
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total aggregated documents: {len(sorted_chunks)}")
    print(f"Clients contributed: {sorted(contributing_clients)}")
    print(f"Question: {question}")
    print(f"{'='*80}\n")


def run_experiment(question: str = None):
    """Run the federated retrieval proof experiment."""
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
    
    print("="*80)
    print("EXPERIMENT 1: FEDERATED RETRIEVAL PROOF")
    print("="*80)
    print("\nGoal: Prove each client only searches its own private database.")
    print("="*80)
    
    # Create clients
    print("\n[1/3] Creating federated clients and building private vector databases...")
    clients, embedder = create_federated_rag_clients_from_csv(
        csv_path=str(csv_path),
        num_clients=num_clients,
        max_docs_per_client=max_docs_per_client,
        embedding_model=rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        vector_db_root=str(PROJECT_ROOT / "federated_vector_db"),
    )
    
    # Default question if not provided
    if question is None:
        question = "What are the cardiovascular risk factors?"
    
    print(f"\n[2/3] Retrieving documents from each client separately...")
    print(f"Question: {question}\n")
    
    # Retrieve from each client separately (without aggregation)
    all_client_chunks: Dict[int, List[RetrievedChunk]] = {}
    epsilon = float(privacy_cfg.get("epsilon", 1.0))
    delta = float(privacy_cfg.get("delta", 1e-5))
    
    for client in clients:
        chunks = client.local_retrieve(
            question,
            top_k=top_k_per_client,
            epsilon=epsilon,
            delta=delta,
        )
        all_client_chunks[client.client_id] = chunks
    
    # Display individual client results
    print("\n" + "="*80)
    print("STEP 1: INDIVIDUAL CLIENT RETRIEVAL RESULTS")
    print("="*80)
    
    for client_id, chunks in sorted(all_client_chunks.items()):
        client = clients[client_id]
        display_client_results(client_id, chunks, client, question)
    
    # Now show aggregated results
    print("\n" + "="*80)
    print("STEP 2: AGGREGATED RESULTS (Coordinator View)")
    print("="*80)
    
    # Use coordinator to get aggregated results
    coordinator = FederatedRAGCoordinator(
        clients=clients,
        epsilon=epsilon,
        delta=delta,
        llm_provider=None,  # Retrieval only, no LLM
    )
    
    aggregated_chunks = coordinator.federated_retrieve(
        question,
        top_k_per_client=top_k_per_client,
    )
    
    display_aggregated_results(aggregated_chunks, clients, question)
    
    # Save results to file
    results_dir = PROJECT_ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "exp1_federated_retrieval_proof.txt"
    
    # Capture output (we'll write it to file)
    print(f"\n[3/3] Saving results to: {results_file}")
    print("="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    # Write summary to file
    with open(results_file, "w") as f:
        f.write("EXPERIMENT 1: FEDERATED RETRIEVAL PROOF\n")
        f.write("="*80 + "\n\n")
        f.write(f"Question: {question}\n\n")
        
        f.write("CLIENT-BY-CLIENT RESULTS:\n")
        f.write("-"*80 + "\n")
        for client_id, chunks in sorted(all_client_chunks.items()):
            f.write(f"\nClient {client_id} - Top {len(chunks)} documents:\n")
            for rank, chunk in enumerate(chunks, 1):
                meta = chunk.metadata
                f.write(f"  {rank}. [{format_similarity_score(chunk.score)}] "
                       f"\"{meta.get('title', 'N/A')}\" "
                       f"({meta.get('journal', 'N/A')}, {meta.get('year', 'N/A')})\n")
        
        f.write("\n\nAGGREGATED RESULTS:\n")
        f.write("-"*80 + "\n")
        contributing_clients = sorted({c.client_id for c in aggregated_chunks[:15]})
        f.write(f"Clients contributed: {contributing_clients}\n")
        f.write(f"Total aggregated documents: {len(aggregated_chunks[:15])}\n\n")
        
        for rank, chunk in enumerate(aggregated_chunks[:15], 1):
            meta = chunk.metadata
            f.write(f"  {rank}. [Client {chunk.client_id}, {format_similarity_score(chunk.score)}] "
                   f"\"{meta.get('title', 'N/A')}\" "
                   f"({meta.get('journal', 'N/A')}, {meta.get('year', 'N/A')})\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Experiment 1: Federated Retrieval Proof"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Medical question to test (default: 'What are the cardiovascular risk factors?')"
    )
    
    args = parser.parse_args()
    run_experiment(question=args.question)
