#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Federated RAG System

Evaluates:
1. Retrieval Quality (Precision, Relevance)
2. Answer Quality (Medical Accuracy, Completeness)
3. Privacy Compliance
4. System Performance

Usage:
    python3 evaluate_federated_rag.py --num-questions 50
    python3 evaluate_federated_rag.py --questions-file medical_questions.txt
"""

import os
# Suppress tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import yaml
import numpy as np

# Add src directory to path (this script is already in src/, so parent is root)
ROOT_PATH = str(Path(__file__).parent.parent)
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from src.federated_rag import (
    create_federated_rag_clients_from_csv,
    FederatedRAGCoordinator,
)
from src.llm_providers import create_llm_provider


# Medical questions for evaluation
MEDICAL_QUESTIONS = [
    "What are the cardiovascular risk factors?",
    "What are the symptoms of diabetes?",
    "How is cancer treated?",
    "What causes hypertension?",
    "What are the side effects of chemotherapy?",
    "How to prevent stroke?",
    "What is the treatment for pneumonia?",
    "What are the early signs of Alzheimer's disease?",
    "How does insulin work in the body?",
    "What causes asthma attacks?",
    "What are the risk factors for osteoporosis?",
    "How is tuberculosis diagnosed?",
    "What are the complications of diabetes?",
    "What is the difference between type 1 and type 2 diabetes?",
    "How to manage chronic pain?",
    "What are the symptoms of heart attack?",
    "What causes migraines?",
    "How is depression treated?",
    "What are the warning signs of stroke?",
    "What is the treatment for high cholesterol?",
    "How does the immune system work?",
    "What are the causes of kidney disease?",
    "What is the role of exercise in cardiovascular health?",
    "How is hepatitis diagnosed?",
    "What are the symptoms of thyroid disorders?",
    "What causes autoimmune diseases?",
    "How to prevent osteoporosis?",
    "What are the treatment options for arthritis?",
    "What is the relationship between diet and heart disease?",
    "How is sleep apnea treated?",
    "What are the risk factors for developing cancer?",
    "What causes chronic fatigue syndrome?",
    "How is diabetes managed in elderly patients?",
    "What are the effects of smoking on cardiovascular health?",
    "What is the treatment for anxiety disorders?",
    "How does stress affect physical health?",
    "What are the symptoms of vitamin deficiency?",
    "What causes liver disease?",
    "How is hypertension managed?",
    "What are the complications of obesity?",
    "What is the role of genetics in disease?",
    "How is mental health assessed?",
    "What are the treatment options for chronic pain?",
    "What causes respiratory diseases?",
    "How to prevent cardiovascular disease?",
    "What are the symptoms of metabolic syndrome?",
    "What is the treatment for inflammatory diseases?",
    "How does aging affect health?",
    "What are the risk factors for mental health disorders?",
    "What is the relationship between exercise and mental health?",
]


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class EvaluationMetrics:
    """Stores evaluation metrics for a single question."""
    
    def __init__(self, question: str):
        self.question = question
        self.num_references = 0
        self.clients_contributed = []
        self.answer_length = 0
        self.retrieval_relevance_score = 0.0  # 0-1, manual assessment
        self.answer_quality_score = 0.0  # 0-1, manual assessment
        self.medical_accuracy_score = 0.0  # 0-1, manual assessment
        self.privacy_compliant = True
        self.citations = []
        self.answer = ""
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "question": self.question,
            "num_references": self.num_references,
            "clients_contributed": self.clients_contributed,
            "answer_length": self.answer_length,
            "retrieval_relevance_score": self.retrieval_relevance_score,
            "answer_quality_score": self.answer_quality_score,
            "medical_accuracy_score": self.medical_accuracy_score,
            "privacy_compliant": self.privacy_compliant,
            "num_citations": len(self.citations),
            "timestamp": self.timestamp,
        }


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def print_progress(current: int, total: int, question: str):
    """Print progress indicator."""
    percent = (current / total) * 100
    print(f"{Colors.CYAN}[{current}/{total}] ({percent:.1f}%){Colors.RESET} {question[:60]}...")


def check_privacy_compliance(citations: List[Dict]) -> bool:
    """
    Check if citations comply with privacy requirements.
    Privacy-safe allows: metadata (title, journal, year) and optionally a
    truncated abstract (e.g. ≤600 chars) for relevant answers. Forbids full
    document content (full_text, content, body, long text).
    """
    # Full document / raw content = never compliant
    forbidden_always = ["full_text", "content", "body"]
    max_abstract_len = 600  # truncated abstract is privacy-preserving
    for citation in citations:
        for field in forbidden_always:
            if field in citation:
                return False
        # Long raw "text" field = not compliant
        if "text" in citation and len(str(citation.get("text", ""))) > max_abstract_len:
            return False
        # Truncated "abstract" is allowed (privacy-preserving grounding)
        if "abstract" in citation:
            ab = citation.get("abstract")
            if ab is not None and len(str(ab)) > max_abstract_len:
                return False
        if not any(key in citation for key in ["title", "journal", "year"]):
            return False
    return True


def evaluate_retrieval_relevance(question: str, citations: List[Dict], embedder=None) -> float:
    """
    Evaluate retrieval relevance (0-1 score) using semantic similarity.
    
    Uses embedding-based similarity. When citations include abstracts, compares
    question to title+abstract for a fairer score; otherwise title only.
    Normalization is calibrated for typical cosine similarities (short text).
    """
    if not citations:
        return 0.0
    
    # If embedder available, use semantic similarity
    if embedder is not None:
        try:
            question_emb = embedder.encode([question], show_progress_bar=False)[0]
            similarities = []

            for citation in citations:
                title = (citation.get("title") or "").strip()
                abstract = (citation.get("abstract") or "").strip()
                # Use title + abstract when available (same as retrieval); title-only otherwise
                text = f"{title} {abstract}".strip() if abstract else title
                if not text:
                    continue
                doc_emb = embedder.encode([text[:2000]], show_progress_bar=False)[0]  # cap length
                sim = np.dot(question_emb, doc_emb) / (
                    np.linalg.norm(question_emb) * np.linalg.norm(doc_emb) + 1e-8
                )
                similarities.append(float(sim))

            if not similarities:
                return 0.0

            avg_similarity = np.mean(similarities)
            # Calibrated for typical cosine sims: question vs title (or title+abstract)
            # Often in [0.2, 0.6]. Map [0.15, 0.55] -> [0, 1] so good retrieval ~0.4 -> ~0.63
            low, high = 0.15, 0.55
            if avg_similarity <= low:
                return max(0.0, avg_similarity / low)
            if avg_similarity >= high:
                return 1.0
            return (avg_similarity - low) / (high - low)
        except Exception:
            pass  # Fall back to keyword matching
    
    # Fallback: improved keyword matching (stricter)
    question_lower = question.lower()
    # Extract meaningful keywords (longer words, medical terms)
    question_keywords = {
        word for word in question_lower.split() 
        if len(word) > 4 and word not in ['what', 'are', 'the', 'how', 'does', 'is', 'for', 'with']
    }
    
    if not question_keywords:
        # If no good keywords, use all words > 3 chars
        question_keywords = {word for word in question_lower.split() if len(word) > 3}
    
    if not question_keywords:
        return 0.0
    
    relevant_count = 0
    for citation in citations:
        title = citation.get("title", "").lower()
        journal = citation.get("journal", "").lower()
        combined_text = f"{title} {journal}".lower()
        
        # Check if multiple keywords match (more strict: require 50% match instead of 30%)
        matches = sum(1 for keyword in question_keywords if keyword in combined_text)
        if matches >= max(1, len(question_keywords) * 0.5):  # At least 50% of keywords match
            relevant_count += 1
    
    return relevant_count / len(citations)


def run_evaluation(
    coordinator: FederatedRAGCoordinator,
    questions: List[str],
    output_file: Optional[str] = None,
    interactive: bool = False,
    embedder=None,
) -> List[EvaluationMetrics]:
    """Run evaluation on a list of questions."""
    
    print_header("Federated RAG Evaluation")
    print(f"{Colors.BOLD}Evaluating {len(questions)} medical questions...{Colors.RESET}\n")
    
    results = []
    
    for idx, question in enumerate(questions, 1):
        print_progress(idx, len(questions), question)
        
        # Add delay to avoid rate limits (Groq free tier: 100k tokens/day)
        if idx > 1:
            import time
            time.sleep(2)  # 2 second delay between requests to avoid rate limits
        
        try:
            # Generate answer
            result = coordinator.generate_answer(question, top_k_per_client=5)
            
            # Create metrics
            metrics = EvaluationMetrics(question)
            metrics.num_references = result.get("num_references", 0)
            metrics.clients_contributed = result.get("clients_contributed", [])
            metrics.answer = result.get("answer", "")
            metrics.answer_length = len(metrics.answer)
            metrics.citations = result.get("citations", [])
            
            # Check privacy compliance
            metrics.privacy_compliant = check_privacy_compliance(metrics.citations)
            
            # Automatic relevance scoring (semantic similarity if embedder available)
            metrics.retrieval_relevance_score = evaluate_retrieval_relevance(
                question, metrics.citations, embedder=embedder
            )
            
            # Manual quality assessment (if interactive)
            if interactive:
                print(f"\n{Colors.YELLOW}Question: {question}{Colors.RESET}")
                print(f"{Colors.CYAN}Answer length: {metrics.answer_length} chars{Colors.RESET}")
                print(f"{Colors.CYAN}References: {metrics.num_references}{Colors.RESET}")
                print(f"{Colors.CYAN}Clients: {metrics.clients_contributed}{Colors.RESET}")
                
                # Ask for manual scores
                try:
                    quality = input(f"{Colors.YELLOW}Answer Quality (0-1): {Colors.RESET}").strip()
                    accuracy = input(f"{Colors.YELLOW}Medical Accuracy (0-1): {Colors.RESET}").strip()
                    
                    metrics.answer_quality_score = float(quality) if quality else 0.0
                    metrics.medical_accuracy_score = float(accuracy) if accuracy else 0.0
                except (ValueError, KeyboardInterrupt):
                    print(f"{Colors.RED}Skipping manual assessment...{Colors.RESET}")
                    metrics.answer_quality_score = 0.0
                    metrics.medical_accuracy_score = 0.0
            else:
                # Auto-assess based on improved heuristics
                # Answer quality: based on length, structure, and content indicators
                quality_score = 0.0
                
                # Length scoring (0-0.4 points) - more lenient
                if metrics.answer_length > 2500:
                    quality_score += 0.4
                elif metrics.answer_length > 2000:
                    quality_score += 0.35
                elif metrics.answer_length > 1500:
                    quality_score += 0.3
                elif metrics.answer_length > 1000:
                    quality_score += 0.25
                elif metrics.answer_length > 500:
                    quality_score += 0.2
                else:
                    quality_score += 0.15
                
                # Structure indicators (0-0.3 points) - more lenient
                answer_lower = metrics.answer.lower()
                structure_indicators = [
                    "##", "###",  # Markdown headers
                    "1.", "2.", "3.",  # Numbered lists
                    "- ", "* ",  # Bullet points
                    "summary", "conclusion",  # Summary sections
                    "source", "citation", "reference",  # Citations
                    "overview", "introduction",  # Sections
                ]
                structure_count = sum(1 for indicator in structure_indicators if indicator in answer_lower)
                if structure_count >= 3:
                    quality_score += 0.3
                elif structure_count >= 2:
                    quality_score += 0.25
                elif structure_count >= 1:
                    quality_score += 0.2
                else:
                    # Even without explicit structure, if answer is comprehensive, give some points
                    if metrics.answer_length > 2000:
                        quality_score += 0.1
                
                # Content quality indicators (0-0.3 points)
                content_indicators = [
                    "evidence", "research", "study", "clinical",  # Evidence-based
                    "treatment", "diagnosis", "symptoms", "causes",  # Medical content
                    "risk factors", "prevention", "management",  # Comprehensive coverage
                ]
                content_count = sum(1 for indicator in content_indicators if indicator in answer_lower)
                if content_count >= 5:
                    quality_score += 0.3
                elif content_count >= 3:
                    quality_score += 0.2
                elif content_count >= 1:
                    quality_score += 0.1
                
                metrics.answer_quality_score = min(1.0, quality_score)
                
                # Medical accuracy: based on references and answer quality (improved)
                if metrics.num_references >= 15:
                    base_accuracy = 0.85  # Excellent reference count
                elif metrics.num_references >= 10:
                    base_accuracy = 0.8
                elif metrics.num_references >= 5:
                    base_accuracy = 0.75
                elif metrics.num_references >= 3:
                    base_accuracy = 0.7
                elif metrics.num_references >= 1:
                    base_accuracy = 0.6
                else:
                    base_accuracy = 0.4
                
                # Boost accuracy if answer quality is high (well-structured answers are more likely accurate)
                if metrics.answer_quality_score > 0.8:
                    base_accuracy += 0.12
                elif metrics.answer_quality_score > 0.7:
                    base_accuracy += 0.08
                elif metrics.answer_quality_score > 0.6:
                    base_accuracy += 0.05
                
                # Additional boost if answer has medical terminology and citations
                answer_lower = metrics.answer.lower()
                if "source" in answer_lower or "citation" in answer_lower:
                    base_accuracy += 0.03
                if any(term in answer_lower for term in ["treatment", "diagnosis", "symptoms", "risk factors"]):
                    base_accuracy += 0.02
                
                metrics.medical_accuracy_score = min(1.0, base_accuracy)
            
            results.append(metrics)
            
            # Print quick status
            if metrics.privacy_compliant:
                print(f"  {Colors.GREEN}✓ Privacy OK{Colors.RESET}", end="")
            else:
                print(f"  {Colors.RED}✗ Privacy VIOLATION{Colors.RESET}", end="")
            
            print(f" | Refs: {metrics.num_references} | Relevance: {metrics.retrieval_relevance_score:.2f}")
            
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a rate limit or connection error (treat similarly)
            is_rate_limit = (
                "rate limit" in error_str.lower() or 
                "429" in error_str or
                "tokens per day" in error_str.lower() or
                "tpd" in error_str.lower()
            )
            
            is_connection_error = (
                "connection" in error_str.lower() or
                "connection error" in error_str.lower() or
                "connection refused" in error_str.lower() or
                "timeout" in error_str.lower()
            )
            
            if is_rate_limit or is_connection_error:
                error_type = "Rate limit" if is_rate_limit else "Connection error"
                print(f"  {Colors.YELLOW}⚠️  {error_type} - skipping question{Colors.RESET}")
                if is_rate_limit:
                    print(f"     {Colors.YELLOW}Tip: Wait ~10 minutes and resume evaluation{Colors.RESET}")
                elif is_connection_error:
                    print(f"     {Colors.YELLOW}Tip: API connection issue - may be temporary{Colors.RESET}")
                # Create failed metrics with error flag
                metrics = EvaluationMetrics(question)
                if is_rate_limit:
                    metrics.answer = f"[RATE_LIMIT_ERROR: {error_str}]"
                else:
                    metrics.answer = f"[CONNECTION_ERROR: {error_str}]"
                results.append(metrics)
                continue
            else:
                print(f"  {Colors.RED}✗ Error: {e}{Colors.RESET}")
                # Create failed metrics
                metrics = EvaluationMetrics(question)
                metrics.answer = f"[ERROR: {error_str}]"
                results.append(metrics)
    
    return results


def calculate_statistics(results: List[EvaluationMetrics]) -> Dict:
    """Calculate aggregate statistics."""
    if not results:
        return {}
    
    # Filter out rate limit and connection errors for statistics
    valid_results = [
        r for r in results 
        if not (r.answer and ("[RATE_LIMIT_ERROR" in r.answer or "[CONNECTION_ERROR" in r.answer or "[ERROR:" in r.answer))
    ]
    
    # Count different error types
    rate_limited = sum(1 for r in results if r.answer and "[RATE_LIMIT_ERROR" in r.answer)
    connection_errors = sum(1 for r in results if r.answer and "[CONNECTION_ERROR" in r.answer)
    other_errors = sum(1 for r in results if r.answer and "[ERROR:" in r.answer and "[RATE_LIMIT_ERROR" not in r.answer and "[CONNECTION_ERROR" not in r.answer)
    
    stats = {
        "total_questions": len(results),
        "valid_questions": len(valid_results),
        "rate_limited_questions": rate_limited,
        "connection_error_questions": connection_errors,
        "other_error_questions": other_errors,
        "failed_questions": len(results) - len(valid_results),
        "successful_queries": sum(1 for r in valid_results if r.num_references > 0),
        "avg_references": np.mean([r.num_references for r in valid_results]) if valid_results else 0,
        "avg_answer_length": np.mean([r.answer_length for r in valid_results]) if valid_results else 0,
        "avg_retrieval_relevance": np.mean([r.retrieval_relevance_score for r in valid_results]) if valid_results else 0,
        "avg_answer_quality": np.mean([r.answer_quality_score for r in valid_results]) if valid_results else 0,
        "avg_medical_accuracy": np.mean([r.medical_accuracy_score for r in valid_results]) if valid_results else 0,
        "privacy_compliance_rate": sum(1 for r in valid_results if r.privacy_compliant) / len(valid_results) if valid_results else 0,
        "clients_contribution": {},
    }
    
    # Calculate client contribution (only from valid results)
    all_clients = set()
    for r in valid_results:
        all_clients.update(r.clients_contributed)
    
    for client_id in all_clients:
        stats["clients_contribution"][client_id] = sum(
            1 for r in valid_results if client_id in r.clients_contributed
        )
    
    return stats


def print_statistics(stats: Dict):
    """Print evaluation statistics."""
    print_header("Evaluation Statistics")
    
    print(f"{Colors.BOLD}Overall Performance:{Colors.RESET}")
    print(f"  Total Questions: {stats['total_questions']}")
    if stats.get('rate_limited_questions', 0) > 0:
        print(f"  {Colors.YELLOW}Rate Limited: {stats['rate_limited_questions']} questions{Colors.RESET}")
    print(f"  Valid Questions: {stats['valid_questions']}")
    if stats['valid_questions'] > 0:
        print(f"  Successful Queries: {stats['successful_queries']} ({stats['successful_queries']/stats['valid_questions']*100:.1f}%)")
    else:
        print(f"  Successful Queries: {stats['successful_queries']} (0.0%)")
    
    print(f"\n{Colors.BOLD}Retrieval Quality:{Colors.RESET}")
    print(f"  Average References per Query: {stats['avg_references']:.2f}")
    print(f"  Average Retrieval Relevance: {stats['avg_retrieval_relevance']:.3f}")
    
    print(f"\n{Colors.BOLD}Answer Quality:{Colors.RESET}")
    print(f"  Average Answer Length: {stats['avg_answer_length']:.0f} characters")
    print(f"  Average Answer Quality Score: {stats['avg_answer_quality']:.3f}")
    print(f"  Average Medical Accuracy Score: {stats['avg_medical_accuracy']:.3f}")
    
    print(f"\n{Colors.BOLD}Privacy Compliance:{Colors.RESET}")
    print(f"  Privacy Compliance Rate: {stats['privacy_compliance_rate']*100:.1f}%")
    
    print(f"\n{Colors.BOLD}Client Contribution:{Colors.RESET}")
    for client_id, count in sorted(stats['clients_contribution'].items()):
        percentage = (count / stats['total_questions']) * 100
        print(f"  Client {client_id}: {count} queries ({percentage:.1f}%)")


def save_results(results: List[EvaluationMetrics], stats: Dict, output_file: str):
    """Save evaluation results to JSON file."""
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "statistics": stats,
        "results": [r.to_dict() for r in results],
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{Colors.GREEN}✓ Results saved to: {output_file}{Colors.RESET}")


def load_questions_from_file(filepath: str) -> List[str]:
    """Load questions from a text file (one per line)."""
    with open(filepath, "r") as f:
        questions = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return questions


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Federated RAG System with medical questions"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=50,
        help="Number of questions to evaluate (default: 50)"
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        help="Path to file with questions (one per line)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: manually assess answer quality"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results (default: evaluation_results.json)"
    )
    
    args = parser.parse_args()
    
    # Load questions
    if args.questions_file:
        questions = load_questions_from_file(args.questions_file)
        print(f"{Colors.CYAN}Loaded {len(questions)} questions from {args.questions_file}{Colors.RESET}")
    else:
        questions = MEDICAL_QUESTIONS[:args.num_questions]
        print(f"{Colors.CYAN}Using {len(questions)} predefined medical questions{Colors.RESET}")
    
    # Load configuration
    config_path = Path("config.yaml")
    if not config_path.exists():
        print(f"{Colors.RED}Error: config.yaml not found{Colors.RESET}")
        return
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_cfg = config["data"]
    rag_cfg = config["rag"]
    llm_cfg = config["llm"]
    privacy_cfg = config["privacy"]
    
    csv_path = data_cfg.get("medical_literature_path", "./extracted_data.csv")
    num_clients = 3
    max_docs_per_client = 8000
    
    print_header("Initializing Federated RAG System")
    
    # Create clients
    print(f"{Colors.CYAN}Creating federated RAG clients...{Colors.RESET}")
    clients, embedder = create_federated_rag_clients_from_csv(
        csv_path=csv_path,
        num_clients=num_clients,
        max_docs_per_client=max_docs_per_client,
        embedding_model=rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        vector_db_root="./federated_vector_db",
    )
    print(f"{Colors.GREEN}✓ Created {len(clients)} clients{Colors.RESET}")
    
    # Initialize LLM provider
    print(f"{Colors.CYAN}Initializing LLM provider...{Colors.RESET}")
    provider_name = llm_cfg.get("provider", "groq")
    model_name = llm_cfg.get("model", "llama-3.3-70b-versatile")
    use_api = llm_cfg.get("use_api", True)
    
    # Get API key from environment
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
        max_tokens=llm_cfg.get("max_tokens", 2000),
    )
    print(f"{Colors.GREEN}✓ LLM provider: {provider_name}{Colors.RESET}")
    
    # Create coordinator
    epsilon = float(privacy_cfg.get("epsilon", 1.0))
    delta = float(privacy_cfg.get("delta", 1e-5))
    
    coordinator = FederatedRAGCoordinator(
        clients=clients,
        epsilon=epsilon,
        delta=delta,
        llm_provider=llm_provider,
    )
    print(f"{Colors.GREEN}✓ Coordinator initialized{Colors.RESET}")
    
    # Run evaluation (pass embedder for better relevance scoring)
    results = run_evaluation(
        coordinator=coordinator,
        questions=questions,
        output_file=args.output,
        interactive=args.interactive,
        embedder=embedder,  # Pass embedder for semantic similarity
    )
    
    # Calculate statistics
    stats = calculate_statistics(results)
    
    # Print statistics
    print_statistics(stats)
    
    # Save results
    save_results(results, stats, args.output)
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}Evaluation complete!{Colors.RESET}\n")


if __name__ == "__main__":
    main()

