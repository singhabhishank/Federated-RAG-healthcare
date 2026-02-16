"""
Federated RAG (Retrieval-Augmented Generation) System for Healthcare

Goal:
    Answer medical questions using distributed private data from multiple
    "hospitals" (clients) without sharing raw documents.

Key properties:
    - Each client has its own private ChromaDB vector store.
    - Queries are executed locally on each client.
    - Clients share only DIFFERENTIALLY PRIVATE embeddings + light metadata.
    - A coordinator securely aggregates results and calls an LLM (Qwen API).

NOTE:
    This module does NOT perform any classification training or centralized
    vector indexing. It is independent from the federated_learning.py module.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import os
import math

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from .llm_providers import create_llm_provider, LLMProvider


def _embedding_to_numpy(emb) -> np.ndarray:
    """Convert embedding (list, ndarray, or torch.Tensor) to np.float32. Avoids 'meta tensor' copy errors."""
    if isinstance(emb, np.ndarray):
        return np.asarray(emb, dtype=np.float32)
    if hasattr(emb, "detach"):  # torch.Tensor
        t = emb.detach()
        if t.is_meta:
            raise RuntimeError("Embedding is on meta device; model may not be fully loaded. Try restarting or use device='cpu'.")
        return t.cpu().numpy().astype(np.float32)
    return np.array(emb, dtype=np.float32)


# ---------------------------------------------------------------------------
# Differential Privacy mechanism for embeddings
# ---------------------------------------------------------------------------


def gaussian_noise_sigma(epsilon: float, delta: float = 1e-5, sensitivity: float = 1.0) -> float:
    """
    Compute Gaussian noise scale (sigma) for (epsilon, delta)-DP.

    Uses standard bound:
        sigma >= sqrt(2 * ln(1.25 / delta)) * sensitivity / epsilon
    """
    # Ensure numeric types
    epsilon = float(epsilon)
    delta = float(delta)
    sensitivity = float(sensitivity)
    
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0 for differential privacy")
    if delta <= 0:
        raise ValueError("delta must be > 0 for differential privacy")
    
    return math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon


def privatize_embedding(
    embedding: np.ndarray,
    epsilon: float,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    clip_norm: bool = True,
) -> np.ndarray:
    """
    Apply Gaussian DP noise to a single embedding vector.

    Steps:
        1. L2-normalize embedding to enforce ||x||_2 <= sensitivity.
        2. Add Gaussian noise N(0, sigma^2 I).
        3. Optional: Clip to prevent extreme values that hurt accuracy.
    
    Args:
        clip_norm: If True, clip the noisy embedding to prevent extreme values
                   that can significantly degrade retrieval accuracy.
    """
    if embedding.ndim != 1:
        raise ValueError("Expected 1D embedding vector")

    # L2-normalize to sensitivity ball
    norm = np.linalg.norm(embedding, ord=2) + 1e-8
    embedding_unit = embedding / norm
    embedding_scaled = embedding_unit * sensitivity

    sigma = gaussian_noise_sigma(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
    noise = np.random.normal(loc=0.0, scale=sigma, size=embedding.shape)
    noisy_embedding = embedding_scaled + noise
    
    # Clip to prevent extreme values that hurt accuracy
    # This helps maintain some semantic meaning while preserving privacy
    if clip_norm:
        noisy_norm = np.linalg.norm(noisy_embedding, ord=2)
        # Clip to 2x original norm to prevent extreme outliers
        max_norm = 2.0 * sensitivity
        if noisy_norm > max_norm:
            noisy_embedding = noisy_embedding * (max_norm / noisy_norm)
    
    return noisy_embedding


# ---------------------------------------------------------------------------
# Federated RAG Client
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    client_id: int
    doc_id: str
    score: float
    embedding: np.ndarray
    metadata: Dict


class FederatedRAGClient:
    """
    Simulated "hospital" with its own private vector database.

    - Holds a subset of the global medical literature.
    - Indexes documents locally in a private ChromaDB collection.
    - For a query, returns ONLY DIFFERENTIALLY-PRIVATE embeddings and metadata.
    """

    def __init__(
        self,
        client_id: int,
        df: pd.DataFrame,
        embedder: SentenceTransformer,
        vector_db_root: str = "./federated_vector_db",
        collection_name: Optional[str] = None,
    ):
        self.client_id = client_id
        self.df = df.reset_index(drop=True)
        self.embedder = embedder

        # Create a dedicated ChromaDB path for this client
        client_db_path = Path(vector_db_root) / f"client_{client_id}"
        client_db_path.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(client_db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name or f"medical_client_{client_id}"
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def build_local_index(self, batch_size: int = 128, include_abstract_in_metadata: bool = False) -> None:
        """
        Build the private vector index for this client from its DataFrame.

        Text for embedding = Title + Abstract (no patient records).
        If include_abstract_in_metadata is True, a truncated abstract is stored in metadata
        (and will be sent to the coordinator with each chunk). Improves grounding; reduces privacy.
        """
        if self.collection.count() > 0:
            print(f"[Client {self.client_id}] Vector DB already contains {self.collection.count()} documents.")
            return

        print(f"[Client {self.client_id}] Building local vector index...")
        texts: List[str] = []
        metadatas: List[Dict] = []
        ids: List[str] = []

        # Normalize column names (CSV may have Title/Abstract or title/abstract)
        def _get(row: pd.Series, *keys: str) -> str:
            for k in keys:
                if k in row.index:
                    v = row.get(k)
                    if pd.notna(v):
                        return str(v).strip()
            return ""

        for idx, row in self.df.iterrows():
            title = _get(row, "Title", "title") or ""
            abstract = _get(row, "Abstract", "abstract") or ""
            if not title and not abstract:
                continue

            combined_text = f"{title} {abstract}".strip()
            doc_id = f"client{self.client_id}_doc{idx}"

            meta = {
                "client_id": self.client_id,
                "doc_id": doc_id,
                "title": (title or "")[:200],  # truncated for safety
                "journal": _get(row, "Journal", "journal") or "",
                "year": _get(row, "Year", "year") or "",
            }
            # DOI from dataset (column DOI or doi); PMC_ID as fallback identifier
            doi_val = _get(row, "DOI", "doi")
            if doi_val:
                meta["doi"] = doi_val.strip()[:256]
            pmc_val = _get(row, "PMC_ID", "pmc_id", "PMC ID")
            if pmc_val:
                meta["pmc_id"] = str(pmc_val).strip()[:64]
            if include_abstract_in_metadata and abstract:
                # ChromaDB metadata value limit is 4096 bytes; keep under 500 chars to be safe
                meta["abstract"] = abstract[:500].strip()  # privacy tradeoff: abstract leaves client

            texts.append(combined_text)
            metadatas.append(meta)
            ids.append(doc_id)

        # Embed and insert in batches
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            batch_texts = texts[start:end]
            batch_ids = ids[start:end]
            batch_metas = metadatas[start:end]

            if not batch_texts:
                continue

            raw_embeddings = self.embedder.encode(batch_texts, show_progress_bar=False)
            # Ensure numpy (avoid meta tensor copy); Chroma expects list of lists
            if hasattr(raw_embeddings, "shape") and len(raw_embeddings.shape) == 2:
                embeddings = [_embedding_to_numpy(raw_embeddings[i]).tolist() for i in range(raw_embeddings.shape[0])]
            else:
                embeddings = [_embedding_to_numpy(e).tolist() for e in raw_embeddings]
            self.collection.add(
                ids=batch_ids,
                metadatas=batch_metas,
                embeddings=embeddings,
            )

        n_docs = self.collection.count()
        n_with_id = sum(1 for m in metadatas if m.get("doi") or m.get("pmc_id"))
        print(f"[Client {self.client_id}] Indexed {n_docs} documents (DOI/PMC_ID in {n_with_id}).")

    def local_retrieve(
        self,
        question: str,
        top_k: int = 5,
        epsilon: float = 1.0,
        delta: float = 1e-5,
    ) -> List[RetrievedChunk]:
        """
        Perform LOCAL retrieval on the client's private DB and return
        DIFFERENTIALLY PRIVATE embeddings for aggregation.
        """
        if self.collection.count() == 0:
            return []

        raw_query_emb = self.embedder.encode([question], show_progress_bar=False)[0]
        query_emb = _embedding_to_numpy(raw_query_emb)

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, self.collection.count()),
            include=["embeddings", "metadatas", "distances"],
        )

        chunks: List[RetrievedChunk] = []
        for doc_id, meta, emb, dist in zip(
            results["ids"][0],
            results["metadatas"][0],
            results["embeddings"][0],
            results["distances"][0],
        ):
            emb_np = _embedding_to_numpy(emb)
            # Apply DP noise BEFORE leaving client
            emb_private = privatize_embedding(
                emb_np,
                epsilon=epsilon,
                delta=delta,
                sensitivity=1.0,
                clip_norm=True,  # Clip extreme values to improve accuracy
            )
            chunks.append(
                RetrievedChunk(
                    client_id=self.client_id,
                    doc_id=doc_id,
                    score=float(dist),
                    embedding=emb_private,
                    metadata=meta,
                )
            )

        return chunks


# ---------------------------------------------------------------------------
# Federated RAG Coordinator
# ---------------------------------------------------------------------------


class FederatedRAGCoordinator:
    """
    Coordinates federated retrieval across multiple clients and calls the LLM.
    """

    def __init__(
        self,
        clients: List[FederatedRAGClient],
        epsilon: float = 1.0,
        delta: float = 1e-5,
        llm_provider: Optional[LLMProvider] = None,
    ):
        self.clients = clients
        self.epsilon = epsilon
        self.delta = delta
        self.llm_provider = llm_provider

    def federated_retrieve(
        self,
        question: str,
        top_k_per_client: int = 5,
    ) -> List[RetrievedChunk]:
        """
        Query all clients and collect privacy-preserving retrieved chunks.
        
        DP noise affects the final ranking: we re-rank using noisy embeddings
        to simulate how DP impacts retrieval quality.
        """
        all_chunks: List[RetrievedChunk] = []
        for client in self.clients:
            client_chunks = client.local_retrieve(
                question,
                top_k=top_k_per_client,
                epsilon=self.epsilon,
                delta=self.delta,
            )
            all_chunks.extend(client_chunks)

        # Re-rank using noisy embeddings to show DP effect
        # This is where DP noise actually affects retrieval quality
        try:
            if all_chunks and self.clients and len(self.clients) > 0:
                # Get embedder from first client
                embedder = self.clients[0].embedder
                if embedder is not None:
                    raw_q = embedder.encode([question], show_progress_bar=False)[0]
                    query_emb = _embedding_to_numpy(raw_q)

                    # Re-compute distances using noisy embeddings (this is where DP affects ranking)
                    for chunk in all_chunks:
                        if chunk.embedding is not None and len(chunk.embedding) > 0:
                            try:
                                # Compute cosine distance between query and noisy embedding
                                noisy_emb = _embedding_to_numpy(chunk.embedding)
                                # Cosine distance = 1 - cosine similarity
                                query_norm = np.linalg.norm(query_emb)
                                noisy_norm = np.linalg.norm(noisy_emb)
                                if query_norm > 0 and noisy_norm > 0:
                                    cosine_sim = np.dot(query_emb, noisy_emb) / (query_norm * noisy_norm + 1e-8)
                                    chunk.score = float(1.0 - cosine_sim)  # Convert similarity to distance
                            except (ValueError, AttributeError, TypeError) as e:
                                # If re-ranking fails, keep original score
                                pass
        except Exception:
            # If re-ranking fails entirely, use original scores
            pass
        
        # Sort by re-computed distance (smaller = more relevant)
        all_chunks.sort(key=lambda c: c.score)
        
        # Adaptive filtering: Use less strict threshold when DP is active
        # DP noise can make good matches appear worse, so we need to be more lenient
        if self.epsilon < 10.0:  # If DP is active (epsilon < 10 means significant noise)
            # More lenient filtering for DP systems
            filter_threshold = 0.90  # Increased from 0.85 to 0.90
            min_results = 12  # Keep more results
        else:
            # Standard filtering for non-DP systems
            filter_threshold = 0.85
            min_results = 10
        
        # Filter out very low-relevance results
        filtered_chunks = [c for c in all_chunks if c.score < filter_threshold]
        
        # If we filtered too many, keep at least min_results
        if len(filtered_chunks) < min_results and len(all_chunks) >= min_results:
            filtered_chunks = all_chunks[:max(min_results, len(filtered_chunks))]
        elif not filtered_chunks:
            # If all filtered out, keep top 5
            filtered_chunks = all_chunks[:5] if all_chunks else []
        
        return filtered_chunks

    def build_llm_context(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        max_refs: int = 10,
    ) -> str:
        """
        Build context for the LLM from retrieved chunks.
        Uses high-level metadata (title, journal, year) and, when stored,
        truncated abstracts so answers can be grounded in document content.
        """
        lines: List[str] = []
        has_abstracts = any(c.metadata.get("abstract", "").strip() for c in chunks[:max_refs])
        if has_abstracts:
            lines.append(
                "You are a medical expert using a privacy-preserving federated RAG system.\n"
                "You have access to high-level metadata (titles, journals, years) and truncated abstracts from multiple hospitals' literature databases. Use this evidence to ground your answer.\n"
            )
        else:
            lines.append(
                "You are a medical expert using a privacy-preserving federated RAG system.\n"
                "You DO NOT have access to raw patient data or full article texts. "
                "You only see high-level metadata (titles, journals, years) from multiple hospitals' literature databases.\n"
            )
        lines.append("Question:")
        lines.append(question)
        lines.append("\nFederated evidence (high-level citations only, sorted by relevance):")

        # Show top references (already sorted by similarity score)
        for i, chunk in enumerate(chunks[:max_refs], 1):
            meta = chunk.metadata
            title = meta.get("title", "[No title]")
            journal = meta.get("journal", "[Unknown journal]")
            year = meta.get("year", "[Unknown year]")
            client_id = meta.get("client_id", chunk.client_id)
            abstract = meta.get("abstract", "").strip()
            doi = (meta.get("doi") or "").strip()
            pmc_id = (meta.get("pmc_id") or "").strip()
            # Include similarity score as relevance indicator
            relevance_indicator = "✓" if chunk.score < 0.5 else "~"  # Lower score = more similar
            lines.append(
                f"- Source {i} {relevance_indicator} (Client {client_id}): \"{title}\" "
                f"({journal}, {year})"
            )
            if doi:
                lines.append(f"  DOI: {doi}")
            if pmc_id and not doi:
                lines.append(f"  PMC ID: {pmc_id}")
            if abstract:
                lines.append(f"  Abstract: {abstract}")

        lines.append(
            "\nInstructions for Answer Generation:\n"
            + (
                "1. **Ground your answer in the evidence above** (titles, journals, years, and abstracts where provided). Prefer claims supported by the cited sources.\n"
                if has_abstracts
                else "1. **Use ONLY the high-level evidence** (titles, journals, years) provided above.\n"
            )
            + "2. **Write in clear, natural prose**—like a knowledgeable clinician or reviewer explaining the topic. Use full paragraphs and normal sentences. Do NOT use a rigid template with ## headers, bullet lists for every point, or a mandatory ## Summary section.\n"
            "3. **Cite sources inline** where they support a claim, e.g. (Source 1) or (Source 2, Source 4). Be accurate and do not fabricate information.\n"
            "4. You may use **bold** for a few key terms if helpful, and occasional bullet points only where a short list is genuinely clearer than prose. Do not structure the entire answer as sections and sub-bullets.\n"
            "5. **Do NOT produce** a formal report with ## Overview, ## Pathophysiology, ## Etiology, ## Summary, etc. That format looks templated and artificial. Instead, answer the question in a direct, readable way with citations.\n"
            "\nWrite your answer now:"
        )

        return "\n".join(lines)

    def _filter_chunks_by_year(
        self,
        chunks: List[RetrievedChunk],
        year_min: Optional[int],
        year_max: Optional[int],
    ) -> List[RetrievedChunk]:
        """Filter chunks by metadata year (inclusive). Non-numeric or missing year is kept if no filter."""
        if year_min is None and year_max is None:
            return chunks
        out: List[RetrievedChunk] = []
        for c in chunks:
            raw = c.metadata.get("year") or ""
            try:
                y = int(str(raw).strip())
            except (ValueError, TypeError):
                # Keep if we can't parse (e.g. empty); or drop. We keep to avoid over-filtering.
                out.append(c)
                continue
            if year_min is not None and y < year_min:
                continue
            if year_max is not None and y > year_max:
                continue
            out.append(c)
        return out

    def generate_answer(
        self,
        question: str,
        top_k_per_client: int = 5,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> Dict:
        """
        Full pipeline:
            1. Federated retrieval with DP.
            2. Optionally filter by year (metadata).
            3. Aggregate results.
            4. Call LLM for answer.
        """
        chunks = self.federated_retrieve(question, top_k_per_client=top_k_per_client)
        chunks = self._filter_chunks_by_year(chunks, year_min, year_max)

        abstracts_included = any(
            (c.metadata.get("abstract") or "").strip() for c in chunks
        )

        def _normalize_citation(m: dict) -> dict:
            """Ensure citation has consistent keys for API/frontend (doi, pmc_id); privacy-safe metadata only."""
            m = m or {}
            return {
                "client_id": m.get("client_id"),
                "doc_id": m.get("doc_id"),
                "title": (m.get("title") or "").strip(),
                "journal": (m.get("journal") or "").strip(),
                "year": (m.get("year") or "").strip(),
                "abstract": (m.get("abstract") or "").strip(),
                "doi": (m.get("doi") or m.get("DOI") or "").strip()[:256],
                "pmc_id": (m.get("pmc_id") or m.get("PMC_ID") or "").strip()[:64],
            }

        if self.llm_provider is None:
            # Retrieval-only mode
            return {
                "answer": "LLM provider not configured. Returning retrieved citations only.",
                "citations": [_normalize_citation(c.metadata) for c in chunks],
                "num_references": len(chunks),
                "model_used": "none",
                "clients_contributed": sorted({c.client_id for c in chunks}),
                "abstracts_included": abstracts_included,
            }

        context = self.build_llm_context(question, chunks)
        answer = self.llm_provider.generate(prompt=context)

        # Summarise which clients contributed
        contributing_clients = sorted({c.client_id for c in chunks})

        return {
            "answer": answer,
            "citations": [_normalize_citation(c.metadata) for c in chunks],
            "num_references": len(chunks),
            "model_used": type(self.llm_provider).__name__,
            "clients_contributed": contributing_clients,
            "abstracts_included": abstracts_included,
        }


# ---------------------------------------------------------------------------
# Helper to build clients from a single CSV
# ---------------------------------------------------------------------------


def create_federated_rag_clients_from_csv(
    csv_path: str,
    num_clients: int = 3,
    max_docs_per_client: int = 8000,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    vector_db_root: str = "./federated_vector_db",
    include_abstract_in_metadata: bool = False,
) -> Tuple[List[FederatedRAGClient], SentenceTransformer]:
    """
    Split a single CSV of medical articles into multiple clients and
    create a FederatedRAGClient for each, each with its own ChromaDB.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Normalize column names (strip spaces) so DOI/PMC_ID etc. are found
    df.columns = [str(c).strip() for c in df.columns]
    # Find Title/Abstract columns (case-insensitive) for cleaning
    title_col = next((c for c in df.columns if c.lower() == "title"), None)
    abstract_col = next((c for c in df.columns if c.lower() == "abstract"), None)
    if title_col is None or abstract_col is None:
        raise ValueError("CSV must contain Title and Abstract columns")
    df = df.dropna(subset=[title_col, abstract_col])
    df = df[df[title_col].astype(str).str.strip() != ""]
    df = df[df[abstract_col].astype(str).str.strip() != ""]
    # Rename to canonical so build_local_index can use _get(row, 'Title', 'title')
    if title_col != "Title":
        df = df.rename(columns={title_col: "Title"})
    if abstract_col != "Abstract":
        df = df.rename(columns={abstract_col: "Abstract"})

    print(f"Loaded {len(df)} cleaned medical articles from {csv_path}")

    # Shuffle and split
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    total = len(df_shuffled)
    per_client = min(max_docs_per_client, total // num_clients)

    print(f"Distributing approximately {per_client} articles per client "
          f"({num_clients} clients)")

    # Load SentenceTransformer once and share across clients
    print(f"Loading embedding model for Federated RAG: {embedding_model}")
    # Prefer offline/cached usage
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache = os.path.join(cache_dir, f"models--{embedding_model.replace('/', '--')}")
    # Avoid meta device / lazy weights (fixes "Cannot copy out of meta tensor")
    model_kwargs = {"low_cpu_mem_usage": False}
    if os.path.exists(model_cache):
        print("   Model found in cache, using local files only (offline mode)...")
        embedder = SentenceTransformer(
            embedding_model, device="cpu", local_files_only=True, model_kwargs=model_kwargs
        )
    else:
        print("   Model not in cache, attempting to download...")
        embedder = SentenceTransformer(embedding_model, device="cpu", model_kwargs=model_kwargs)

    # Materialize model on CPU to avoid "Cannot copy out of meta tensor" (lazy/meta device)
    try:
        _ = embedder.encode(["warmup"], show_progress_bar=False)
    except Exception as e:
        print(f"   Warning: warmup encode failed: {e}")

    clients: List[FederatedRAGClient] = []
    start_idx = 0
    for client_id in range(num_clients):
        end_idx = start_idx + per_client
        client_df = df_shuffled.iloc[start_idx:end_idx].copy()
        start_idx = end_idx

        print(f"\n[Client {client_id}] Assigned {len(client_df)} articles")
        client = FederatedRAGClient(
            client_id=client_id,
            df=client_df,
            embedder=embedder,
            vector_db_root=vector_db_root,
        )
        client.build_local_index(include_abstract_in_metadata=include_abstract_in_metadata)
        clients.append(client)

    return clients, embedder


