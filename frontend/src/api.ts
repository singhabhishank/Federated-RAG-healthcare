/**
 * API client for Federated RAG backend.
 * Uses relative /api in dev (Vite proxy) and same in prod if served behind same host.
 */

const API_BASE = '';

export interface QueryRequest {
  question: string;
  top_k_per_client?: number;
  epsilon?: number;
  year_min?: number;
  year_max?: number;
}

export interface Citation {
  client_id?: number;
  doc_id?: string;
  title?: string;
  journal?: string;
  year?: string;
  abstract?: string;
  /** DOI from dataset (e.g. 10.1056/NEJMoa240123) */
  doi?: string;
  /** PubMed Central ID when DOI not available */
  pmc_id?: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  num_references: number;
  clients_contributed: number[];
  model_used: string;
  /** True if the LLM received document abstracts (answer is grounded in content). */
  abstracts_included?: boolean;
}

export interface ClientInfo {
  id: number;
  name: string;
  status: string;
  docs: number;
  latency?: string;
  version: string;
}

export interface EvaluationStats {
  total_questions?: number;
  successful_queries?: number;
  avg_references?: number;
  avg_retrieval_relevance?: number;
  avg_answer_quality?: number;
  avg_medical_accuracy?: number;
  privacy_compliance_rate?: number;
  [key: string]: unknown;
}

export interface EvaluationResponse {
  timestamp?: string;
  statistics: EvaluationStats;
  results: unknown[];
  message?: string;
}

async function handleRes<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    let detail = text;
    try {
      const j = JSON.parse(text);
      detail = j.detail || text;
    } catch {
      // use text
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function query(params: QueryRequest): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: params.question,
      top_k_per_client: params.top_k_per_client ?? 5,
      epsilon: params.epsilon,
      year_min: params.year_min,
      year_max: params.year_max,
    }),
  });
  return handleRes<QueryResponse>(res);
}

export interface PrivacyBudgetResponse {
  total_epsilon_used: number;
  budget_cap: number | null;
  remaining: number | null;
  delta: number;
}

export async function getPrivacyBudget(): Promise<PrivacyBudgetResponse> {
  const res = await fetch(`${API_BASE}/api/privacy-budget`);
  return handleRes<PrivacyBudgetResponse>(res);
}

export async function getClients(): Promise<ClientInfo[]> {
  const res = await fetch(`${API_BASE}/api/clients`);
  return handleRes<ClientInfo[]>(res);
}

export async function getEvaluation(): Promise<EvaluationResponse> {
  const res = await fetch(`${API_BASE}/api/evaluation`);
  return handleRes<EvaluationResponse>(res);
}

export async function runEvaluate(numQuestions: number = 10): Promise<EvaluationResponse> {
  const res = await fetch(`${API_BASE}/api/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ num_questions: numQuestions }),
  });
  return handleRes<EvaluationResponse>(res);
}

/** Rebuild the federated vector index from CSV so citations include DOI/PMC_ID from the dataset. */
export async function rebuildIndex(): Promise<{ status: string; message: string }> {
  const res = await fetch(`${API_BASE}/api/rebuild-index`, { method: 'POST' });
  return handleRes<{ status: string; message: string }>(res);
}
