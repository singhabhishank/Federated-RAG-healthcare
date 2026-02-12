# Federated-RAG-System-healthcare
## Federated RAG – Full Stack Guide (Backend + Frontend)

This project combines a **Python backend** (FastAPI) implementing a privacy‑preserving Federated RAG system with a **React/Vite frontend** for an end‑to‑end web experience.

---

### 1. Project Structure (high level)

- `api/server.py` – FastAPI backend:
  - `/api/query` – run federated RAG and return answer + citations
  - `/api/clients` – client status (doc counts, etc.)
  - `/api/evaluation` / `/api/evaluate` – view/run evaluation
  - `/api/health` – health check
- `src/federated_rag.py` – Federated RAG core (clients, coordinator, DP)
- `src/llm_providers.py` – LLM providers (OpenRouter, Groq, DeepSeek, etc.)
- `config.yaml` – RAG + LLM + privacy config
- `frontend/` – React + Vite + TypeScript UI
  - `src/App.tsx` – routing and auth protection
  - `src/auth.tsx` – simple email-based “login” stored in `localStorage`
  - `src/pages/` – dashboard, ask, results, answers, clients, evaluation, auth pages
  - `src/api.ts` – frontend API client for `/api/*`

---

### 2. Backend Setup

From project root (`/Users/macbookpro/Downloads/FL_RAG`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 2.1 Configure LLM (DeepSeek example)

`config.yaml` (already set up for DeepSeek):

```yaml
llm:
  provider: deepseek
  model: deepseek-chat
  temperature: 0.3
  max_tokens: 1500
  use_api: true
```

Create/update `.env` in the project root:

```bash
DEEPSEEK_API_KEY=your_deepseek_key_here
```

> Rotate keys if any were pasted into chat earlier.

#### 2.2 Run the backend API

```bash
cd /Users/macbookpro/Downloads/FL_RAG
source .venv/bin/activate
uvicorn api.server:app --reload --host 127.0.0.1 --port 8000
```

Test:

```bash
curl http://127.0.0.1:8000/api/health
# -> {"status":"ok"}
```

---

### 3. Frontend Setup

In another terminal:

```bash
cd /Users/macbookpro/Downloads/FL_RAG/frontend
npm install
npm run dev
```

The Vite dev server runs at:

- `http://localhost:5173/`

`frontend/vite.config.ts` proxies `/api` → `http://127.0.0.1:8000`, so the UI talks to the FastAPI backend in dev.

---

### 4. Auth & Navigation

The frontend uses a lightweight in‑browser auth:

- **Login / Signup**:
  - Email is stored in `localStorage` as `authUser`.
  - After login/signup, you are redirected to `/dashboard`.
- **Protected routes** (`/dashboard`, `/ask`, `/results`, `/answers`, `/clients`, `/evaluation`, `/settings`):
  - Wrapped in `ProtectedRoute` – if not authenticated, you’re sent to `/login`.
- **Sign out**:
  - Sidebar “Sign Out” button clears auth state (`authUser`) and redirects to `/login`.
- **Top bar user text**:
  - Shows the logged‑in email and “Signed in”.

---

### 5. Query Flow & History

1. **Ask a Question**:
   - Go to `/ask`.
   - Enter a medical question and optional settings (Top‑K, etc.).
   - Click **Run Federated Retrieval**.
   - Frontend calls `POST /api/query`:
     - Backend performs federated retrieval with DP.
     - LLM (DeepSeek) generates an answer using metadata‑only evidence.
   - You are redirected to `/results` (Federated Evidence).

2. **Federated Evidence (`/results`)**:
   - Shows:
     - Current question.
     - Evidence list (citations).
     - A **History** panel with previous questions.
   - **History click behavior**:
     - New entries store the full `QueryResponse` (answer + citations).
     - Clicking a history row:
       - If it has stored `response`, navigates back to `/results` with that response and shows the previous answer/evidence.
       - If it’s an older entry without `response`, it falls back to re‑running via `/ask`.

3. **Answer Output (`/answers`)**:
   - Shows the generated answer and evidence list for the current `queryResponse`.

4. **Dashboard**:
   - “Recent Questions” reads `federatedQueryHistory` from `localStorage`.
   - “Client Status” calls `/api/clients` to show live client doc counts and online status.

History is stored per browser/device (via `localStorage`), so each user’s browser keeps its own question history.

---

### 6. API Summary (for reference)

Backend (FastAPI in `api/server.py`):

- `GET /api/health`
  - Simple health check.
- `POST /api/query`
  - Body: `{ question: string, top_k_per_client?: number, epsilon?: number }`
  - Returns:
    - `answer` (string)
    - `citations` (array of metadata objects)
    - `num_references`
    - `clients_contributed`
    - `model_used`
- `GET /api/clients`
  - Returns array of:
    - `id`, `name`, `status`, `docs`, `version`
- `GET /api/evaluation`
  - Returns contents of `evaluation_results.json` if present.
- `POST /api/evaluate`
  - Body: `{ num_questions: number }`
  - Runs `evaluate_federated_rag.py`, then returns aggregated results.

---

### 7. Common Issues & Notes

- **401 / auth errors from DeepSeek**:
  - Check `DEEPSEEK_API_KEY` in `.env`.
  - Restart backend after changing `.env`.
- **503 `RAG init failed`**:
  - Usually due to missing/bad API key or environment mismatch.
  - Check backend logs in the terminal for the full error.
- **Torch / transformers version mismatch**:
  - For this venv, `torch 2.2.x` works with the pinned `transformers`/`sentence-transformers` in `requirements.txt`.
  - Avoid upgrading torch beyond what wheels are available for your macOS/Python combo unless you also re-pin transformers.

---

### 8. Where to Customize

- **Change LLM provider/model**: `config.yaml` → `llm` section.
- **Change DP parameters**: `config.yaml` → `privacy` (`epsilon`, `delta`).
- **Change embedding model / retrieval K**: `config.yaml` → `rag`.
- **Adjust UI text / branding**: `frontend/src/pages/LandingPage.tsx`, `TopBar.tsx`, `Sidebar.tsx`.

