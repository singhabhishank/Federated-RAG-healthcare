# Federated RAG Web UI

React + Vite + TypeScript frontend for the Privacy-Preserving Federated RAG system.

## Prerequisites

- Node.js 18+
- Backend API running (see main [README](../README.md))

## Setup

```bash
npm install
```

## Development

Start the dev server (proxies `/api` to the backend at `http://127.0.0.1:8000`):

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

**Ensure the API is running first** (from project root):

```bash
uvicorn api.server:app --reload --host 127.0.0.1 --port 8000
```

## Build

```bash
npm run build
```

Output is in `dist/`. Serve with any static host; set the API base URL if the backend is on a different origin.

## Pages

- **Landing / Login / Signup** – Auth UI (mock; no backend auth)
- **Dashboard** – Overview and quick link to Ask
- **Ask** – Enter a medical question, set Top-K and options, run federated retrieval
- **Results** – View retrieval citations (after running a query)
- **Answers** – View generated answer and evidence sources
- **Clients** – Connected clients and document counts (from API)
- **Evaluation** – View or run evaluation metrics (from API)
- **Settings** – App settings
- **Privacy** – Privacy monitor
