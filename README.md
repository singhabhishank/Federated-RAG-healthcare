 <img width="1663" height="957" alt="2e6335c3f27d72834f0e5deea46adced" src="https://github.com/user-attachments/assets/e3a0b7e4-bd1b-4e33-ac82-2155e8d15848" />
FederatedMed: Privacy-Preserving Federated RAG for Healthcare
"I recently implemented FederatedMed, a federated retrieval-augmented generation (RAG) framework designed for evidence-based medical QA under strict data locality constraints.

🎯 Motivation

Healthcare institutions require high-quality, citation-backed answers from distributed medical literature.
However, centralizing documents across institutions is often infeasible due to:
 • Regulatory constraints
 • Institutional data governance policies
 • Confidentiality and IP concerns
 <img width="1663" height="957" alt="bcc319aa3e9ad969fa8f4f7c08a80ab2" src="https://github.com/user-attachments/assets/c5d2f0fa-aa1f-42fa-ae0c-7241de914015" />
🏗 System Design

The system simulates multiple “hospital” nodes, each maintaining a local vector database.

Query flow:
 1. A user query is broadcast to all sites
 2. Each site performs local semantic retrieval
 3. Only privacy-protected signals + limited metadata are shared
 4. A coordinator aggregates results and generates a cited response

No raw documents or full embeddings leave local storage.

🧠 Core Technical Components

1️⃣ Federated Retrieval
 • Retrieval is performed exclusively on local vector stores
 • The coordinator never accesses raw corpora
 • Aggregation is performed over protected signals

2️⃣ RAG Pipeline
 • Sentence Transformers for embedding generation
 • ChromaDB as local vector store
 • Coordinator-level context construction
 • Citation grounding via DOI / PMC references

3️⃣ Differential Privacy Mechanism
 • Gaussian noise injected into embeddings
 • (ε, δ)-DP formulation
 • Re-ranking performed over noised representations
 • Privacy perturbation explicitly reflected in final ranking

This allows controlled privacy–utility trade-off analysis.
<img width="1663" height="957" alt="432d360e44dce16f76acaeb89e6315c4" src="https://github.com/user-attachments/assets/82d5844b-c47c-4af9-91a2-04c9d83bf148" />
<img width="1663" height="957" alt="85a518ee4f3848b60c1fc004d42c2457" src="https://github.com/user-attachments/assets/0f16e6c9-4894-4095-9a0c-f08cd25c4631" />
4️⃣ Metadata-Constrained Context
 • The LLM receives only:
 • Titles
 • Journal names
 • Publication years
 • No full-text exposure by default
 • Reduces leakage risk in prompt construction

5️⃣ Multi-Provider LLM Layer

Supports interchangeable inference backends:
 • DeepSeek
 • Groq
 • OpenRouter
 • OpenAI
 • Qwe
 <img width="1663" height="957" alt="5f0e133ed04d1fd36c75a4164b241a7e" src="https://github.com/user-attachments/assets/38667cb5-d14a-43c5-8096-ba15a1ab1821" />
n
 nables cross-model behavior analysis under identical retrieval conditions.

6️⃣ Full-Stack Implementation
 • FastAPI backend
 • React + Vite frontend
 • Lightweight authentication
 • Query traceability
<img width="1663" height="957" alt="9de5cac733201cd62ba7efbb4b73cf1a" src="https://github.com/user-attachments/assets/f28c486a-c002-4500-8d46-881574086aba" />
<img width="1663" height="957" alt="9ecd9e002353306be7cbda144c633e26" src="https://github.com/user-attachments/assets/d0b1d095-ff14-4147-8cda-b92eb5777063" />


