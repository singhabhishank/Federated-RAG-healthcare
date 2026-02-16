<img width="1663" height="957" alt="2e6335c3f27d72834f0e5deea46adced" src="https://github.com/user-attachments/assets/9b74f345-7655-4ba3-b218-8910d2044d8f" />
<img width="1663" height="957" alt="bcc319aa3e9ad969fa8f4f7c08a80ab2" src="https://github.com/user-attachments/assets/c556138c-d292-4719-9b2b-3d49b5b2e2fd" />
<img width="1663" height="957" alt="640d500307eb32b2a344bb71cf7bae2a" src="https://github.com/user-attachments/assets/a5a94e70-b217-4300-bec9-07caadc0c872" />
<img width="1663" height="957" alt="432d360e44dce16f76acaeb89e6315c4" src="https://github.com/user-attachments/assets/b49e3c6c-6c60-4aa0-8762-7792244a5374" />
<img width="1663" height="957" alt="85a518ee4f3848b60c1fc004d42c2457" src="https://github.com/user-attachments/assets/01a8fa0b-3e65-4562-86f7-c1448223f6ca" />
<img width="1663" height="957" alt="5f0e133ed04d1fd36c75a4164b241a7e" src="https://github.com/user-attachments/assets/2d026b8d-8526-433f-8325-a2034f72fd4c" />
<img width="1663" height="957" alt="1e732ea2c2851734c9b09952131d0876" src="https://github.com/user-attachments/assets/6d9fbbfc-f498-4f49-a839-f0f95d40d951" />
<img width="1663" height="957" alt="9ecd9e002353306be7cbda144c633e26" src="https://github.com/user-attachments/assets/721eed79-3abd-42d7-af10-4deb28b92978" />

# FederatedMed — Federated RAG for Privacy-Preserving Medical QA

FederatedMed is a federated retrieval-augmented generation (RAG) framework designed for **evidence-based medical question answering** under strict **data locality** and **privacy constraints**. The system enables multiple institutions to contribute retrieval evidence without centralizing raw documents, while still producing **citation-backed answers** using stable identifiers such as **DOI** and **PMC** references.

---

## Motivation

Healthcare institutions need high-quality, verifiable answers grounded in medical literature. In practice, medical corpora are distributed across hospitals, research centers, and institutions, and centralizing this content is often not possible due to:

- Regulatory constraints
- Institutional data governance policies
- Confidentiality and IP concerns

FederatedMed is built to address this gap by enabling cross-site retrieval and generation while preserving local control of data.

---

## System Design

FederatedMed simulates multiple independent “hospital” nodes. Each node maintains its own **local vector database** containing a private subset of the literature.

### Query Flow

1. A user query is broadcast to all participating sites  
2. Each site performs local semantic retrieval  
3. Only privacy-protected signals and limited metadata are shared  
4. A coordinator aggregates results and generates a cited response  

**No raw documents or full embeddings leave local storage.**

---

## Core Technical Components

### 1) Federated Retrieval
- Retrieval runs exclusively on **local vector stores**
- The coordinator never accesses raw corpora
- Aggregation is performed over **protected signals**

### 2) RAG Pipeline
- Sentence Transformers for embedding generation  
- ChromaDB as the local vector store  
- Coordinator-level context construction  
- Citation grounding via **DOI / PMC** references  

### 3) Differential Privacy Mechanism
- Gaussian noise injected into embeddings  
- Standard \((\varepsilon, \delta)\)-DP formulation  
- Re-ranking performed over noised representations  
- Privacy perturbation explicitly reflected in ranking outputs  

This supports controlled privacy–utility trade-off analysis.

### 4) Metadata-Constrained Context
The LLM receives only:
- Titles  
- Journal names  
- Publication years  

Full-text exposure is disabled by default, reducing leakage risk during prompt construction.

### 5) Multi-Provider LLM Layer
FederatedMed supports interchangeable inference backends:
- DeepSeek  
- Groq  
- OpenRouter  
- OpenAI  
- Qwen  

This enables cross-model behavior analysis under identical retrieval conditions.

### 6) Full-Stack Implementation
- FastAPI backend  
- React + Vite frontend  
- Lightweight authentication  
- Query traceability and evidence inspection  

---

## Key Outcome

FederatedMed demonstrates that citation-grounded medical QA can be achieved:

- Without centralizing literature  
- Without sharing raw embeddings  
- While preserving verifiability through DOI/PMC references  

If you are working on federated systems, privacy-preserving machine learning, or healthcare RAG, feel free to connect and discuss.

---
