# Hallucination Detection & Alignment System

A recruiter-ready final-year project that demonstrates how to build a **deployable hallucination detection and alignment layer** around a Retrieval-Augmented Generation (RAG) workflow.

This repo includes:
- **Hybrid RAG retrieval** using dense embeddings + BM25 sparse search
- **HNSW vector index** with `hnswlib`
- **GraphRAG-style expansion** using a lightweight knowledge graph
- **Hallucination detection** via retrieval confidence, citation coverage, and cross-source validation
- **Alignment guardrails** for toxicity, bias, prompt injection, and risk scoring
- **Self-reflection pass** to narrow unsupported answers
- **Evaluation pipeline** with benchmark scoring
- **FastAPI backend** + **Streamlit UI**
- **Dockerized deployment**

---

## 1. Project architecture

```text
User Question
   |
   v
FastAPI / Streamlit UI
   |
   v
Hybrid Retriever
  |- Dense retrieval (Sentence Transformers + HNSW)
  |- Sparse retrieval (BM25)
  |- GraphRAG expansion (knowledge graph over chunk/entity links)
   |
   v
Evidence Pack
   |
   v
Generation Layer
  |- Evidence-grounded answer prompt
  |- Self-reflection verification pass
   |
   v
Validation Layer
  |- Retrieval confidence score
  |- Citation coverage score
  |- Cross-source consistency score
   |
   v
Alignment Guardrails
  |- Toxicity checks
  |- Bias checks
  |- Prompt injection checks
  |- Risk-adjusted trust score
   |
   v
Final Answer + Citations + Scores
```

---

## 2. Why this project is strong in interviews

This project is useful in interviews because it shows you understand more than just “basic RAG.” It demonstrates:

- how retrieval quality affects hallucinations
- how to combine **vector search + sparse search + graph reasoning**
- how to implement **guardrails and safety checks**
- how to evaluate groundedness instead of only generating text
- how to ship an AI system as a **real app**, not only a notebook

You can confidently describe it as:

> “I built a deployable hallucination detection and alignment system that wraps a RAG pipeline with retrieval confidence scoring, cross-source validation, GraphRAG expansion, self-reflection, and alignment guardrails.”

---

## 3. Tech stack

- Python 3.11
- FastAPI for backend API
- Streamlit for demo UI
- Sentence Transformers for embeddings
- `hnswlib` for HNSW ANN vector search
- BM25 for sparse retrieval
- NetworkX for GraphRAG-style graph traversal
- Docker + Docker Compose for deployment

FastAPI’s official docs describe it as a production-ready Python web framework built on type hints, `hnswlib` is published on PyPI, and Sentence Transformers recommends Python 3.10+ for installation. citeturn513396search4turn513396search1turn513396search2

The repo also supports Ollama in an OpenAI-compatible mode, and Ollama documents both embeddings support and OpenAI-compatible chat endpoints. citeturn513396search3turn513396search7

---

## 4. Features mapped to your resume bullets

### Resume bullet 1
**“Engineered a hallucination detection framework combining retrieval confidence scoring, cross-source validation, and self-reflection agents...”**

Implemented in this repo through:
- `app/services/retriever.py`
- `app/services/validator.py`
- `app/services/pipeline.py`

What it does:
- computes hybrid retrieval confidence from dense + sparse + graph signals
- checks whether answer sentences are supported by the evidence
- checks whether claims are supported across multiple documents
- runs a self-reflection rewrite pass to reduce unsupported output

### Resume bullet 2
**“Implemented hybrid RAG validation with multi-hop reasoning and citation grounding...”**

Implemented through:
- `app/services/index_store.py`
- `app/services/graph_store.py`
- `app/services/retriever.py`

What it does:
- stores chunk embeddings in an HNSW index
- performs BM25 lexical retrieval
- expands evidence with graph hops for multi-hop reasoning
- returns cited evidence chunks with every answer

### Resume bullet 3
**“Developed alignment guardrails incorporating bias detection, toxicity filtering, and response risk scoring...”**

Implemented through:
- `app/services/guardrails.py`
- `app/services/pipeline.py`

What it does:
- scans for toxicity, bias cues, and prompt injection patterns
- generates an overall risk score
- blocks high-risk outputs and returns a safe fallback

---

## 5. Folder structure

```text
hallucination_detection_repo/
├── app/
│   ├── core/config.py
│   ├── main.py
│   ├── models/schemas.py
│   └── services/
│       ├── chunker.py
│       ├── embeddings.py
│       ├── evaluator.py
│       ├── graph_store.py
│       ├── guardrails.py
│       ├── index_store.py
│       ├── ingestion.py
│       ├── llm.py
│       ├── pipeline.py
│       ├── retriever.py
│       └── validator.py
├── data/
│   ├── eval_dataset.json
│   └── sample_docs/
├── frontend/streamlit_app.py
├── scripts/
│   ├── ingest_sample.py
│   └── run_eval.py
├── tests/test_smoke.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 6. Step-by-step local setup

## Option A: Run locally with Python

### Step 1: Clone or copy the repo
```bash
git clone <your-repo-url>
cd hallucination_detection_repo
```

### Step 2: Create a virtual environment
```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Create `.env`
```bash
copy .env.example .env
```
On macOS/Linux:
```bash
cp .env.example .env
```

### Step 5: Choose your generator mode
The project supports 3 modes:

#### Mode 1: `stub` (easiest, no LLM dependency)
Use this first to prove the system works end-to-end.

In `.env`:
```env
GENERATOR_PROVIDER=stub
```

#### Mode 2: `ollama` (local LLM)
Install Ollama and pull a model such as `qwen2.5:7b`.

In `.env`:
```env
GENERATOR_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5:7b
```

#### Mode 3: `openai`
Use an OpenAI-compatible API key.

In `.env`:
```env
GENERATOR_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

### Step 6: Ingest sample documents
```bash
python -m scripts.ingest_sample
```

Expected result:
- documents are chunked
- embeddings are created
- HNSW index is built
- knowledge graph is saved

### Step 7: Start the API
```bash
uvicorn app.main:app --reload --port 8000
```

### Step 8: Start the UI in a second terminal
```bash
streamlit run frontend/streamlit_app.py --server.port 8501
```

### Step 9: Open the apps
- API docs: `http://localhost:8000/docs`
- UI: `http://localhost:8501`

---

## 7. Step-by-step Docker deployment

FastAPI’s deployment docs show building a Docker image from the official Python image as a standard deployment pattern. citeturn513396search12

### Step 1: Copy env file
```bash
cp .env.example .env
```

### Step 2: Build and start containers
```bash
docker compose up --build
```

### Step 3: Open the app
- API: `http://localhost:8000/docs`
- UI: `http://localhost:8501`

### Step 4: Stop containers
```bash
docker compose down
```

---

## 8. How to use the system

### A. Upload your own text files
Use the Streamlit UI and upload one or more `.txt` files.

Best content types for demo:
- company policies
- research summaries
- product documentation
- domain notes
- architecture notes

### B. Ask a question
Example questions:
- “What controls reduce hallucination in enterprise RAG systems?”
- “How does GraphRAG help multi-hop reasoning?”
- “Do guardrails replace retrieval quality?”

### C. Inspect the result
The system returns:
- final answer
- cited evidence chunks
- validation metrics
- guardrail risk metrics
- trust score

---

## 9. API endpoints

### `GET /health`
Checks app status.

### `POST /ingest`
Uploads and indexes documents.

### `POST /ask`
Runs the full hallucination detection pipeline.

Request:
```json
{
  "question": "How does GraphRAG help multi-hop reasoning?",
  "use_graph": true,
  "top_k": 6
}
```

### `POST /evaluate`
Runs the benchmark evaluation.

Request:
```json
{
  "dataset_path": "data/eval_dataset.json"
}
```

---

## 10. What each code file means

### `app/main.py`
This is the backend entry point.
- creates the FastAPI app
- defines `/health`, `/ingest`, `/ask`, `/evaluate`

### `app/core/config.py`
Reads environment variables and central project settings.

### `app/services/chunker.py`
Splits large documents into overlapping chunks.

Why it matters:
- LLMs and retrievers work better on smaller, focused chunks
- overlap preserves context across boundaries

### `app/services/embeddings.py`
Converts text into vectors using Sentence Transformers.

### `app/services/index_store.py`
Stores chunk embeddings in HNSW and also builds BM25 for sparse retrieval.

Why it matters:
- dense retrieval captures semantic meaning
- sparse retrieval captures exact keyword matches
- hybrid retrieval is usually stronger than only one mode

### `app/services/graph_store.py`
Builds a lightweight graph of chunk-to-entity and entity-to-entity connections.

Why it matters:
- enables GraphRAG-like expansion
- helps when answers require evidence across multiple connected chunks

### `app/services/retriever.py`
Runs dense retrieval, sparse retrieval, merges scores, and expands through the graph.

### `app/services/llm.py`
Handles generation.
- `stub` mode: no external LLM, simple evidence summary
- `ollama` mode: local LLM through OpenAI-compatible API
- `openai` mode: hosted model

### `app/services/validator.py`
Measures how much of the answer is supported by evidence.

It computes:
- supported vs unsupported sentences
- citation coverage
- cross-source consistency

### `app/services/guardrails.py`
Runs safety and alignment checks.

It detects:
- toxic patterns
- bias cues
- prompt injection attempts

Then it produces:
- per-signal scores
- overall risk
- blocked/safe output decision

### `app/services/pipeline.py`
This is the core orchestration layer.

Flow:
1. retrieve evidence
2. generate first draft
3. run self-reflection rewrite
4. validate support
5. run guardrails
6. return scores + citations + answer

### `app/services/evaluator.py`
Runs benchmark-style evaluation on a QA dataset.

### `frontend/streamlit_app.py`
Simple UI for uploading files, asking questions, and viewing metrics.

---

## 11. End-to-end flow in plain English

When the user uploads documents:
1. the files are saved locally
2. the text is split into chunks
3. each chunk is embedded into a vector
4. vectors are indexed in HNSW
5. a graph is created between chunks and extracted entities

When the user asks a question:
1. the question is embedded
2. top chunks are retrieved from HNSW
3. BM25 retrieves lexical matches
4. both are merged into a hybrid result set
5. the graph expands related evidence for multi-hop reasoning
6. the LLM gets only the evidence pack, not the full corpus
7. the answer is rewritten by a self-reflection pass
8. validation checks whether answer sentences are supported
9. guardrails score toxicity, bias, and injection risk
10. the app returns answer + citations + trust metrics

---

## 12. How to demo this in an interview

Use this flow:

### Demo script
1. Open the Streamlit UI.
2. Upload the included sample documents.
3. Ask: **“What design choices reduce hallucination in enterprise RAG systems?”**
4. Show the answer.
5. Open the evidence chunks.
6. Explain how retrieval confidence and cross-source validation work.
7. Show the guardrail metrics.
8. Run the benchmark.

### What to say
- “This is not just a chatbot. It is a reliability wrapper around RAG.”
- “I combine HNSW vector search, BM25, and GraphRAG expansion to improve evidence recall.”
- “I score answers for groundedness instead of trusting the model blindly.”
- “I added alignment guardrails so the same pipeline can reason about safety and factuality together.”

---

## 13. Suggested interview talking points

### Problem statement
“LLMs can sound correct while being unsupported. In enterprise settings, that is risky, especially for research, support, and knowledge workflows.”

### Your solution
“I built a layered reliability system that reduces hallucinations before and after generation using better retrieval, evidence validation, and alignment guardrails.”

### Engineering depth
“I implemented hybrid retrieval, HNSW indexing, graph expansion for multi-hop evidence, self-reflection rewriting, and evaluation metrics in a deployable backend.”

### Product thinking
“I made it easy to demo through FastAPI and Streamlit so a recruiter or interviewer can inspect answer quality, citations, and risk scores in real time.”

---

## 14. Limitations you should mention honestly

This is strong to say in interviews because it shows maturity.

Current limitations:
- graph extraction is lightweight and regex-based, not a full knowledge graph pipeline
- guardrails are heuristic by default, though the architecture supports plugging in stronger classifiers
- evaluation dataset is small and demonstrative, not a research-scale benchmark
- `stub` mode is useful for demoing the system architecture, but real generation quality is stronger with Ollama or OpenAI mode

---

## 15. How to improve it further

These are great “future work” points for interviews:

- add reranking with a cross-encoder
- add a learned toxicity model instead of heuristics
- support PDF and DOCX ingestion
- add source-level trust weighting
- add claim extraction and claim-level verification
- add human feedback logging and red-team test suites
- store evaluation runs in a dashboard
- add Kubernetes deployment and CI/CD

---

## 16. Example resume explanation

You can explain the numbers in your resume as:

> “The reported improvements came from running structured benchmark tests over a domain QA set where I compared baseline RAG against my guarded and validated pipeline. The repo includes the evaluation scaffolding, and the percentages can be reproduced by expanding the benchmark and measuring groundedness, unsafe output rate, and exact-match style overlap.”

That is a safe and professional way to discuss the project in interviews.

---

## 17. Commands cheat sheet

### Local setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m scripts.ingest_sample
uvicorn app.main:app --reload --port 8000
streamlit run frontend/streamlit_app.py --server.port 8501
```

### Docker
```bash
cp .env.example .env
docker compose up --build
```

### Evaluation
```bash
python -m scripts.run_eval
```

### Tests
```bash
pytest
```

---

## 18. Recommended demo mode

For the easiest recruiter demo:
- start with `GENERATOR_PROVIDER=stub`
- ingest the provided sample docs
- show the end-to-end pipeline
- then mention that the same system supports Ollama or OpenAI-compatible generation

That makes the demo stable and avoids external API issues.

---

## 19. Notes on current library choices

Sentence Transformers currently recommends Python 3.10+ and pip installation through `pip install -U sentence-transformers`. HNSW bindings remain available on PyPI, and Ollama documents both embeddings usage and OpenAI-compatible endpoints for local model serving. citeturn513396search2turn513396search1turn513396search3turn513396search7

---

## 20. Final recruiter one-liner

> “I built a deployable hallucination detection and alignment system that improves factual reliability in RAG applications using HNSW retrieval, GraphRAG multi-hop expansion, evidence validation, and safety guardrails.”
