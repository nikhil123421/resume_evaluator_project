# Resume Evaluator — https://resumeeeval.vercel.app/

**Evidence-grounded resume evaluation using semantic retrieval, dense embeddings, and structured LLM inference.**

A RAG-style pipeline that decomposes job descriptions into typed requirements, retrieves semantically relevant resume evidence, and scores requirement-level coverage. Every `Supported` verdict has to survive a server-side verbatim-quote check against the original resume text — or it gets downgraded and thrown out.

### Architecture & Internals

- **PDF Intake & Validation** — `%PDF-` signature check, encrypted-PDF rejection, page and character limits enforced before any text is trusted. Extraction via PyPDF.
- **Structured Information Extraction** — Groq-hosted LLM inference converts job descriptions into typed requirement objects (Required / Preferred) and resumes into structured candidate data, constrained by Pydantic schemas.
- **Evidence Chunking** — the *original* extracted PDF text is split into overlapping chunks. Not the LLM's summary. Evidence quotes always come from what the candidate actually wrote.
- **Dense Semantic Retrieval** — a 1024-dimensional embedding model (`jina-embeddings-v5-text-small`), with `retrieval.passage` for resume chunks and `retrieval.query` for requirements. Concurrent embedding calls, cosine similarity, top-K retrieval.
- **Retrieval-Grounded Verification** — retrieved chunks are passed to an LLM for per-requirement assessment. Every `Supported` must include an `evidence_quote` that appears verbatim in the retrieved excerpts; unmatched quotes are rejected server-side and downgraded to `Unverified`.
- **Evidence-Based Classification** — each requirement receives `Supported`, `Unverified`, or `Missing`, with reasoning that has to be traceable to the source text.
- **Weighted Evidence Aggregation** — Required = 2 points, Preferred = 1. Only `Supported` counts toward the numerator. Failed verifications stay in the denominator. No silent drops. Incomplete runs produce conservative scores, on purpose.
- **Prompt & Input-Safety Engineering** — untrusted-document handling, injection-resistant extraction and scoring prompts, explicit policy on LLM-product-API vs AI-assisted-development, disjunctive/conjunctive/example-based requirement logic.
- **Hybrid AI Architecture** — probabilistic LLM judgments wrapped in deterministic application logic, typed data contracts, and a modular Python service layer.

### Stack

`Python` · `FastAPI` · `Pydantic` · `Groq API` · `jina-embeddings-v5-text-small` · `PyPDF` · `NumPy` · `httpx` · `React` · `TypeScript` · `Tailwind CSS`

### Core Principle

**Semantic similarity is a retrieval signal, not proof of competence.**
**A quote that can't be found in the source text isn't evidence, it's a hallucination with good posture.**
