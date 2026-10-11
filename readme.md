# Resume Evaluator - https://resumeeeval.vercel.app/

**Evidence grounded resume evaluation using semantic retrieval, dense embeddings, and structured LLM inference.**

A RAG-style application that decomposes job descriptions into structured requirements, retrieves semantically relevant resume evidence, and evaluates requirement level coverage.

### Architecture & Internals

- **Document Ingestion & Chunking** — PDF text extraction with PyPDF, followed by chunking to create smaller, retrievable evidence units.
- **Dense Semantic Retrieval** — `BAAI/bge-base-en-v1.5` via Sentence Transformers, with embedding normalization, vector representations, cosine similarity, and top-*k* retrieval to surface contextually relevant resume excerpts.
- **Structured Information Extraction** — Groq-hosted LLM inference converts unstructured job descriptions into typed requirement objects, constrained by Pydantic schemas.
- **Retrieval-Grounded Evaluation** — retrieved evidence is passed to an LLM for requirement level assessment, reducing reliance on keyword overlap and unsupported inference.
- **Evidence-Based Classification** — each requirement receives a `Supported`, `Unverified`, or `Missing` status, with evidence quotes intended to make the assessment traceable to the source text.
- **Weighted Evidence Aggregation** — requirement level results are combined according to their assigned importance to calculate weighted evidence coverage, rather than a hiring probability score.
- **Prompt & Input-Safety Engineering** — explicit evaluation policies, untrusted-document handling, and adversarial test cases designed to resist prompt injection.
- **Hybrid AI Architecture** — probabilistic LLM judgments combined with deterministic application logic, typed data contracts, and a modular Python service layer.

### Stack

`Python` · `FastAPI` · `Pydantic` · `Groq API` · `Sentence Transformers` · `BAAI/bge-base-en-v1.5` · `PyPDF` · `React` · `TypeScript` · `Tailwind CSS`

### Core Principle

**Semantic similarity is a retrieval signal, not proof of competence.**

The system separates candidate evidence retrieval from requirement satisfaction, distinguishes missing information from demonstrated absence, and makes uncertainty visible instead of disguising it as a precise-looking score.

*An evidence first approach to resume evaluation, not another keyword matcher, and not a hiring oracle.*
