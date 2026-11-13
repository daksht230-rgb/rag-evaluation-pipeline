# Production-Grade RAG Evaluation Pipeline

A high-performance Python framework engineered to evaluate, benchmark, and score Retrieval-Augmented Generation (RAG) pipelines at scale. Designed to measure context relevance precision, hallucination risks, and generation faithfulness against retrieved knowledge bases.

## Architecture & Core Features
- **Context Relevance Scoring:** Automated token-level similarity evaluation comparing user queries against retrieved vector contexts.
- **Faithfulness & Hallucination Auditing:** Heuristic verification engines ensuring generated model responses are strictly grounded in retrieved documents.
- **RESTful Benchmarking Pipeline:** Built with FastAPI and Pydantic for asynchronous, high-throughput RAG evaluation runs.

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python main.py
