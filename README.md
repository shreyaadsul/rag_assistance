# RAG Observability System

A research-grade Multi-Document Retrieval-Augmented Generation (RAG) platform built with Flask, FAISS, LangChain, and Groq LLMs.
This system focuses not only on document question-answering, but also on retrieval transparency, evaluation observability, confidence analysis, and grounded response verification.

---

# Overview

The RAG Observability System enables users to:

* Upload and index multiple PDF documents
* Generate embeddings and persistent FAISS vector stores
* Query across all documents or a specific document
* Retrieve grounded contextual chunks
* Analyze retrieval quality and confidence metrics
* Evaluate hallucination risk and synthesis correctness
* Visualize the internal lifecycle of the RAG pipeline

The platform was designed with a research-oriented architecture emphasizing modularity, observability, and explainability.

---

# Core Features

## Multi-Document RAG Retrieval

* Upload multiple PDF documents simultaneously
* Unified FAISS vector database
* Cross-document semantic retrieval
* Document-specific filtering using metadata

## Retrieval Observability

* Chunk-level retrieval tracking
* Relevance score visualization
* Source attribution
* Page-level grounding transparency

## Evaluation Layer

* Confidence scoring
* Correctness analysis
* Completeness estimation
* Hallucination risk detection

## Analysis Dashboard

* RAG lifecycle telemetry visualization
* Retrieval confidence analytics
* Pipeline operational metrics
* Observability graphing

## Persistent Storage

* Persistent FAISS vector indices
* Local metadata tracking
* Stored uploaded documents
* Survives server restarts

## Modern Research Dashboard UI

* Dark research-grade interface
* Modular observability tabs
* Cross-document retrieval selector
* Structured document manager

---

# Tech Stack

## Backend

* Flask
* LangChain
* FAISS
* SentenceTransformers
* Groq API

## Frontend

* HTML
* CSS
* JavaScript

## Embedding Model

* all-MiniLM-L6-v2

## LLM

* Groq Llama-3 API

---

# System Architecture

```text
PDF Upload
     ↓
Document Loader
     ↓
Chunking Pipeline
     ↓
Embedding Generation
     ↓
Unified FAISS Vector Store
     ↓
Retriever Orchestrator
     ↓
LLM Synthesis
     ↓
Evaluation Engine
     ↓
Observability Dashboard
```

---

# Project Structure

```text
backend/
│
├── ingestion.py
├── retriever.py
├── evaluator.py
├── vector_manager.py
├── metadata_manager.py
├── llm_manager.py
│
templates/
│
├── index.html
│
static/
│
├── style.css
├── script.js
│
data/
│
├── vector_store/
├── uploads/
├── metadata.json
│
app.py
requirements.txt
README.md
```

---

# Observability Components

## Overview Tab

Displays:

* grounded synthesized response
* confidence metrics
* retrieval grounding status
* contextual source tracking

## Retrieval Tab

Displays:

* retrieved chunks
* source document names
* page references
* relevance telemetry
* chunk observability

## Evaluation Tab

Displays:

* correctness score
* hallucination probability
* completeness analysis
* confidence estimation

## Analysis Tab

Displays:

* lifecycle observability graph
* retrieval analytics
* pipeline execution metrics
* synthesis telemetry

---

# Research-Oriented Design Goals

This project was designed to explore:

* Retrieval transparency in RAG systems
* Grounded answer synthesis
* Confidence-aware generation
* Retrieval observability
* Hallucination analysis
* Multi-document semantic orchestration

The goal was not only to build a chatbot, but to create an inspectable and research-grade RAG pipeline.

---

# Future Scope

Potential future improvements include:

* Hybrid search (BM25 + Vector Retrieval)
* Reranking pipelines
* Advanced chunk scoring normalization
* Streaming responses
* Agentic retrieval orchestration
* Knowledge graph integration
* Research evaluation benchmarking
* Docker deployment

---

# Installation

## Clone Repository

```bash
git clone <your-repository-link>
cd rag-observability-system
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
python app.py
```

---

# Screenshots

Add screenshots here:

* Dashboard Overview
* Retrieval Observability
* Evaluation Metrics
* Multi-Document Retrieval
* Analysis Graphs

---

# Author

Shreya Adsul

B.Sc. AIML Student
Research-focused AI/ML Developer
Interested in Retrieval-Augmented Generation (RAG), AI Systems, and Explainable AI

---

# License

This project is intended for educational, research, and portfolio purposes.
