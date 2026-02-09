# 📄 PDF Intel-Link

**Faithful Question Answering over Documents using Retrieval-Augmented Generation**

## Overview

PDF Intel-Link is an applied AI system designed to enable **faithful, traceable question answering over unstructured PDF documents**. The system follows a **retrieval-first, hallucination-aware architecture**, ensuring that all generated answers are strictly grounded in the source document.

The project explores the intersection of **Natural Language Understanding, Information Retrieval, and Large Language Models**, with a strong emphasis on **experimental rigor, system reliability, and real-world usability**—key principles aligned with Microsoft Research’s applied AI philosophy.

---

## Motivation

Large Language Models are prone to hallucination when answering questions over long documents. This project investigates how **retrieval-augmented pipelines, prompt constraints, and citation-aware context formatting** can significantly improve answer faithfulness while maintaining usability.

---

## Research Contributions

This project makes the following applied research contributions:

### 1. **Hallucination Mitigation via Prompt-Constrained RAG**

* Designed a **strict prompt protocol** that enforces:

  * Context-only answering
  * Explicit refusal when evidence is absent
* Evaluated zero-temperature inference to reduce generative variance.

**Insight:** Prompt-level constraints combined with retrieval significantly reduce unsupported responses compared to vanilla LLM querying.

---

### 2. **Citation-Aware Context Formatting**

* Introduced **page-number–annotated context injection** during retrieval.
* Enabled traceable answers that map directly back to the original document.

**Insight:** Lightweight metadata grounding improves user trust and answer verifiability without additional model complexity.

---

### 3. **Persistent Vector Store for Repeated Query Efficiency**

* Implemented a **persistent ChromaDB-backed embedding store**, enabling reuse across sessions.
* Reduced redundant embedding computation for large documents.

**Insight:** Persistence improves system efficiency and mirrors production-scale document intelligence systems.

---

### 4. **End-to-End RAG System using LCEL**

* Built the pipeline using **LangChain Expression Language (LCEL)** for modularity and composability.
* Supports clean separation between retrieval, prompting, and generation.

**Insight:** LCEL-based design simplifies experimentation and rapid system iteration.

---

## System Architecture

1. **Document Ingestion**

   * PDF parsing using `PyPDFLoader`
2. **Text Segmentation**

   * Overlapping chunking to preserve semantic continuity
3. **Embedding Generation**

   * Sentence-level semantic embeddings via MiniLM
4. **Vector Indexing**

   * Persistent ChromaDB storage
5. **Context Retrieval**

   * Top-k semantic similarity search
6. **Constrained Generation**

   * Gemini LLM with anti-hallucination prompt
7. **Interactive Interface**

   * Gradio-based UI for rapid experimentation

---

## Technical Stack

* **Language:** Python
* **LLM:** Google Gemini (gemini-2.5-flash)
* **Retrieval Framework:** LangChain (LCEL)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* **Vector Database:** ChromaDB (persistent)
* **UI:** Gradio

---

## Experimental Design Choices

* Zero temperature decoding for determinism
* Chunk overlap to prevent boundary information loss
* Retrieval-first architecture (no free-form LLM answers)
* Explicit fallback response for unanswerable queries

---

## Applications

* Research paper analysis
* Academic and technical document comprehension
* Policy and compliance document querying
* AI-assisted reading with verifiable citations

---

## Limitations & Future Work

* Quantitative faithfulness evaluation (e.g., attribution metrics)
* Hybrid retrieval (BM25 + dense embeddings)
* Cross-document reasoning
* Query re-ranking and reretrieval
* Comparative analysis across LLMs

---

## Author

**Chesta Vashishtha**
B.Tech Undergraduate | Applied Machine Learning & NLP
Interested in Applied AI Research, LLM Systems, and Information Retrieval

---


Just tell me which one to do next 🚀
