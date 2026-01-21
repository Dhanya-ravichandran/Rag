# Retrieval-Augmented Generation (RAG) Application

## Project Overview

This project is a **User-Driven Retrieval-Augmented Generation (RAG) application** that enables users to ask questions based on their own knowledge sources, including **PDF documents** and **YouTube videos**.
The system retrieves relevant information from user-provided data and generates **context-aware, grounded responses** using Large Language Models (LLMs).

The application is designed to reduce hallucinations by ensuring that all responses are strictly based on retrieved context rather than model memory alone.

---

## Key Features

* **User-Provided Knowledge Sources**
  Supports uploading multiple PDF documents and optional YouTube links for building a custom knowledge base.

* **YouTube Transcription Pipeline**
  Automatically downloads and transcribes YouTube audio using Whisper for inclusion in the retrieval process.

* **Semantic Retrieval**
  Uses vector embeddings and ChromaDB to retrieve the most relevant document chunks for each query.

* **Conversational Memory**
  Maintains chat history using LangGraph, enabling follow-up questions and contextual continuity.

* **LLM-Powered Answer Generation**
  Generates concise, context-grounded answers using Groq-hosted LLaMA models.

* **User-Driven Workflow**
  The system validates input and does not respond unless the user provides at least one document or URL.

---

## Architecture Overview

1. **Ingestion Layer**

   * PDF loading using PyPDFLoader
   * YouTube audio extraction via yt-dlp
   * Speech-to-text transcription using Faster-Whisper

2. **Processing Layer**

   * Intelligent text chunking using RecursiveCharacterTextSplitter
   * Embedding generation using HuggingFace sentence-transformer models

3. **Storage Layer**

   * Persistent vector storage using ChromaDB

4. **Retrieval and Generation**

   * Context retrieval through semantic similarity search
   * Answer generation using LLMs grounded in retrieved context

5. **Conversation Memory**

   * Managed with LangGraph and MemorySaver for multi-turn conversations

6. **User Interface**

   * Interactive chat interface built with Streamlit

---

## Technology Stack

* **Programming Language**: Python
* **Frameworks**: LangChain, LangGraph
* **Large Language Models**: Groq (LLaMA family)
* **Embeddings**: HuggingFace Sentence Transformers
* **Vector Database**: ChromaDB
* **Speech-to-Text**: Faster-Whisper
* **Frontend**: Streamlit

---

## Use Cases

* Academic document question answering
* Learning from PDFs and educational videos
* Research paper analysis
* Internal knowledge assistants
* Interview preparation using custom materials

---

## Design Goals

* Ground LLM responses in user-provided data
* Minimize hallucinations
* Support scalable document ingestion
* Enable conversational, memory-aware interactions
* Maintain modular and production-ready code structure

---

## Repository Structure (High-Level)

* `ingest.py` – Document and YouTube ingestion pipeline
* `config.py` – Centralized configuration and constants
* `app.py` – Streamlit application and chat logic
* `requirements.txt` – Project dependencies
* `.env.example` – Environment variable template

---

## Notes

* API keys must be stored in environment variables and never committed to the repository.
* The vector database must be built before querying the system.
* The application supports persistent storage for faster subsequent queries.
