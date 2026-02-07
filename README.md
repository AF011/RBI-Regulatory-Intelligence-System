# RBI Regulatory Intelligence System README

---
title: "RBI Regulatory Intelligence System"
description: "AI-powered research assistant for querying RBI Master Circulars with hybrid retrieval, LLM answers, and PDF viewer."
tech_stack:
  - Python 3.10+
  - Streamlit
  - LangChain
  - Ollama LLM (Qwen2.5)
  - Ollama Embeddings (nomic-embed-text)
  - ChromaDB (vector storage)
  - Sentence Transformers (Cross-Encoder MS-MARCO MiniLM)
  - BM25 lexical retrieval

overview: |
  The RBI Regulatory Intelligence System is a document-grounded AI assistant designed
  to provide answers to queries about RBI Master Circulars. It uses a hybrid
  retrieval approach combining BM25 lexical search and semantic embeddings, and
  leverages an LLM for strictly document-grounded answers. Users can interact with
  the system via a Streamlit chat interface and view PDF sources with page-level highlights.

features:
  - "Hybrid retrieval using BM25 and vector similarity search"
  - "Reranking of retrieved documents using Cross-Encoder (MS-MARCO MiniLM)"
  - "LLM responses strictly based on the provided document context"
  - "Interactive chat interface for querying RBI Master Circulars"
  - "Page-level PDF viewer with highlighted relevant text"
  - "Automatic citation of sources for all factual information"

installation:
  prerequisites:
    - "Python 3.10 or higher"
    - "Ollama installed and running locally"
      steps:
        - "ollama serve"
        - "ollama pull nomic-embed-text"
        - "ollama pull qwen2.5:1.5b"
  using_poetry: |
    git clone <repository_url>
    cd <repository_directory>
    poetry install
    poetry run streamlit run app.py
  using_pip: |
    git clone <repository_url>
    cd <repository_directory>
    pip install -r requirements.txt
    streamlit run app.py

usage: |
  1. Launch the Streamlit application in your browser.
  2. Adjust retrieval and rerank parameters in the sidebar as needed.
  3. Enter queries related to RBI Master Circulars in the chat input.
  4. View the assistant's answer along with source citations and highlighted text.
  5. You can download RBI Master Circulars for testing from the official RBI website: [RBI Master Circulars](https://rbi.org.in/Scripts/BS_ViewMasterCirculardetails.aspx)

folder_structure:
  - main.py: "Streamlit application"
  - data/pdfs/: "Directory containing PDF master circulars"
  - notebook/chroma_db/: "ChromaDB database for embeddings"
  - requirements.txt: "Python dependencies for pip"
  - pyproject.toml: "Poetry project file"
  - README.md: "Project documentation"

system_components:
  hybrid_retriever: "Combines BM25 lexical search and semantic vector embeddings"
  reranker: "Cross-Encoder (MS-MARCO MiniLM) for document relevance scoring"
  llm: "Qwen2.5 LLM via Ollama for context-grounded responses"
  chromadb: "Vector database storing embeddings for semantic retrieval"
  streamlit_interface: "Web-based chat interface with PDF viewing"

known_issues:
  - "Queries unrelated to indexed documents may return no results"
  - "System performance depends on ChromaDB size and system resources"
  - "Ollama must be running locally for embeddings and LLM inference"

connection_options: |
  The system can connect to:
    - Local Ollama instance for embeddings and LLM inference
    - ChromaDB vector database for semantic retrieval
    - BM25 lexical search for keyword-based retrieval
    - Cross-Encoder reranker for relevance scoring

license: "Apache 2.0"

signature: "An AF's Code Written with 💡 and 🥤"