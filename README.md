# Local Hugging Face RAG Demo

This project is a basic local RAG system using Streamlit, Hugging Face models, recursive chunking, PDF support, and FAISS vector search.

## What It Uses

- `Streamlit` for the web UI
- `RecursiveCharacterTextSplitter` for chunking documents
- `sentence-transformers/all-MiniLM-L6-v2` for Hugging Face embeddings
- `FAISS` for local vector search
- `google/flan-t5-small` for local Hugging Face answer generation
- `pypdf` for reading PDF files

## Install

```bash
pip install -r requirements.txt
```

The first run may download Hugging Face models, so it can take a little while.

## Run

```bash
streamlit run build_rag.py
```

Then open the local Streamlit URL shown in the terminal.

## How It Works

1. Loads `.txt`, `.md`, and `.pdf` files from `docs/`.
2. Lets you upload extra `.txt`, `.md`, or `.pdf` files in the UI.
3. Splits documents with recursive chunking.
4. Creates embeddings with a Hugging Face sentence-transformer model.
5. Stores the embeddings in a FAISS vector index.
6. Retrieves the most relevant chunks for your question.
7. Sends the retrieved context to a Hugging Face generation model.

## Files

```text
build_rag.py        Streamlit RAG app
requirements.txt    Python dependencies
docs/               Local knowledge base
```
