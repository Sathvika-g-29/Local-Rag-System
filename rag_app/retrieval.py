from __future__ import annotations

import json
import math

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .config import (
    CHROMA_COLLECTION_PREFIX,
    EMBEDDING_MODEL,
    RETRIEVAL_VERSION,
    VECTOR_DB_DIR,
    VECTOR_DB_META,
)
from .documents import serialize_chunks
from .models import get_embeddings, get_reranker
from .utils import hash_text, important_terms


@st.cache_resource(show_spinner=False)
def get_bm25_retriever(
    chunk_key: str,
    serialized_chunks: tuple[tuple[str, str, int | None, str, str, int | None], ...],
) -> BM25Retriever:
    """Build the BM25 retriever once per chunk set instead of once per question."""
    chunks = [
        Document(
            page_content=text,
            metadata={
                "source": source,
                "origin": origin,
                **({"sheet": sheet} if sheet else {}),
                **({"page": page} if page else {}),
                **({"chunk": chunk} if chunk else {}),
            },
        )
        for source, text, page, origin, sheet, chunk in serialized_chunks
    ]
    return BM25Retriever.from_documents(chunks)


def build_vector_store(chunks: list[Document], index_key: str) -> Chroma:
    """Create or reuse a persistent Chroma vector database."""
    collection_name = f"{CHROMA_COLLECTION_PREFIX}_{index_key[:12]}"

    if VECTOR_DB_DIR.exists() and VECTOR_DB_META.exists():
        metadata = json.loads(VECTOR_DB_META.read_text(encoding="utf-8"))
        if (
            metadata.get("database") == "chroma"
            and metadata.get("index_key") == index_key
        ):
            return Chroma(
                collection_name=metadata.get("collection", collection_name),
                embedding_function=get_embeddings(),
                persist_directory=str(VECTOR_DB_DIR),
            )

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=str(VECTOR_DB_DIR),
    )


def save_vector_store(index_key: str) -> None:
    """Save metadata so repeated questions can reuse the Chroma database."""
    collection_name = f"{CHROMA_COLLECTION_PREFIX}_{index_key[:12]}"
    VECTOR_DB_META.parent.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_META.write_text(
        json.dumps(
            {
                "index_key": index_key,
                "collection": collection_name,
                "embedding_model": EMBEDDING_MODEL,
                "database": "chroma",
                "retrieval_version": RETRIEVAL_VERSION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def doc_identity(doc: Document) -> str:
    """Stable identity for deduplicating results from multiple retrievers."""
    metadata = doc.metadata
    return "|".join(
        [
            str(metadata.get("source", "")),
            str(metadata.get("page", "")),
            str(metadata.get("sheet", "")),
            str(metadata.get("chunk", "")),
            hash_text(doc.page_content[:500]),
        ]
    )


def build_bm25_retriever(chunks: list[Document], top_k: int) -> BM25Retriever:
    """Create a true BM25 keyword retriever over the current chunks."""
    serialized_chunks = serialize_chunks(chunks)
    chunk_key = hash_text(json.dumps(serialized_chunks, ensure_ascii=False))
    retriever = get_bm25_retriever(chunk_key, serialized_chunks)
    retriever.k = top_k
    return retriever


def invoke_retriever(retriever, query: str) -> list[Document]:
    """Support both old and new LangChain retriever APIs."""
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    return retriever.get_relevant_documents(query)


def ensemble_documents(
    semantic_docs: list[Document],
    bm25_docs: list[Document],
    semantic_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[Document]:
    """Combine semantic and BM25 rankings using weighted reciprocal rank fusion."""
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for docs, weight in ((semantic_docs, semantic_weight), (bm25_docs, bm25_weight)):
        for rank, doc in enumerate(docs, start=1):
            key = doc_identity(doc)
            documents[key] = doc
            scores[key] = scores.get(key, 0.0) + weight / (rank + 60)

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [documents[key] for key in ranked_keys]


def hybrid_retrieve(
    question: str,
    vector_store: Chroma,
    chunks: list[Document],
    top_k: int,
    candidate_count: int,
) -> list[Document]:
    """Retrieve with semantic search plus BM25 keyword search, then ensemble them."""
    search_k = max(candidate_count, top_k)
    semantic_docs = vector_store.similarity_search(question, k=search_k)
    try:
        bm25_retriever = build_bm25_retriever(chunks, top_k=search_k)
        bm25_docs = invoke_retriever(bm25_retriever, question)
    except Exception:
        bm25_docs = []
    return ensemble_documents(semantic_docs, bm25_docs)[:search_k]


def rerank_documents(
    question: str,
    documents: list[Document],
    top_k: int,
) -> list[tuple[Document, float]]:
    """Use a cross-encoder to reorder retrieved chunks by direct relevance."""
    if not documents:
        return []

    try:
        reranker = get_reranker()
        pairs = [(question, doc.page_content) for doc in documents]
        scores = reranker.predict(pairs)
    except Exception:
        return [(doc, 0.0) for doc in documents[:top_k]]

    ranked = sorted(
        zip(documents, [float(score) for score in scores]),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:top_k]


def confidence_score(reranked_docs: list[tuple[Document, float]], question: str) -> int:
    """Estimate answer confidence from retrieval evidence, score margin, and coverage."""
    if not reranked_docs:
        return 0

    scores = [score for _doc, score in reranked_docs]
    top_score = max(-20.0, min(20.0, scores[0]))
    reranker_confidence = 1 / (1 + math.exp(-top_score))
    second_score = scores[1] if len(scores) > 1 else scores[0]
    margin = max(0.0, min(1.0, (scores[0] - second_score) / 5.0))
    query_terms = important_terms(question)
    supporting_docs = 0
    for doc, _score in reranked_docs:
        doc_terms = important_terms(doc.page_content)
        if query_terms & doc_terms:
            supporting_docs += 1
    top_terms = important_terms(reranked_docs[0][0].page_content)
    coverage = len(query_terms & top_terms) / max(len(query_terms), 1)
    support = supporting_docs / max(len(reranked_docs), 1)
    combined = (
        0.45 * reranker_confidence
        + 0.2 * margin
        + 0.2 * coverage
        + 0.15 * support
    )
    return max(0, min(100, round(combined * 100)))
