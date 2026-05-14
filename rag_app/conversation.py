from __future__ import annotations

import streamlit as st
from langchain_core.documents import Document

from .models import generate_short_text, is_low_quality_answer
from .utils import important_terms


def initialize_conversation_memory() -> None:
    """Create a lightweight conversation buffer in Streamlit session state."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def conversation_history_text(max_turns: int = 4) -> str:
    """Format recent conversation turns for query reformulation."""
    turns = st.session_state.get("chat_history", [])[-max_turns:]
    return "\n".join(
        f"User: {turn['question']}\nAssistant: {turn['answer']}"
        for turn in turns
    )


def reformulate_question(question: str) -> str:
    """Rewrite follow-up questions into standalone retrieval queries."""
    history = conversation_history_text()
    if not history:
        return question

    prompt = f"""Rewrite the latest user question as a standalone search query.
Use the conversation history only to resolve references.
Return only the rewritten query.

Conversation:
{history}

Latest question: {question}
Standalone query:"""

    rewritten = generate_short_text(prompt, max_new_tokens=80)
    if is_low_quality_answer(rewritten, question):
        return question
    return rewritten


def filter_relevant_docs(question: str, retrieved_docs: list[Document]) -> list[Document]:
    """Keep only chunks that share meaningful terms with the question."""
    query_terms = important_terms(question)
    if not query_terms:
        return retrieved_docs

    relevant_docs = []
    required_overlap = 2 if len(query_terms) >= 3 else 1
    for doc in retrieved_docs:
        chunk_terms = important_terms(doc.page_content)
        overlap = query_terms & chunk_terms
        if len(overlap) >= required_overlap:
            relevant_docs.append(doc)

    return relevant_docs


def is_uploaded_overview_question(question: str) -> bool:
    """Detect broad summary requests about an uploaded document."""
    normalized = question.lower()
    upload_words = {"upload", "uploaded", "file", "document", "pdf"}
    overview_words = {"overview", "summary", "summarize", "explain", "about"}
    terms = important_terms(normalized)
    return bool(terms & upload_words) and bool(terms & overview_words)


def uploaded_overview_chunks(chunks: list[Document], top_k: int) -> list[Document]:
    """Use the beginning chunks of uploaded files for broad overview questions."""
    uploaded_chunks = [
        chunk for chunk in chunks if chunk.metadata.get("origin") == "uploaded"
    ]
    return uploaded_chunks[:top_k]


def confidence_label(confidence: int) -> str:
    """Map a numeric score to a simple qualitative label."""
    if confidence >= 75:
        return "High"
    if confidence >= 45:
        return "Medium"
    return "Low"
