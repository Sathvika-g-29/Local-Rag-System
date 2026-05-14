from __future__ import annotations

import re

import streamlit as st
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import EMBEDDING_MODEL, GENERATION_MODEL, RERANKER_MODEL


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the Hugging Face embedding model once."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_generator():
    """Load the Hugging Face generation model once."""
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL)
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def get_reranker() -> CrossEncoder:
    """Load the cross-encoder reranker once."""
    return CrossEncoder(RERANKER_MODEL)


def source_reference(doc: Document) -> str:
    """Build a readable source label with file, page/sheet, and chunk."""
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page")
    sheet = doc.metadata.get("sheet")
    chunk = doc.metadata.get("chunk")

    parts = [str(source)]
    if page:
        parts.append(f"page {page}")
    if sheet:
        parts.append(f"sheet {sheet}")
    if chunk:
        parts.append(f"chunk {chunk}")
    return " - ".join(parts)


def build_prompt(question: str, retrieved_docs: list[Document]) -> str:
    """Create a strict prompt so the answer stays grounded in context."""
    context = "\n\n".join(
        f"Source: {source_reference(doc)}\n{doc.page_content}"
        for doc in retrieved_docs
    )

    return f"""Answer the question using only the context.
If the answer is not in the context, say: I do not know from the provided documents.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, retrieved_docs: list[Document]) -> str:
    """Generate an answer with the local Hugging Face model."""
    prompt = build_prompt(question, retrieved_docs)
    tokenizer, model = get_generator()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        **inputs,
        max_new_tokens=220,
        num_beams=4,
        do_sample=False,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def generate_short_text(prompt: str, max_new_tokens: int = 80) -> str:
    """Run the local seq2seq model for small utility generations."""
    tokenizer, model = get_generator()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        do_sample=False,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def fallback_answer(retrieved_docs: list[Document]) -> str:
    """Return a simple extractive answer when the small local model returns blank."""
    passages = []
    for doc in retrieved_docs[:2]:
        text = " ".join(doc.page_content.split())
        passages.append(text[:700])

    return "\n\n".join(passages).strip()


def is_low_quality_answer(answer: str, question: str) -> bool:
    """Catch common tiny or empty outputs from small local generation models."""
    normalized = answer.strip().lower()
    if not normalized:
        return True

    if normalized == question.strip().lower():
        return True

    if len(re.sub(r"[^a-z0-9]", "", normalized)) < 8:
        return True

    return False
