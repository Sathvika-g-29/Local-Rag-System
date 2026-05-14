"""
Streamlit + Hugging Face local RAG demo.

Run:
    streamlit run app.py

This app keeps the RAG pipeline intentionally small and readable:
1. Load local text/PDF files from docs/ and optional uploaded files.
2. Split text with RecursiveCharacterTextSplitter.
3. Convert chunks into Hugging Face sentence embeddings.
4. Store/search the embeddings with ChromaDB.
5. Ask a Hugging Face model to answer using only retrieved context.
"""

from __future__ import annotations

import streamlit as st

from rag_app.config import DOCS_DIR, EMBEDDING_MODEL, RETRIEVAL_VERSION, UPLOADED_DOCS_DIR
from rag_app.conversation import (
    confidence_label,
    filter_relevant_docs,
    initialize_conversation_memory,
    is_uploaded_overview_question,
    reformulate_question,
    uploaded_overview_chunks,
)
from rag_app.documents import (
    document_signature,
    load_documents_folder,
    save_uploaded_files,
    serialize_documents,
    split_documents_cached,
)
from rag_app.models import fallback_answer, generate_answer, is_low_quality_answer
from rag_app.retrieval import build_vector_store, confidence_score, hybrid_retrieve, rerank_documents, save_vector_store
from rag_app.ui import add_compact_styles, show_retrieved_chunks
from rag_app.utils import hash_text


def main() -> None:
    st.set_page_config(page_title="Local Hugging Face RAG", page_icon="RAG")
    initialize_conversation_memory()
    add_compact_styles()
    st.title("Local Hugging Face RAG")

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Retrieved chunks", min_value=1, max_value=6, value=3)
        use_reranker = st.checkbox("Use cross-encoder reranker", value=False)
        rerank_candidates = st.slider(
            "Candidate chunks",
            min_value=4,
            max_value=20,
            value=8,
            step=2,
        )
        chunk_size = st.slider("Chunk size", min_value=200, max_value=1200, value=500, step=50)
        chunk_overlap = st.slider(
            "Chunk overlap",
            min_value=0,
            max_value=300,
            value=100,
            step=25,
        )
        if st.button("Clear chat memory"):
            st.session_state.chat_history = []
            st.rerun()

        if st.session_state.chat_history:
            with st.expander("Conversation memory", expanded=False):
                for turn in st.session_state.chat_history[-4:]:
                    st.markdown(f"**Q:** {turn['question']}")
                    st.markdown(f"**A:** {turn['answer'][:250]}")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=None,
        accept_multiple_files=True,
    )

    question = st.text_input(
        "Ask a question",
        value="What are the main steps in a RAG system?",
    )

    save_uploaded_files(uploaded_files)
    folder_docs = load_documents_folder(DOCS_DIR, origin="local")
    uploaded_docs = load_documents_folder(UPLOADED_DOCS_DIR, origin="uploaded")
    documents = folder_docs + uploaded_docs

    if not documents:
        st.warning("Add files to docs/ or upload documents above.")
        return

    document_key = document_signature(documents)
    serialized_documents = serialize_documents(documents)
    chunks = split_documents_cached(
        document_key,
        serialized_documents,
        chunk_size,
        chunk_overlap,
    )
    index_key = hash_text(
        f"{document_key}|{chunk_size}|{chunk_overlap}|{EMBEDDING_MODEL}|{RETRIEVAL_VERSION}"
    )

    if st.button("Build index and answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        retrieval_question = question
        if st.session_state.chat_history:
            with st.spinner("Rewriting follow-up question..."):
                retrieval_question = reformulate_question(question)

        if retrieval_question != question:
            st.caption(f"Search query: {retrieval_question}")

        with st.spinner("Opening vector index..."):
            vector_store = build_vector_store(chunks, index_key)
            save_vector_store(index_key)

        with st.spinner("Running hybrid retrieval..."):
            if is_uploaded_overview_question(retrieval_question):
                retrieved_docs = uploaded_overview_chunks(chunks, top_k)
                reranked_docs = [(doc, 0.0) for doc in retrieved_docs]
                candidate_docs = retrieved_docs
            else:
                candidate_docs = hybrid_retrieve(
                    retrieval_question,
                    vector_store,
                    chunks,
                    top_k,
                    rerank_candidates,
                )
                filtered_docs = filter_relevant_docs(retrieval_question, candidate_docs)
                if filtered_docs:
                    candidate_docs = filtered_docs
                retrieved_docs = candidate_docs[:top_k]
                reranked_docs = [(doc, 0.0) for doc in retrieved_docs]

        if retrieved_docs and use_reranker and not is_uploaded_overview_question(retrieval_question):
            with st.spinner("Reranking retrieved chunks..."):
                reranked_docs = rerank_documents(
                    retrieval_question,
                    candidate_docs[:rerank_candidates],
                    top_k,
                )
                retrieved_docs = [doc for doc, _score in reranked_docs]

        if not retrieved_docs:
            st.warning(
                "I could not find relevant information in the saved or uploaded documents. Add a document that contains this topic, then build the index again."
            )
            st.stop()

        confidence = confidence_score(reranked_docs, retrieval_question)
        st.metric("Answer confidence estimate", f"{confidence}%", confidence_label(confidence))

        show_retrieved_chunks(retrieved_docs)

        with st.spinner("Generating answer with Hugging Face..."):
            answer = generate_answer(retrieval_question, retrieved_docs)

        st.subheader("Answer")
        if is_low_quality_answer(answer, retrieval_question):
            answer = fallback_answer(retrieved_docs)
            st.info("The local generation model returned a weak answer, so showing the most relevant extracted text instead.")

        if not answer:
            st.warning("No answer could be generated from the retrieved documents.")
            st.stop()

        st.write(answer)
        st.session_state.chat_history.append(
            {
                "question": question,
                "rewritten_question": retrieval_question,
                "answer": answer,
                "confidence": confidence,
            }
        )


if __name__ == "__main__":
    main()
