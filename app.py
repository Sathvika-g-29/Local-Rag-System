"""
Streamlit + Hugging Face local RAG demo.

Run:
    streamlit run build_rag.py

This app keeps the RAG pipeline intentionally small and readable:
1. Load local text/PDF files from docs/ and optional uploaded files.
2. Split text with RecursiveCharacterTextSplitter.
3. Convert chunks into Hugging Face sentence embeddings.
4. Store/search the embeddings with FAISS.
5. Ask a Hugging Face model to answer using only retrieved context.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DOCS_DIR = Path("docs")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "google/flan-t5-small"


def load_docs_folder() -> list[Document]:
    """Load .txt, .md, and .pdf files from the local docs/ folder."""
    documents: list[Document] = []

    for path in sorted(DOCS_DIR.glob("*")):
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            documents.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"source": path.name},
                )
            )
            continue

        if suffix == ".pdf":
            documents.extend(load_pdf(path, source_name=path.name))

    return documents


def load_pdf(file, source_name: str) -> list[Document]:
    """Extract each PDF page as a separate LangChain Document."""
    reader = PdfReader(file)
    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={"source": source_name, "page": page_number},
            )
        )

    return documents


def load_uploaded_files(uploaded_files) -> list[Document]:
    """Turn Streamlit uploads into LangChain Document objects."""
    documents: list[Document] = []

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix == ".pdf":
            documents.extend(load_pdf(uploaded_file, source_name=uploaded_file.name))
            continue

        text = uploaded_file.read().decode("utf-8")
        documents.append(
            Document(
                page_content=text,
                metadata={"source": uploaded_file.name},
            )
        )

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """
    Split documents recursively.

    The splitter tries paragraph breaks first, then lines, then spaces, then
    characters. That keeps chunks natural while still enforcing a max size.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


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


def build_vector_store(chunks: list[Document]) -> FAISS:
    """Create a FAISS index from document chunks."""
    return FAISS.from_documents(chunks, get_embeddings())


def build_prompt(question: str, retrieved_docs: list[Document]) -> str:
    """Create a strict prompt so the answer stays grounded in context."""
    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
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


def main() -> None:
    st.set_page_config(page_title="Local Hugging Face RAG", page_icon="RAG")
    st.title("Local Hugging Face RAG")

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Retrieved chunks", min_value=1, max_value=6, value=3)
        chunk_size = st.slider("Chunk size", min_value=200, max_value=1200, value=500, step=50)
        chunk_overlap = st.slider(
            "Chunk overlap",
            min_value=0,
            max_value=300,
            value=100,
            step=25,
        )

    uploaded_files = st.file_uploader(
        "Upload .txt, .md, or .pdf files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    question = st.text_input(
        "Ask a question",
        value="What are the main steps in a RAG system?",
    )

    folder_docs = load_docs_folder()
    uploaded_docs = load_uploaded_files(uploaded_files)
    documents = folder_docs + uploaded_docs

    if not documents:
        st.warning("Add .txt, .md, or .pdf files to docs/ or upload files above.")
        return

    chunks = split_documents(documents, chunk_size, chunk_overlap)

    if st.button("Build index and answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Building FAISS index with Hugging Face embeddings..."):
            vector_store = build_vector_store(chunks)

        with st.spinner("Retrieving relevant chunks..."):
            retrieved_docs = vector_store.similarity_search(question, k=top_k)

        with st.spinner("Generating answer with Hugging Face..."):
            answer = generate_answer(question, retrieved_docs)

        st.subheader("Answer")
        st.write(answer)


if __name__ == "__main__":
    main()
