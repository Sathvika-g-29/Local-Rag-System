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

import hashlib
import html
import importlib
import json
import math
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
from pypdf import PdfReader
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DOCS_DIR = Path("docs")
UPLOADED_DOCS_DIR = Path("uploaded_docs")
VECTOR_DB_DIR = Path("vector_db/chroma")
VECTOR_DB_META = Path("vector_db/index_meta.json")
CHROMA_COLLECTION_PREFIX = "local_rag_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "google/flan-t5-small"
RERANKER_MODEL = "BAAI/bge-reranker-base"
RETRIEVAL_VERSION = "hybrid_bm25_rerank_v1"
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".py", ".html", ".xml", ".log"}
EXCEL_EXTENSIONS = {".xlsx", ".xls"}
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "between",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "with",
}


def hash_text(text: str) -> str:
    """Create a stable hash for cache/index invalidation."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def important_terms(text: str) -> set[str]:
    """Extract simple keyword terms for a lightweight relevance check."""
    terms = {
        term.strip(".,:;!?()[]{}\"'").lower()
        for term in text.split()
    }
    return {term for term in terms if len(term) > 2 and term not in STOP_WORDS}


def safe_filename(filename: str) -> str:
    """Create a simple safe filename for local upload storage."""
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name)


def save_uploaded_files(uploaded_files) -> None:
    """Persist uploaded files so they are available after app restarts."""
    if not uploaded_files:
        return

    UPLOADED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for uploaded_file in uploaded_files:
        destination = UPLOADED_DOCS_DIR / safe_filename(uploaded_file.name)
        destination.write_bytes(uploaded_file.getvalue())


def load_documents_folder(folder: Path, origin: str) -> list[Document]:
    """Load every readable file from a folder."""
    documents: list[Document] = []
    if not folder.exists():
        return documents

    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue

        documents.extend(
            load_file(
                path,
                source_name=str(path.relative_to(folder)),
                origin=origin,
            )
        )

    return documents


def load_file(file, source_name: str, origin: str) -> list[Document]:
    """Load one file using the best parser available for its type."""
    suffix = Path(source_name).suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file, source_name=source_name, origin=origin)

    if suffix in EXCEL_EXTENSIONS:
        return load_excel(file, source_name=source_name, origin=origin)

    if suffix in TEXT_EXTENSIONS:
        return load_text_file(file, source_name=source_name, origin=origin)

    parsed_documents = load_with_unstructured(file, source_name=source_name, origin=origin)
    if parsed_documents:
        return parsed_documents

    return load_text_file(file, source_name=source_name, origin=origin)


def load_text_file(file, source_name: str, origin: str) -> list[Document]:
    """Read text-like files with a forgiving UTF-8 decoder."""
    if isinstance(file, (str, Path)):
        raw_text = Path(file).read_text(encoding="utf-8", errors="ignore")
    else:
        raw_text = file.read().decode("utf-8", errors="ignore")

    if not raw_text.strip():
        return []

    return [
        Document(
            page_content=raw_text,
            metadata={"source": source_name, "origin": origin},
        )
    ]


def load_excel(file, source_name: str, origin: str) -> list[Document]:
    """Read Excel sheets and expose columns plus rows as searchable text."""
    try:
        sheets = pd.read_excel(file, sheet_name=None)
    except Exception:
        return []

    documents: list[Document] = []
    for sheet_name, dataframe in sheets.items():
        dataframe = dataframe.dropna(how="all").dropna(axis=1, how="all")
        if dataframe.empty:
            continue

        columns = [str(column) for column in dataframe.columns]
        sample_rows = dataframe.head(30).fillna("").to_dict(orient="records")
        rows_text = "\n".join(
            ", ".join(f"{key}: {value}" for key, value in row.items())
            for row in sample_rows
        )
        text = (
            f"Excel file: {source_name}\n"
            f"Sheet: {sheet_name}\n"
            f"Columns present: {', '.join(columns)}\n"
            f"Sample rows:\n{rows_text}"
        )

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "origin": origin,
                    "sheet": str(sheet_name),
                },
            )
        )

    return documents


def load_with_unstructured(file, source_name: str, origin: str) -> list[Document]:
    """Use unstructured to parse Office files, HTML, emails, and other formats."""
    try:
        partition_module = importlib.import_module("unstructured.partition.auto")
        partition = partition_module.partition
    except ImportError:
        return []

    try:
        if isinstance(file, (str, Path)):
            elements = partition(filename=str(file))
        else:
            suffix = Path(source_name).suffix
            with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(file.read())

            try:
                elements = partition(filename=str(temp_path))
            finally:
                temp_path.unlink(missing_ok=True)
    except Exception:
        return []

    text = "\n\n".join(str(element) for element in elements if str(element).strip())
    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={"source": source_name, "origin": origin},
        )
    ]


def load_pdf(file, source_name: str, origin: str) -> list[Document]:
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
                metadata={"source": source_name, "page": page_number, "origin": origin},
            )
        )

    return documents


def document_signature(documents: list[Document]) -> str:
    """Hash all document text and metadata so stale indexes are not reused."""
    hasher = hashlib.sha256()
    for doc in documents:
        source = str(doc.metadata.get("source", "unknown"))
        page = str(doc.metadata.get("page", ""))
        origin = str(doc.metadata.get("origin", ""))
        sheet = str(doc.metadata.get("sheet", ""))
        hasher.update(source.encode("utf-8"))
        hasher.update(page.encode("utf-8"))
        hasher.update(origin.encode("utf-8"))
        hasher.update(sheet.encode("utf-8"))
        hasher.update(doc.page_content.encode("utf-8"))
    return hasher.hexdigest()


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
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk"] = index
    return chunks


@st.cache_data(show_spinner=False)
def split_documents_cached(
    document_key: str,
    serialized_documents: tuple[tuple[str, str, int | None, str, str], ...],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Cache recursive chunking for the same document set and settings."""
    documents = [
        Document(
            page_content=text,
            metadata={
                "source": source,
                "origin": origin,
                **({"sheet": sheet} if sheet else {}),
                **({"page": page} if page else {}),
            },
        )
        for source, text, page, origin, sheet in serialized_documents
    ]
    return split_documents(documents, chunk_size, chunk_overlap)


def serialize_documents(documents: list[Document]) -> tuple[tuple[str, str, int | None, str, str], ...]:
    """Make documents cache-friendly for Streamlit."""
    return tuple(
        (
            str(doc.metadata.get("source", "unknown")),
            doc.page_content,
            doc.metadata.get("page"),
            str(doc.metadata.get("origin", "")),
            str(doc.metadata.get("sheet", "")),
        )
        for doc in documents
    )


def serialize_chunks(chunks: list[Document]) -> tuple[tuple[str, str, int | None, str, str, int | None], ...]:
    """Serialize chunk metadata so cached retrievers can be rebuilt exactly once."""
    return tuple(
        (
            str(doc.metadata.get("source", "unknown")),
            doc.page_content,
            doc.metadata.get("page"),
            str(doc.metadata.get("origin", "")),
            str(doc.metadata.get("sheet", "")),
            doc.metadata.get("chunk"),
        )
        for doc in chunks
    )


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
    """Retrieve with semantic search + BM25 keyword search, then ensemble them."""
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
    """Catch common tiny/empty outputs from small local generation models."""
    normalized = answer.strip().lower()
    if not normalized:
        return True

    if normalized == question.strip().lower():
        return True

    if len(re.sub(r"[^a-z0-9]", "", normalized)) < 8:
        return True

    return False


def confidence_label(confidence: int) -> str:
    """Map a numeric score to a simple qualitative label."""
    if confidence >= 75:
        return "High"
    if confidence >= 45:
        return "Medium"
    return "Low"


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


def add_compact_styles() -> None:
    """Keep the Streamlit UI compact and make retrieved chunks easy to scan."""
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-size: 14px;
        }

        .stMarkdown, .stTextInput, .stFileUploader, .stButton, .stSlider {
            font-size: 14px;
        }

        h1 {
            font-size: 1.7rem !important;
        }

        h2, h3 {
            font-size: 1.15rem !important;
        }

        .chunk-box {
            border: 1px solid #d8dee4;
            border-left: 4px solid #4f46e5;
            border-radius: 8px;
            padding: 0.75rem;
            margin: 0.65rem 0;
            background: #f8fafc;
            font-size: 13px;
            line-height: 1.45;
        }

        .chunk-title {
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 0.35rem;
        }

        .chunk-meta {
            color: #6b7280;
            font-size: 12px;
            margin-bottom: 0.45rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_retrieved_chunks(retrieved_docs: list[Document]) -> None:
    """Show retrieved chunks as separate blocks for easier debugging."""
    with st.expander("Retrieved chunks", expanded=False):
        for index, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            sheet = doc.metadata.get("sheet")
            page_text = f" | page {page}" if page else ""
            sheet_text = f" | sheet {sheet}" if sheet else ""
            chunk_text = html.escape(doc.page_content)
            st.markdown(
                f"""
                <div class="chunk-box">
                    <div class="chunk-title">Chunk {index}</div>
                    <div class="chunk-meta">Source: {source}{page_text}{sheet_text} | chunk {doc.metadata.get("chunk", index)}</div>
                    <div>{chunk_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


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
