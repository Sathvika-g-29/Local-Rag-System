from __future__ import annotations

import importlib
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import EXCEL_EXTENSIONS, TEXT_EXTENSIONS, UPLOADED_DOCS_DIR
from .utils import safe_filename


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
    from .utils import hash_text

    parts: list[str] = []
    for doc in documents:
        parts.extend(
            [
                str(doc.metadata.get("source", "unknown")),
                str(doc.metadata.get("page", "")),
                str(doc.metadata.get("origin", "")),
                str(doc.metadata.get("sheet", "")),
                doc.page_content,
            ]
        )
    return hash_text("||".join(parts))


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split documents into recursive chunks."""
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
