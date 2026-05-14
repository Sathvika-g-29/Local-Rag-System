from pathlib import Path


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
