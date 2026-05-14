from __future__ import annotations

import html

import streamlit as st
from langchain_core.documents import Document


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
