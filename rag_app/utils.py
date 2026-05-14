from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .config import STOP_WORDS


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
