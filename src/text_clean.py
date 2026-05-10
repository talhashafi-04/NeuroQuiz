"""Shared text cleaning — matches notebooks/preprocessing.ipynb Cell 3."""

from __future__ import annotations

import re
from typing import Any


def clean_text(text: Any) -> str:
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def truncate_article(text: str, max_words: int = 500) -> str:
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text
