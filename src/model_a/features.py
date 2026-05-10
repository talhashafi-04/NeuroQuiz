"""
Model A verification features — parity with notebooks/preprocessing.ipynb.
Builds one RACE row's four (row, option) sparse combined vectors: shape (4, 18019).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from src.text_clean import clean_text, truncate_article

STOPWORDS = set(ENGLISH_STOP_WORDS)
LETTERS = ["A", "B", "C", "D"]


def preprocess_quiz_row(
    article: str,
    question: str,
    option_texts: dict[str, str],
) -> dict[str, str]:
    """Return *_clean fields like clean_dataframe for one row."""
    row: dict[str, str] = {}
    row["article_clean"] = truncate_article(clean_text(article))
    row["question_clean"] = clean_text(question)
    for L in LETTERS:
        row[f"{L}_clean"] = clean_text(option_texts[L])
    return row


def verification_combined_strings(row: dict[str, str]) -> list[str]:
    """Four strings in A,B,C,D order for TF-IDF / OHE (matches build_verification_texts)."""
    art = row["article_clean"]
    q = row["question_clean"]
    texts: list[str] = []
    for L in LETTERS:
        opt_text = row[f"{L}_clean"]
        texts.append(f"{art} {art} {q} {opt_text}")
    return texts


def build_cosine_features_one_row(row: dict[str, str], vectorizer: Any) -> np.ndarray:
    """Shape (4, 6) — same as preprocessing build_cosine_features for n_rows=1."""
    features = np.zeros((4, 6), dtype=np.float32)
    art = row["article_clean"]
    q = row["question_clean"]
    art_vec = vectorizer.transform([art])
    q_vec = vectorizer.transform([q])

    cos_art_opt_scores = []
    for j, opt_letter in enumerate(LETTERS):
        opt_text = row[f"{opt_letter}_clean"]
        qopt = q + " " + opt_text
        opt_vec = vectorizer.transform([opt_text])
        qopt_vec = vectorizer.transform([qopt])
        cos_ao = float(cosine_similarity(art_vec, opt_vec)[0, 0])
        cos_aq = float(cosine_similarity(art_vec, q_vec)[0, 0])
        cos_qo = float(cosine_similarity(q_vec, opt_vec)[0, 0])
        cos_aqo = float(cosine_similarity(art_vec, qopt_vec)[0, 0])
        features[j, 0] = cos_ao
        features[j, 1] = cos_aq
        features[j, 2] = cos_qo
        features[j, 3] = cos_aqo
        cos_art_opt_scores.append(cos_ao)

    scores = np.array(cos_art_opt_scores, dtype=np.float32)
    ranks = scores.argsort().argsort()
    is_max = (scores == scores.max()).astype(np.float32)
    for j in range(4):
        features[j, 4] = ranks[j]
        features[j, 5] = is_max[j]

    return features


def get_words(text: str) -> set[str]:
    return set(text.split()) - STOPWORDS


def get_question_type_code(question_clean: str) -> int:
    q = str(question_clean).lower().strip()
    if "_" in q or "blank" in q:
        return 0
    for wh in ("what", "who", "where", "when", "why", "how", "which"):
        if q.startswith(wh):
            return 1
    return 2


def build_lexical_features_one_row(row: dict[str, str]) -> np.ndarray:
    """Shape (4, 13)."""
    features = np.zeros((4, 13), dtype=np.float32)
    art_words = get_words(row["article_clean"])
    q_words = get_words(row["question_clean"])
    art_len = len(row["article_clean"].split())
    q_len = len(row["question_clean"].split())
    qtype = get_question_type_code(row["question_clean"])

    overlap_scores: list[float] = []
    for j, opt_letter in enumerate(LETTERS):
        opt_text = row[f"{opt_letter}_clean"]
        opt_words = get_words(opt_text)
        opt_len = len(opt_text.split())

        inter_ao = art_words & opt_words
        union_ao = art_words | opt_words
        n_overlap = len(inter_ao)
        jaccard_ao = n_overlap / len(union_ao) if union_ao else 0.0
        ratio_opt = n_overlap / (opt_len + 1)
        ratio_art = n_overlap / (art_len + 1)

        inter_qo = q_words & opt_words
        union_qo = q_words | opt_words
        n_q_overlap = len(inter_qo)
        jaccard_qo = n_q_overlap / len(union_qo) if union_qo else 0.0

        features[j, 0] = n_overlap
        features[j, 1] = jaccard_ao
        features[j, 2] = ratio_opt
        features[j, 3] = ratio_art
        features[j, 4] = n_q_overlap
        features[j, 5] = jaccard_qo
        features[j, 6] = art_len
        features[j, 7] = opt_len
        features[j, 8] = q_len
        features[j, 9] = opt_len / (art_len + 1)
        features[j, 12] = qtype
        overlap_scores.append(float(n_overlap))

    scores_arr = np.array(overlap_scores, dtype=np.float32)
    ranks = scores_arr.argsort().argsort()
    is_max = (scores_arr == scores_arr.max()).astype(np.float32)
    for j in range(4):
        features[j, 10] = ranks[j]
        features[j, 11] = is_max[j]

    return features


def build_combined_stack(
    X_tfidf: Any,
    X_ohe: Any,
    cos_feats: np.ndarray,
    lex_feats: np.ndarray,
) -> Any:
    cos_sparse = sp.csr_matrix(cos_feats)
    lex_sparse = sp.csr_matrix(lex_feats)
    return sp.hstack([X_tfidf, X_ohe, cos_sparse, lex_sparse], format="csr")


def build_combined_four_rows(
    row: dict[str, str],
    tfidf_vec: Any,
    ohe_vec: Any,
    cos_vec: Any,
) -> Any:
    """CSR matrix shape (4, 18019) for one quiz row."""
    combined_texts = verification_combined_strings(row)
    X_tfidf = tfidf_vec.transform(combined_texts)
    X_ohe = ohe_vec.transform(combined_texts)
    cos_f = build_cosine_features_one_row(row, cos_vec)
    lex_f = build_lexical_features_one_row(row)
    return build_combined_stack(X_tfidf, X_ohe, cos_f, lex_f)


def letter_index(letter: str) -> int:
    L = str(letter).strip().upper()
    if L not in LETTERS:
        raise ValueError(f"Bad option letter: {letter}")
    return LETTERS.index(L)
