"""
Model A — template question generation + ML ranker (aligned with notebooks/model_a_train.ipynb).

Sentence extraction uses the TF-IDF vectorizer; ranker features use the OHE vectorizer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.model_b.inference import clean_text

ROOT = Path(__file__).resolve().parents[2]
MODELA_DIR = ROOT / "models" / "model_a" / "traditional"


class GenerationPackLoadError(RuntimeError):
    """Raised when generation artifacts are missing or fail to load."""


def _ensure_files(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise GenerationPackLoadError(
            "MODEL_A_GENERATION: missing artifact(s): "
            + ", ".join(missing)
            + " — train model_a or check models path."
        )


@dataclass(frozen=True)
class GenerationPack:
    tfidf_vec: Any
    ohe_vec: Any
    rf_ranker: Any
    svm_ranker: Any
    traditional_dir: Path


def load_generation_pack(models_dir: Path | None = None) -> GenerationPack:
    mdir = models_dir or MODELA_DIR
    _ensure_files(
        mdir / "tfidf_vectorizer.pkl",
        mdir / "ohe_vectorizer.pkl",
        mdir / "rf_ranker.pkl",
        mdir / "svm_ranker.pkl",
    )
    try:
        tfidf_vec = joblib.load(mdir / "tfidf_vectorizer.pkl")
        ohe_vec = joblib.load(mdir / "ohe_vectorizer.pkl")
        rf_ranker = joblib.load(mdir / "rf_ranker.pkl")
        svm_ranker = joblib.load(mdir / "svm_ranker.pkl")
    except Exception as e:
        raise GenerationPackLoadError(f"MODEL_A_GENERATION: failed loading from {mdir}: {e}") from e
    return GenerationPack(
        tfidf_vec=tfidf_vec,
        ohe_vec=ohe_vec,
        rf_ranker=rf_ranker,
        svm_ranker=svm_ranker,
        traditional_dir=mdir,
    )


def split_sentences(text: str) -> list[str]:
    """Split article into sentences without NLTK (matches model_a_train.ipynb)."""
    sentences = re.split(r"(?<=[.!?])\s+", str(text))
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def extract_candidate_sentences(
    article: str,
    correct_answer_text: str,
    vectorizer: Any,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Step 1: rank sentences by TF-IDF cosine vs answer + word-overlap tiebreaker.

    ``vectorizer`` must be the Model A **tfidf_vectorizer** (matches notebook behavior).
    """
    sentences = split_sentences(article)
    if not sentences:
        return [(str(article)[:100], 0.0)]

    ans_words = set(str(correct_answer_text).lower().split())

    v_ans = vectorizer.transform([clean_text(correct_answer_text)])
    v_sents = vectorizer.transform([clean_text(s) for s in sentences])
    cos_sims = cosine_similarity(v_ans, v_sents).flatten()

    def word_overlap(sent: str) -> float:
        sent_words = set(sent.lower().split())
        if not ans_words:
            return 0.0
        return len(ans_words & sent_words) / len(ans_words)

    overlap = np.array([word_overlap(s) for s in sentences])
    combined = cos_sims + 0.3 * overlap
    ranked = sorted(zip(sentences, combined), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def apply_wh_template(sentence: str, question_type: str, correct_answer: str = "") -> str:
    """
    Step 2: Wh-word / fill-in templates (matches model_a_train.ipynb).
    """
    s = sentence.strip().rstrip(".").rstrip("?").rstrip("!")
    s_trimmed = s[:120] if len(s) > 120 else s

    if question_type == "Fill-in":
        if correct_answer:
            ans_clean = correct_answer.strip().lower()
            s_lower = s.lower()
            idx = s_lower.find(ans_clean)
            if idx != -1:
                blanked = s[:idx] + "_____" + s[idx + len(ans_clean) :]
                return f"According to the passage, {blanked}."
        return f"According to the passage, _____ {s_trimmed}."

    templates = {
        "What": f"According to the passage, what {s_trimmed}?",
        "Who": f"According to the passage, who {s_trimmed}?",
        "Where": f"According to the passage, where {s_trimmed}?",
        "When": f"According to the passage, when {s_trimmed}?",
        "Why": f"According to the passage, why {s_trimmed}?",
        "How": f"According to the passage, how {s_trimmed}?",
        "Which": f"Which of the following is true? {s_trimmed}.",
        "Other": f"According to the passage, {s_trimmed}.",
    }

    return templates.get(question_type, f"According to the passage, {s_trimmed}.")


def compute_question_features(
    generated_q: str,
    real_article: str,
    correct_answer: str,
    vectorizer: Any,
) -> np.ndarray:
    """
    Six dense features for the question ranker.

    ``vectorizer`` must be the Model A **ohe_vectorizer** (matches training).
    """
    v_q = vectorizer.transform([clean_text(generated_q)])
    v_art = vectorizer.transform([clean_text(real_article[:1000])])
    v_ans = vectorizer.transform([clean_text(correct_answer)])

    cos_art = float(cosine_similarity(v_q, v_art)[0][0])
    cos_ans = float(cosine_similarity(v_q, v_ans)[0][0])

    words = generated_q.lower().split()
    q_len = len(words) / 30.0
    wh_words = {"what", "who", "where", "when", "why", "how", "which"}
    has_wh = float(any(w in wh_words for w in words))
    has_qmark = float("?" in generated_q)
    unique_ratio = len(set(words)) / max(len(words), 1)

    return np.array([cos_art, cos_ans, q_len, has_wh, has_qmark, unique_ratio], dtype=np.float32)


def generate_question_candidates(
    article: str,
    correct_answer: str,
    question_type: str,
    vectorizer: Any,
    top_k: int = 5,
) -> list[tuple[str, float, str]]:
    """
    Steps 1+2: extract top-k sentences (``vectorizer`` = Model A **tfidf_vectorizer**), apply templates.
    Returns list of ``(generated_question, similarity_score, source_sentence)``.
    """
    ranked_sents = extract_candidate_sentences(article, correct_answer, vectorizer, top_k=top_k)
    generated: list[tuple[str, float, str]] = []
    for sent, score in ranked_sents:
        q = apply_wh_template(sent, question_type, correct_answer=correct_answer)
        generated.append((q, float(score), sent))
    return generated


def generate_best_question(
    article: str,
    correct_answer_text: str,
    question_type: str,
    tfidf_vectorizer: Any,
    ohe_vectorizer: Any,
    ranker: Any,
    top_k: int = 5,
) -> tuple[str, list[tuple[str, float, str]]]:
    """
    Steps 1-3: return ranked best question plus all candidates.

    The notebook exposes a single ``vectorizer`` to both pipelines; inference uses
    ``tfidf_vectorizer`` for sentence mining and ``ohe_vectorizer`` for ranker features.
    """
    candidates = generate_question_candidates(
        article,
        correct_answer_text,
        question_type,
        tfidf_vectorizer,
        top_k=top_k,
    )

    if not candidates:
        return "What does the passage discuss?", []

    best_q = candidates[0][0]
    best_score = -1e30

    for q_text, _cos_score, _sent in candidates:
        feats = compute_question_features(q_text, article, correct_answer_text, ohe_vectorizer).reshape(
            1, -1
        )
        if hasattr(ranker, "decision_function"):
            score = float(np.asarray(ranker.decision_function(feats)).ravel()[0])
        else:
            probs = np.asarray(ranker.predict_proba(feats)[0], dtype=float).ravel()
            score = float(probs[1])
        if score > best_score:
            best_score = score
            best_q = q_text

    return best_q, candidates
