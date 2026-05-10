"""
Model A: answer verification using preprocessing-aligned features + verifier pickles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import joblib
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
MODELA_DIR = ROOT / "models" / "model_a" / "traditional"

from src.model_a.features import (  # noqa: E402
    build_combined_four_rows,
    letter_index,
    preprocess_quiz_row,
    verification_combined_strings,
)

VerifierName = Literal["lr_binary", "svm_binary", "ensemble"]

_LETTERS = ("A", "B", "C", "D")


class ModelAPackLoadError(RuntimeError):
    """Raised when Model A artifacts are missing or fail to load."""


def _ensure_files(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise ModelAPackLoadError(
            f"MODELA: missing artifact(s): {', '.join(missing)} — run preprocessing + model_a_train or check models path."
        )


@dataclass(frozen=True)
class ModelAPack:
    lr_binary: Any
    svm_binary: Any
    lr_ensemble: Any
    nb_ensemble: Any
    tfidf_vec: Any
    ohe_vec: Any
    cos_vec: Any
    traditional_dir: Path


def load_model_a_pack(models_dir: Path | None = None) -> ModelAPack:
    mdir = models_dir or MODELA_DIR
    _ensure_files(
        mdir / "lr_binary.pkl",
        mdir / "svm_binary.pkl",
        mdir / "lr_ensemble.pkl",
        mdir / "nb_ensemble.pkl",
        mdir / "tfidf_vectorizer.pkl",
        mdir / "ohe_vectorizer.pkl",
        mdir / "cos_vectorizer.pkl",
    )
    try:
        lr_binary = joblib.load(mdir / "lr_binary.pkl")
        svm_binary = joblib.load(mdir / "svm_binary.pkl")
        lr_ensemble = joblib.load(mdir / "lr_ensemble.pkl")
        nb_ensemble = joblib.load(mdir / "nb_ensemble.pkl")
        tfidf_vec = joblib.load(mdir / "tfidf_vectorizer.pkl")
        ohe_vec = joblib.load(mdir / "ohe_vectorizer.pkl")
        cos_vec = joblib.load(mdir / "cos_vectorizer.pkl")
    except Exception as e:
        raise ModelAPackLoadError(f"MODELA: failed loading pickles from {mdir}: {e}") from e

    return ModelAPack(
        lr_binary=lr_binary,
        svm_binary=svm_binary,
        lr_ensemble=lr_ensemble,
        nb_ensemble=nb_ensemble,
        tfidf_vec=tfidf_vec,
        ohe_vec=ohe_vec,
        cos_vec=cos_vec,
        traditional_dir=mdir,
    )


def compose_prd_options(correct_letter: str, gold_text: str, distractors: list[str]) -> dict[str, str]:
    """
    PRD quiz: gold at teacher key letter; three Model B strings on the other letters.
    distractors must have length 3 (Model B pads).
    """
    c = str(correct_letter).strip().upper()
    if c not in _LETTERS:
        raise ValueError("correct_letter must be A,B,C, or D")
    d = list(distractors)
    while len(d) < 3:
        d.append("")
    d = d[:3]
    out: dict[str, str] = {}
    wi = 0
    for L in _LETTERS:
        if L == c:
            out[L] = str(gold_text)
        else:
            out[L] = str(d[wi])
            wi += 1
    return out


def _prob_correct_from_proba_row(
    proba_row: np.ndarray,
    classes: np.ndarray,
    proba_threshold: float,
) -> tuple[float, float, bool]:
    """Return (p_correct for label 1, p_incorrect, predicts_correct)."""
    pr = np.asarray(proba_row, dtype=float).ravel()
    cl = np.asarray(classes).ravel()
    pmap = {int(c): float(p) for c, p in zip(cl, pr)}
    p_correct = float(pmap.get(1, pr[-1] if pr.size > 1 else pr[0]))
    p_incorrect = float(pmap.get(0, 1.0 - p_correct))
    return p_correct, p_incorrect, p_correct >= proba_threshold


def _predict_proba_one(clf: Any, x1: Any) -> tuple[np.ndarray, np.ndarray]:
    proba = np.asarray(clf.predict_proba(x1)[0], dtype=float).ravel()
    classes = np.asarray(getattr(clf, "classes_", np.array([0, 1]))).ravel()
    return proba, classes


def predict_verification(
    article: str,
    question: str,
    texts: dict[str, str],
    selected_letter: str,
    pack: ModelAPack,
    gold_answer_letter: Optional[str] = None,
    proba_threshold: float = 0.5,
    verifier: VerifierName = "lr_binary",
) -> dict[str, Any]:
    """
    Binary verifier for the selected option.

    texts: A–D strings (quiz options as shown to the verifier, keyed by teacher letters A–D).
    selected_letter: original A–D key for the chosen option (UI maps shuffle → original).

    verifier:
      lr_binary — full TF-IDF + OHE + cosine + lexical (combined), logistic predict_proba
      svm_binary — same combined features; LinearSVC uses expit(decision_function) as soft score
      ensemble — TF-IDF only; soft average of lr_ensemble and nb_ensemble predict_proba, threshold 0.5 on P(class 1)
    """
    idx = letter_index(selected_letter)
    row = preprocess_quiz_row(article, question, texts)

    if verifier == "ensemble":
        combined_texts = verification_combined_strings(row)
        Xt = pack.tfidf_vec.transform(combined_texts)
        xt = Xt[idx : idx + 1]
        p_lr, c_lr = _predict_proba_one(pack.lr_ensemble, xt)
        p_nb, c_nb = _predict_proba_one(pack.nb_ensemble, xt)
        if not np.array_equal(np.asarray(c_lr).ravel(), np.asarray(c_nb).ravel()):
            raise ValueError("Ensemble LR/NB class order mismatch.")
        avg = (p_lr + p_nb) / 2.0
        p_correct, p_incorrect, predicts_correct = _prob_correct_from_proba_row(
            avg, c_lr, proba_threshold
        )
    elif verifier == "svm_binary":
        X4 = build_combined_four_rows(row, pack.tfidf_vec, pack.ohe_vec, pack.cos_vec)
        x1 = X4[idx : idx + 1]
        svm = pack.svm_binary
        if hasattr(svm, "predict_proba"):
            proba, classes = _predict_proba_one(svm, x1)
            p_correct, p_incorrect, predicts_correct = _prob_correct_from_proba_row(
                proba, classes, proba_threshold
            )
        else:
            s = float(np.asarray(svm.decision_function(x1)).ravel()[0])
            p_correct = float(expit(s))
            p_incorrect = float(1.0 - p_correct)
            predicts_correct = p_correct >= proba_threshold
    else:
        X4 = build_combined_four_rows(row, pack.tfidf_vec, pack.ohe_vec, pack.cos_vec)
        x1 = X4[idx : idx + 1]
        proba, classes = _predict_proba_one(pack.lr_binary, x1)
        p_correct, p_incorrect, predicts_correct = _prob_correct_from_proba_row(
            proba, classes, proba_threshold
        )

    out: dict[str, Any] = {
        "is_correct": bool(predicts_correct),
        "confidence": float(p_correct),
        "predicted_label": "VERIFIED_LIKELY_CORRECT" if predicts_correct else "VERIFIED_LIKELY_INCORRECT",
        "proba_incorrect": p_incorrect,
        "explanation": (
            "The verifier estimates this option is consistent with the passage and question."
            if predicts_correct
            else "The verifier estimates this option is unlikely to be the best answer given the passage."
        ),
    }

    if gold_answer_letter is not None:
        ga = str(gold_answer_letter).strip().upper()
        oracle = str(selected_letter).strip().upper() == ga
        out["oracle_match"] = oracle

    return out
