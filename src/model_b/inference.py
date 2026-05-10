"""
Model B inference: distractors + hints, aligned with notebooks/model_b_train.ipynb.

Loads are delegated to callers (Streamlit ``st.cache_resource``) via ``load_model_pack``.
ROOT for paths: src/model_b -> src -> repo root is parents[2].
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
MODELB_DIR = ROOT / "models" / "model_b" / "traditional"


class ModelBPackLoadError(RuntimeError):
    """Raised when Model B artifacts are missing or cannot be deserialized."""


def clean_text(text: Any) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def truncate_article(text: str, max_words: int = 500) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def split_sentences(article_clean: str, min_tokens: int = 8, chunk_words: int = 40, overlap: int = 15) -> list[str]:
    """Split passage into snippets for hints / distractor mining — matches notebook Cell 2."""
    tokens = article_clean.split()
    if len(tokens) < min_tokens:
        return ([article_clean] if article_clean.strip() else [])

    parts = re.split(r"(?<=[.!?])\s+", article_clean)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if len(p.split()) >= min_tokens:
            out.append(p)

    if len(out) < 2:
        out.clear()
        step = max(1, chunk_words - overlap)
        i = 0
        while i < len(tokens):
            chunk = tokens[i : i + chunk_words]
            if len(chunk) >= min_tokens:
                out.append(" ".join(chunk))
            i += step
        if len(out) < 2 and len(tokens) >= min_tokens * 2:
            mid = max(min_tokens, len(tokens) // 2)
            out = [" ".join(tokens[:mid]), " ".join(tokens[mid:])]

    if not out and tokens:
        out = [" ".join(tokens[: max(min_tokens, min(len(tokens), 30))])]
    return out


def norm_key(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def jaccard_tokens(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def article_prefix(article_clean: str, n_words: int) -> str:
    return " ".join(article_clean.split()[:n_words])


def _ensure_files(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise ModelBPackLoadError(
            f"MODELB: missing artifact(s): {', '.join(missing)} — train Model B first or check models path."
        )


@dataclass(frozen=True)
class ModelBPack:
    tfidf_vec: Any
    cos_vec: Any
    lr_distractor: Any
    rf_distractor: Any
    hint_scorer: Any
    article_prefix_words: int
    traditional_dir: Path


def load_model_pack(models_dir: Path | None = None) -> ModelBPack:
    """
    Lazy-friendly loader: call once from ``st.cache_resource`` in the Streamlit app.
    """
    mdir = models_dir or MODELB_DIR
    _ensure_files(
        mdir / "tfidf_vectorizer.pkl",
        mdir / "cos_vectorizer.pkl",
        mdir / "distractor_ranker.pkl",
        mdir / "random_forest_distractor.pkl",
        mdir / "hint_scorer.pkl",
        mdir / "model_b_config.pkl",
    )
    try:
        cfg = joblib.load(mdir / "model_b_config.pkl")
        apw = int(cfg.get("article_prefix_words", 120))
        tfidf_vec = joblib.load(mdir / "tfidf_vectorizer.pkl")
        cos_vec = joblib.load(mdir / "cos_vectorizer.pkl")
        lr_distractor = joblib.load(mdir / "distractor_ranker.pkl")
        rf_distractor = joblib.load(mdir / "random_forest_distractor.pkl")
        hint_scorer = joblib.load(mdir / "hint_scorer.pkl")
    except Exception as e:
        raise ModelBPackLoadError(f"MODELB: failed loading pickles from {mdir}: {e}") from e

    return ModelBPack(
        tfidf_vec=tfidf_vec,
        cos_vec=cos_vec,
        lr_distractor=lr_distractor,
        rf_distractor=rf_distractor,
        hint_scorer=hint_scorer,
        article_prefix_words=apw,
        traditional_dir=mdir,
    )


def cand_features_row(
    cand: str,
    gold: str,
    question: str,
    article_prefix_txt: str,
    pack: ModelBPack,
) -> np.ndarray:
    """8-dim dense vector for one candidate snippet — notebook Cell 4."""
    tfidf_vec = pack.tfidf_vec
    cos_vec = pack.cos_vec
    m = tfidf_vec.transform([cand, gold, question, article_prefix_txt])
    c, g, q, ap = m[0], m[1], m[2], m[3]
    cos_cg = float(cosine_similarity(c, g)[0, 0])
    cos_cq = float(cosine_similarity(c, q)[0, 0])
    cos_cap = float(cosine_similarity(c, ap)[0, 0])
    cav = cos_vec.transform([cand])
    mav = cos_vec.transform([question + " " + gold])
    car = cos_vec.transform([article_prefix_txt])
    cos_c_art = float(cosine_similarity(cav, car)[0, 0])
    cos_c_qgold = float(cosine_similarity(cav, mav)[0, 0])
    return np.array(
        [
            cos_cg,
            cos_cq,
            cos_cap,
            cos_c_art,
            cos_c_qgold,
            np.log1p(len(cand.split())),
            jaccard_tokens(cand, question),
            jaccard_tokens(cand, gold),
        ],
        dtype=np.float32,
    )


def generate_distractors(
    article_clean: str,
    question_clean: str,
    gold_answer_clean: str,
    pack: ModelBPack,
    min_sent_tokens: int = 10,
    beam: int = 48,
) -> list[str]:
    """
    Notebook Cell 6: greedy top distractors via LR distractor probability
    plus Jaccard diversity; pad from lower-ranked pool if needed.
    """
    model = pack.lr_distractor
    sentences = split_sentences(article_clean, min_tokens=min_sent_tokens)
    ap = article_prefix(article_clean, pack.article_prefix_words)
    gold = norm_key(gold_answer_clean)

    scored: list[tuple[float, str]] = []
    for s in sentences:
        cand = norm_key(s)
        if len(cand) < 12:
            continue
        if jaccard_tokens(cand, gold) > 0.92:
            continue
        feats = cand_features_row(cand, gold, question_clean, ap, pack).reshape(1, -1)
        prob = float(model.predict_proba(feats)[0, 1])
        scored.append((prob, cand))

    scored.sort(key=lambda x: x[0], reverse=True)

    chosen: list[str] = []
    for prob, cand in scored:
        if len(chosen) >= 3:
            break
        if all(jaccard_tokens(cand, ch) < 0.55 for ch in chosen):
            chosen.append(cand)
        if len(scored) < beam and len(chosen) >= 3:
            break

    if len(chosen) < 3:
        for prob, cand in scored[len(chosen) :]:
            if cand in chosen:
                continue
            if jaccard_tokens(cand, gold) > 0.85:
                continue
            chosen.append(cand)
            if len(chosen) >= 3:
                break

    if len(chosen) >= 3:
        return chosen[:3]

    # Meaningful fallback: extra windows from passage (distinct from chosen / gold).
    tokens = article_clean.split()
    step = 30
    for i in range(0, len(tokens), step):
        chunk = norm_key(" ".join(tokens[i : i + 50]))
        if len(chunk.split()) < 8:
            continue
        if chunk == gold:
            continue
        if chunk in chosen:
            continue
        if jaccard_tokens(chunk, gold) > 0.92:
            continue
        chosen.append(chunk)
        if len(chosen) >= 3:
            break

    while len(chosen) < 3:
        qw = question_clean.split()[-12:] if question_clean.strip() else []
        filler = norm_key(" ".join(qw)) if qw else ""
        if not filler.strip():
            filler = norm_key(" ".join(tokens[: min(40, len(tokens))])) if tokens else "passage excerpt"
        tag = len(chosen)
        piece = (f"{filler} [{tag}]".strip())[:500]
        if piece not in chosen:
            chosen.append(piece)
        else:
            chosen.append((piece + " backup").strip()[:500])
    return chosen[:3]


def hint_features_row(
    sentence: str,
    question_clean: str,
    gold_norm_key: str,
    article_clean: str,
    sentence_index: int,
    n_sents: int,
    pack: ModelBPack,
) -> np.ndarray:
    """8-d notebook Cell 8 / ``rank_hints`` feature row (dtype float32, same column order)."""
    ap_txt = article_prefix(article_clean, pack.article_prefix_words)
    cos_vec = pack.cos_vec
    s = sentence
    j = sentence_index
    slen = len(s.split())
    cav = cos_vec.transform([s])
    qv = cos_vec.transform([question_clean])
    gv = cos_vec.transform([gold_norm_key])
    return np.array(
        [
            slen,
            np.log1p(slen),
            jaccard_tokens(s, question_clean),
            jaccard_tokens(s, gold_norm_key),
            float(cosine_similarity(cav, qv)[0, 0]),
            float(cosine_similarity(cav, gv)[0, 0]),
            float(cosine_similarity(cav, cos_vec.transform([ap_txt]))[0, 0]),
            j / max(1, n_sents - 1),
        ],
        dtype=np.float32,
    )


def generate_hints(
    article_clean: str,
    question_clean: str,
    gold_answer_clean: str,
    pack: ModelBPack,
) -> list[str]:
    """
    Score every ``split_sentences(..., min_tokens=6)`` chunk with hint_scorer;
    pick the **three highest** ``predict_proba[:,1]``, then reorder **weak→strong**

    for the UI: among those three, ascending probability ⇒ Hint 1 (vaguest)
    … Hint 3 (strongest toward the gold). Matches PRD gradual reveal semantics.
    """
    gold = norm_key(gold_answer_clean)
    sents = split_sentences(article_clean, min_tokens=6)
    if not sents:
        return []

    feats = [
        hint_features_row(s, question_clean, gold, article_clean, j, len(sents), pack)
        for j, s in enumerate(sents)
    ]
    F = np.stack(feats)
    prob = pack.hint_scorer.predict_proba(F)[:, 1]

    n = len(sents)
    if n <= 3:
        triples = sorted(zip(prob, sents, range(n)), key=lambda x: float(x[0]))
        return [t[1] for t in triples]

    # Top-3 strongest by scorer (high prob = stronger / more revealing in training silver labels).
    order = np.argsort(-prob, kind="stable")
    top3_idx = order[:3]
    band = [(float(prob[i]), sents[i]) for i in top3_idx]
    band.sort(key=lambda x: x[0])
    # hints_weak_to_strong: index 0 = weakest among the selected band for Tab 3
    return [text for _, text in band]


def prepare_article_question_gold_for_model_b(
    article: str, question: str, gold_answer: str
) -> tuple[str, str, str]:
    """Apply the same cleaning as training preprocessing for passage / Q / gold option."""
    aq = truncate_article(clean_text(article))
    qq = clean_text(question)
    gq = clean_text(gold_answer)
    return aq, qq, gq
