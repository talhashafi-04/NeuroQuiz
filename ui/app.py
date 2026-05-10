"""
NeuroQuiz Streamlit entry point. Run from repo root:
  streamlit run ui/app.py
"""

from __future__ import annotations

import csv
import io
import os
import random
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_a.generation import (  # noqa: E402
    GenerationPackLoadError,
    generate_best_question,
    load_generation_pack,
)
from src.model_a.inference import (  # noqa: E402
    ModelAPackLoadError,
    VerifierName,
    compose_prd_options,
    load_model_a_pack,
    predict_verification,
)
from src.model_b.inference import (  # noqa: E402
    ModelBPackLoadError,
    generate_distractors,
    generate_hints,
    load_model_pack,
    prepare_article_question_gold_for_model_b,
    truncate_article,
)

VAL_CSV = ROOT / "data" / "processed" / "val_split.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
Y_VAL_BINARY_PATH = DATA_PROCESSED / "y_val_binary.npy"
X_VAL_COMBINED_PATH = DATA_PROCESSED / "X_val_combined.npz"
MODELA_TRAD_PATH = ROOT / "models" / "model_a" / "traditional"
RESULTS_MODEL_B_DIR = ROOT / "results" / "model_b"
MODEL_B_HOLDOUT_CM_PNG = RESULTS_MODEL_B_DIR / "distractor_confusion_holdout.png"
REQUIRED_COLUMNS = ("article", "question", "A", "B", "C", "D", "answer")
OFFLINE_METRICS_CSV = ROOT / "notebooks" / "results" / "model_a" / "classification_results.csv"

UNIFIED_NOT_READY_MSG = (
    "Submit an article first — use **Tab 1** to **Run Model B** (or load a random row from the sidebar)."
)
ARTICLE_SHORT_WORDS_HINT = 40


def quiz_ready() -> bool:
    return bool(st.session_state.get("shuffled_pairs") and st.session_state.get("quiz_composed_texts"))


def offline_reference_val_accuracy() -> tuple[float | None, str]:
    """Parse offline CSV for LR binary val accuracy when available."""
    if not OFFLINE_METRICS_CSV.exists():
        return None, "No offline CSV at notebooks/results/model_a/classification_results.csv"
    try:
        df = pd.read_csv(OFFLINE_METRICS_CSV)
        if df.empty:
            return None, "CSV empty"
        col_acc = None
        for c in df.columns:
            if str(c).lower().strip() in ("accuracy", "acc"):
                col_acc = c
                break
        if col_acc is None:
            return None, "No Accuracy column found"
        name_col = None
        for c in df.columns:
            if "model" in str(c).lower() or "method" in str(c).lower() or "name" in str(c).lower():
                name_col = c
                break
        if name_col is not None:
            names = df[name_col].astype(str).str.lower()
            mask_bv = names.str.contains("binary", na=False) & names.str.contains("verifier", na=False)
            sub = df[mask_bv]
            if not sub.empty:
                return float(sub.iloc[0][col_acc]), "Binary verifier row"
            mask_lr = names.str.contains("lr", na=False) & names.str.contains("binary", na=False)
            sub = df[mask_lr]
            if not sub.empty:
                return float(sub.iloc[0][col_acc]), "LR binary row"
        return float(df.iloc[0][col_acc]), "first CSV row (fallback)"
    except Exception as ex:
        return None, str(ex)

VERIFIER_LABEL_TO_KEY: dict[str, VerifierName] = {
    "LR Binary": "lr_binary",
    "SVM Binary": "svm_binary",
    "Ensemble (LR + NB soft voting)": "ensemble",
}


def effective_quiz_mode() -> str:
    """prd | evaluation; custom Submit always behaves as PRD."""
    if st.session_state.get("quiz_source") != "dataset":
        return "prd"
    return str(st.session_state.get("quiz_mode", "prd"))


def dataset_rerun_inference_from_session_state() -> None:
    article = str(st.session_state.get("article_body") or "").strip()
    question = str(st.session_state.get("question_text") or "").strip()
    if not article or not question:
        return
    if not all(str(st.session_state.get(f"option_{L.lower()}", "")).strip() for L in "ABCD"):
        return
    texts = {L: st.session_state[f"option_{L.lower()}"] for L in "ABCD"}
    corr = str(st.session_state.get("correct_letter", "A")).strip().upper()
    use_prd = effective_quiz_mode() != "evaluation"
    run_inference_for_row(article, question, texts, corr, use_prd_quiz=use_prd)


def _on_quiz_mode_radio_changed() -> None:
    sel = str(st.session_state.get("_quiz_mode_widget") or "")
    st.session_state["quiz_mode"] = "evaluation" if sel.startswith("Evaluation") else "prd"
    if st.session_state.get("quiz_source") == "dataset":
        dataset_rerun_inference_from_session_state()


@st.cache_resource
def get_model_b_pack():
    return load_model_pack()


@st.cache_resource
def get_model_a_pack():
    return load_model_a_pack()


@st.cache_resource
def get_generation_pack():
    return load_generation_pack()


QUESTION_GEN_TYPES = ("What", "Who", "Where", "When", "Why", "Fill-in")
RANKER_CHOICE_RF = "Random Forest (rf_ranker)"
RANKER_CHOICE_SVM = "SVM (svm_ranker)"


@st.cache_data
def load_validation_df():
    if not VAL_CSV.exists():
        raise FileNotFoundError(
            f"Missing {VAL_CSV}. Run EDA + preprocessing notebooks to build data/processed/*_split.csv."
        )
    df = pd.read_csv(VAL_CSV)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"val_split.csv missing columns: {missing}")
    return df


@st.cache_data(show_spinner="Computing Model A validation predictions…")
def cached_model_a_val_binary_predictions(cache_bust: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cache_bust: mtimes so cache invalidates when numpy/pkl artifacts change."""
    y = np.asarray(np.load(Y_VAL_BINARY_PATH)).astype(np.int64, copy=False).ravel()
    X = sp.load_npz(X_VAL_COMBINED_PATH)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X_val ({X.shape[0]} rows) and y_val_binary ({y.shape[0]} rows) length mismatch."
        )
    lr = joblib.load(MODELA_TRAD_PATH / "lr_binary.pkl")
    svm_mod = joblib.load(MODELA_TRAD_PATH / "svm_binary.pkl")
    pred_lr = np.asarray(lr.predict(X)).astype(np.int64, copy=False).ravel()
    pred_svm = np.asarray(svm_mod.predict(X)).astype(np.int64, copy=False).ravel()
    return y, pred_lr, pred_svm


def _artifact_mtimes(paths: tuple[Path, ...]) -> tuple[float, ...]:
    return tuple(float(p.stat().st_mtime) if p.exists() else -1.0 for p in paths)


def plot_model_a_confusion_fig(y_true: np.ndarray, y_pred: np.ndarray, *, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.25, 3.85))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=[0, 1],
        ax=ax,
        colorbar=False,
        text_kw={"fontsize": 10},
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Predicted label (0=incorrect, 1=correct)")
    ax.set_ylabel("True label (0=incorrect, 1=correct)")
    fig.tight_layout()
    return fig


MODEL_B_DISTRACTOR_METRICS_HARD = pd.DataFrame(
    [
        {
            "Model": "Logistic Regression",
            "Accuracy": 0.9506,
            "Precision": 0.8949,
            "Recall": 0.9633,
            "F1": 0.9278,
        },
        {
            "Model": "Random Forest",
            "Accuracy": 0.9819,
            "Precision": 0.9601,
            "Recall": 0.9860,
            "F1": 0.9729,
        },
    ]
)

# Notebook model_b_train Cell 8 (val heuristic); Cell 7 does not print Precision@K or R² — see Notes column.
MODEL_B_HINT_METRICS_HARD = pd.DataFrame(
    [
        {
            "Metric": "Precision at K",
            "Value": 0.5646,
            "K": 3,
            "Notes": "Val sample: mean word-overlap of top-3 hints with gold (Cell 8).",
        },
        {
            "Metric": "R² score",
            "Value": None,
            "K": "",
            "Notes": "Not reported in model_b_train.ipynb (Cell 7 prints train accuracy & F1 only).",
        },
    ]
)


def load_model_b_offline_metrics_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Prefer CSV summaries under results/model_b when present."""
    filenames: list[str] = []
    if RESULTS_MODEL_B_DIR.is_dir():
        filenames = sorted(os.listdir(RESULTS_MODEL_B_DIR))

    distr_df = MODEL_B_DISTRACTOR_METRICS_HARD.copy()
    hint_df = MODEL_B_HINT_METRICS_HARD.copy()
    for fn in filenames:
        if fn.lower().endswith(".csv"):
            path = RESULTS_MODEL_B_DIR / fn
            try:
                raw = pd.read_csv(path)
            except Exception:
                continue
            name = raw.columns.str.lower().astype(str).str.cat(sep="|")
            if "f1" in name and ("distractor" in fn.lower() or "ranker" in fn.lower()):
                distr_df = raw
            if "precision" in name and "hint" in fn.lower():
                hint_df = raw
    return distr_df, hint_df, filenames


def session_live_verifier_tp_fp_tn_fn(log_rows: list) -> tuple[int, int, int, int]:
    tp = fp = tn = fn_ = 0
    for r in log_rows:
        om = r.get("oracle_match")
        vv = r.get("verifier_verdict_ok")
        if om is None or vv is None:
            continue
        o = bool(om)
        v = bool(vv)
        if o and v:
            tp += 1
        elif (not o) and v:
            fp += 1
        elif (not o) and (not v):
            tn += 1
        else:
            fn_ += 1
    return tp, fp, tn, fn_


def init_session_state() -> None:
    defaults = {
        "article_body": "",
        "question_text": "",
        "option_a": "",
        "option_b": "",
        "option_c": "",
        "option_d": "",
        "correct_letter": "A",
        "shuffled_pairs": None,
        "display_to_original": None,
        "quiz_composed_texts": None,
        "distractors_gen": None,
        "hints_weak_to_strong": None,
        "quiz_result": None,
        "model_b_error": None,
        "model_a_error": None,
        "shuffle_seed": 0,
        "last_b_inference_ms": None,
        "last_a_inference_ms": None,
        "analytics_log": [],
        "reveal_answer_unlocked": False,
        "quiz_mode": "prd",
        "quiz_source": "",
        "verifier_ui_label": "LR Binary",
        "question_generation_log": [],
        "last_generation_candidates_preview": [],
        "distractor_fallback": False,
        "verifier_available": True,
        "hint_stage": 1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def shuffle_labeled_options(
    correct_letter: str,
    texts: dict[str, str],
    rng: random.Random,
) -> tuple[list[tuple[str, str]], str, dict[str, str]]:
    letters = ["A", "B", "C", "D"]
    items = [(L, texts[L]) for L in letters]
    rng.shuffle(items)
    display_pairs = [(letters[i], items[i][1]) for i in range(4)]
    orig_by_display = {letters[i]: items[i][0] for i in range(4)}
    orig_idx = next(i for i in range(4) if items[i][0] == correct_letter)
    gold_after_shuffle = letters[orig_idx]
    return display_pairs, gold_after_shuffle, orig_by_display


def run_inference_for_row(
    article: str,
    question: str,
    texts: dict[str, str],
    correct_letter: str,
    *,
    use_prd_quiz: bool,
) -> None:
    gold_raw = texts[str(correct_letter).strip().upper()]
    aq, qq, gq = prepare_article_question_gold_for_model_b(article, question, gold_raw)
    corr_u = str(correct_letter).strip().upper()
    race_options = {L: str(texts[L]) for L in "ABCD"}

    if use_prd_quiz:
        t0 = time.perf_counter()
        st.session_state["distractor_fallback"] = False
        d: list = []
        h: list = []
        try:
            pack_b = get_model_b_pack()
            try:
                d = generate_distractors(aq, qq, gq, pack_b)
            except Exception:
                d = []
            try:
                h = generate_hints(aq, qq, gq, pack_b)
            except Exception:
                h = []
            st.session_state["model_b_error"] = None
        except ModelBPackLoadError as e:
            st.session_state["model_b_error"] = str(e)
            d, h = [], []
        st.session_state["last_b_inference_ms"] = (time.perf_counter() - t0) * 1000.0
        usable_d = [str(x).strip() for x in d if str(x).strip()]
        if len(usable_d) >= 3:
            st.session_state["distractors_gen"] = d
            composed = compose_prd_options(corr_u, gold_raw, d)
            st.session_state["distractor_fallback"] = False
        else:
            st.session_state["distractor_fallback"] = True
            st.session_state["distractors_gen"] = []
            composed = dict(race_options)
        st.session_state["hints_weak_to_strong"] = h
    else:
        t0 = time.perf_counter()
        st.session_state["distractor_fallback"] = False
        try:
            pack_b = get_model_b_pack()
            h = generate_hints(aq, qq, gq, pack_b)
            st.session_state["model_b_error"] = None
        except ModelBPackLoadError as e:
            st.session_state["model_b_error"] = str(e)
            h = []
        st.session_state["last_b_inference_ms"] = (time.perf_counter() - t0) * 1000.0
        st.session_state["distractors_gen"] = []
        st.session_state["hints_weak_to_strong"] = h
        composed = dict(race_options)

    st.session_state["quiz_composed_texts"] = composed

    seed = int(st.session_state.get("shuffle_seed", 0)) or random.randint(0, 2**31 - 1)
    rng = random.Random(seed)
    pairs, gold_disp, disp_map = shuffle_labeled_options(corr_u, composed, rng)
    st.session_state["shuffled_pairs"] = pairs
    st.session_state["correct_letter_after_shuffle"] = gold_disp
    st.session_state["display_to_original"] = disp_map
    st.session_state["reveal_answer_unlocked"] = False
    st.session_state["hint_stage"] = 1
    for j in range(1, 4):
        st.session_state.pop(f"hint_rev_{j}", None)


def append_analytics_row(payload: dict) -> None:
    log = st.session_state.get("analytics_log", [])
    log.append(payload)
    st.session_state["analytics_log"] = log


def main() -> None:
    st.set_page_config(page_title="NeuroQuiz Lab", layout="wide")
    init_session_state()

    st.title("NeuroQuiz — Adaptive Reading Comprehension Lab")
    st.info(
        "Per course PRD default: **Quiz Mode** uses gold + three Model B distractors. "
        "**Evaluation Mode** (validation row load only) keeps original RACE options. "
        "**Model A** scores your choice with the trained feature pipeline (depends on verifier choice below)."
    )

    try:
        get_model_b_pack()
    except ModelBPackLoadError as e:
        st.error(str(e))
        return

    try:
        get_model_a_pack()
        st.session_state["model_a_error"] = None
        st.session_state["verifier_available"] = True
    except ModelAPackLoadError as e:
        st.session_state["model_a_error"] = str(e)
        st.session_state["verifier_available"] = False
        st.error(f"Verifier model failed to load — check `models/model_a/traditional/` exists. ({e})")

    try:
        get_generation_pack()
        st.session_state["generation_pack_error"] = None
    except GenerationPackLoadError as e:
        st.session_state["generation_pack_error"] = str(e)

    try:
        val_df = load_validation_df()
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
        return

    with st.sidebar:
        st.header("Sample")
        st.caption("After an attempt on **Tab 2**, pick another passage with the button below.")
        if st.button("Load random validation row"):
            with st.spinner("Loading row and running inference…"):
                row = val_df.sample(1, random_state=random.randint(0, 10_000)).iloc[0]
                st.session_state["quiz_source"] = "dataset"
                st.session_state["article_body"] = str(row["article"])
                st.session_state["question_text"] = str(row["question"])
                st.session_state["option_a"] = str(row["A"])
                st.session_state["option_b"] = str(row["B"])
                st.session_state["option_c"] = str(row["C"])
                st.session_state["option_d"] = str(row["D"])
                st.session_state["correct_letter"] = str(row["answer"]).strip().upper()
                st.session_state["shuffle_seed"] = random.randint(0, 2**31 - 1)

                texts = {
                    "A": st.session_state["option_a"],
                    "B": st.session_state["option_b"],
                    "C": st.session_state["option_c"],
                    "D": st.session_state["option_d"],
                }
                use_prd = effective_quiz_mode() != "evaluation"
                run_inference_for_row(
                    st.session_state["article_body"],
                    st.session_state["question_text"],
                    texts,
                    st.session_state["correct_letter"],
                    use_prd_quiz=use_prd,
                )

    tab_read, tab_quiz, tab_hints, tab_distractors, tab_analytics, tab_export = st.tabs(
        [
            "1 · Passage & setup",
            "2 · Quiz",
            "3 · Hints",
            "4 · Generated distractors",
            "5 · Analytics",
            "6 · Export",
        ]
    )

    with tab_read:
        st.subheader("Passage")
        article = st.text_area("Article", key="article_body", height=220)
        st.caption(
            "Model A **Question Generation** (below) uses the passage plus the **gold answer text** "
            "for your selected teacher key."
        )

        qsrc = st.session_state.get("quiz_source")
        if qsrc != "dataset":
            st.session_state.pop("_quiz_mode_widget", None)
            if str(st.session_state.get("quiz_mode")) == "evaluation":
                st.session_state["quiz_mode"] = "prd"
            st.info(
                "**Evaluation Mode** only works with **Load random validation row** in the sidebar "
                "(dataset rows). Custom **Submit** stays in **Quiz Mode (PRD)** — we switch automatically."
            )
        else:
            prev = str(st.session_state.get("quiz_mode", "prd"))
            default_idx = 1 if prev == "evaluation" else 0
            st.radio(
                "Quiz setup mode",
                ["Quiz Mode (PRD)", "Evaluation Mode"],
                index=default_idx,
                horizontal=True,
                key="_quiz_mode_widget",
                on_change=_on_quiz_mode_radio_changed,
            )

        st.subheader("Question & reference options (gold source)")
        question = st.text_area("Question", key="question_text", height=100)
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("A", key="option_a")
            st.text_input("B", key="option_b")
        with c2:
            st.text_input("C", key="option_c")
            st.text_input("D", key="option_d")
        corr = st.selectbox(
            "Teacher / evaluation key (gold letter)",
            ["A", "B", "C", "D"],
            key="correct_letter",
        )

        st.subheader("Question Generation (Model A)")
        st.caption("Uses the **gold option text** for the letter you select above.")
        if st.session_state.get("generation_pack_error"):
            st.warning(
                "Question generation is unavailable until artifacts load: "
                f"{st.session_state['generation_pack_error']}"
            )
        st.selectbox("Question type", QUESTION_GEN_TYPES, index=0, key="qg_question_type")
        st.selectbox(
            "Ranker",
            [RANKER_CHOICE_RF, RANKER_CHOICE_SVM],
            index=0,
            key="qg_ranker_choice",
            help="Random Forest is the default ranker trained in model_a_train.ipynb.",
        )
        if st.button("Generate Question from Passage") and not st.session_state.get("generation_pack_error"):
            art_txt = str(st.session_state.get("article_body") or "").strip()
            ck = str(st.session_state.get("correct_letter") or "A").strip().upper()
            gold_opt = str(st.session_state.get(f"option_{ck.lower()}") or "").strip()
            if not art_txt:
                st.warning("Article must be non-empty to generate a question.")
            elif not gold_opt:
                st.warning(
                    f"Fill in the correct answer text for **{ck}** before generating (gold option is empty)."
                )
            else:
                try:
                    pack = get_generation_pack()
                    rk = pack.rf_ranker if st.session_state["qg_ranker_choice"] == RANKER_CHOICE_RF else pack.svm_ranker
                    best_q, cand = generate_best_question(
                        art_txt,
                        gold_opt,
                        str(st.session_state.get("qg_question_type") or "What"),
                        pack.tfidf_vec,
                        pack.ohe_vec,
                        rk,
                        top_k=5,
                    )
                    st.session_state["question_text"] = best_q
                    st.session_state["last_generation_candidates_preview"] = [
                        {
                            "rank": i + 1,
                            "question": cq,
                            "similarity": round(float(sc), 4),
                            "source_snippet": sent[:240] + ("…" if len(sent) > 240 else ""),
                        }
                        for i, (cq, sc, sent) in enumerate(cand[:3])
                    ]
                    log = list(st.session_state.get("question_generation_log") or [])
                    log.append(
                        {
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "ranker": st.session_state["qg_ranker_choice"],
                            "question_type": st.session_state.get("qg_question_type"),
                        }
                    )
                    st.session_state["question_generation_log"] = log
                    st.success("Question generated by Model A")
                except Exception as ex:
                    st.error(f"Question generation failed: {ex}")

        prev_c = st.session_state.get("last_generation_candidates_preview") or []
        if prev_c:
            with st.expander("See all candidates", expanded=False):
                for row in prev_c:
                    st.markdown(
                        f"**{row['rank']}.** (similarity≈{row['similarity']})  \n{row['question']}"
                    )
                    st.caption(f"Source: {row.get('source_snippet', '')}")
                cand_labels = [
                    f"Candidate {r['rank']}: {str(r['question'])[:120]}{'…' if len(str(r['question'])) > 120 else ''}"
                    for r in prev_c
                ]
                picked = st.radio(
                    "Use candidate (apply to Question field)",
                    range(len(prev_c)),
                    format_func=lambda i: cand_labels[i],
                    key="qg_pick_idx",
                    horizontal=True,
                )
                if st.button("Apply selected candidate", key="qg_apply_pick"):
                    st.session_state["question_text"] = str(prev_c[picked]["question"])
                    st.success("Applied candidate to Question field.")

        if st.session_state.get("distractor_fallback"):
            st.warning(
                "Could not generate distractors — showing **original reference (RACE) options** instead. "
                "You can still take the quiz and use hints."
            )

        colx, coly = st.columns(2)
        with colx:
            if st.button("Submit — Run Model B", type="primary"):
                if not str(article).strip():
                    st.error("Please paste a reading passage first")
                elif not str(question).strip():
                    st.warning("Question must be non-empty.")
                elif not all(str(st.session_state[f"option_{L.lower()}"]).strip() for L in "ABCD"):
                    st.warning("All four reference options (A–D) must be non-empty.")
                else:
                    with st.spinner("Running Model B inference…"):
                        st.session_state["quiz_source"] = "custom"
                        st.session_state["quiz_mode"] = "prd"
                        texts = {L: st.session_state[f"option_{L.lower()}"] for L in "ABCD"}
                        st.session_state["shuffle_seed"] = random.randint(0, 2**31 - 1)
                        run_inference_for_row(article, question, texts, corr, use_prd_quiz=True)
                    st.success(
                        "Quiz is ready (shuffled options in **Tab 2**). Custom Submit uses **Quiz Mode (PRD)**."
                    )
        with coly:
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["field", "value"])
            w.writerow(["article", article[:2000]])
            w.writerow(["question", question])
            for L in "ABCD":
                w.writerow([L, st.session_state[f"option_{L.lower()}"]])
            w.writerow(["quiz_composed", repr(st.session_state.get("quiz_composed_texts"))])
            w.writerow(["hints_weak_to_strong", repr(st.session_state.get("hints_weak_to_strong"))])
            st.download_button(
                label="Export setup snapshot CSV",
                data=buf.getvalue(),
                file_name="neuroquiz_export.csv",
                mime="text/csv",
            )

    with tab_quiz:
        if not quiz_ready():
            st.info(UNIFIED_NOT_READY_MSG)
        else:
            pairs = st.session_state.get("shuffled_pairs")
            disp_map = st.session_state.get("display_to_original") or {}
            composed = st.session_state.get("quiz_composed_texts")
            assert pairs and composed is not None
            comp = composed

            if st.session_state.get("distractor_fallback"):
                st.warning(
                    "Could not generate distractors — quiz is using **original reference options**."
                )

            st.subheader("Question")
            qt = str(st.session_state.get("question_text") or "").strip()
            if qt:
                st.markdown(qt)
            else:
                st.caption("(Empty question — edit in Tab 1)")

            with st.expander("Article", expanded=False):
                ab = str(st.session_state.get("article_body") or "")
                st.write(truncate_article(ab, max_words=800) if ab else "—")

            if effective_quiz_mode() == "evaluation":
                st.caption(
                    "**Evaluation Mode:** original **dataset A/B/C/D** (shuffled). "
                    "Verifier sees the training distribution."
                )
            else:
                st.caption(
                    "**Quiz Mode (PRD):** options use **teacher-key gold** plus **three Model B distractors** "
                    "when distractor generation succeeds; otherwise reference options."
                )

            for lbl, txt in pairs:
                st.markdown(f"**{lbl}.** {txt}")

            choice = st.radio("Your answer", [p[0] for p in pairs], horizontal=True)
            orig_pick = disp_map.get(choice, choice)

            v_opts = list(VERIFIER_LABEL_TO_KEY.keys())
            vlabel = st.selectbox(
                "Verifier model",
                v_opts,
                key="verifier_ui_label",
                help=(
                    "LR / SVM use the full sparse pipeline (TF-IDF + OHE + cosine + lexical). "
                    "Ensemble soft-averages logistic + naive Bayes predict_proba on TF-IDF only."
                ),
            )
            vkey = VERIFIER_LABEL_TO_KEY[vlabel]

            if not st.session_state.get("verifier_available", True):
                st.error(
                    "Verifier model failed to load — Check is disabled. "
                    "Fix `models/model_a/traditional/` and restart the app."
                )

            can_check = st.session_state.get("verifier_available", True)
            if st.button("Check — Model A verifier", disabled=not can_check):
                t0 = time.perf_counter()
                try:
                    ma_go = get_model_a_pack()
                    res = predict_verification(
                        str(st.session_state["article_body"]),
                        st.session_state["question_text"],
                        comp,
                        orig_pick,
                        ma_go,
                        gold_answer_letter=str(st.session_state.get("correct_letter", "A")),
                        proba_threshold=0.5,
                        verifier=vkey,
                    )
                    st.session_state["quiz_result"] = res
                except Exception as ex:
                    st.session_state["quiz_result"] = None
                    st.error(f"Verifier error: {ex}")
                st.session_state["last_a_inference_ms"] = (time.perf_counter() - t0) * 1000.0

                res = st.session_state.get("quiz_result")
                if res:
                    oracle = res.get("oracle_match")
                    append_analytics_row(
                        {
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "mode": effective_quiz_mode(),
                            "verifier_model": vlabel,
                            "display_choice": choice,
                            "original_letter": orig_pick,
                            "verifier_prob_correct": round(res["confidence"], 4),
                            "verifier_verdict_ok": res["is_correct"],
                            "oracle_match": oracle,
                            "latency_ms_a": round(st.session_state.get("last_a_inference_ms") or 0.0, 2),
                            "latency_ms_b_last_submit": round(
                                st.session_state.get("last_b_inference_ms") or 0.0, 2
                            ),
                        }
                    )

            r = st.session_state.get("quiz_result")
            if r:
                om = r.get("oracle_match")
                if om is True:
                    st.success("**Correct!** (your choice matches the teacher key)")
                elif om is False:
                    st.error("**Incorrect** (your choice does not match the teacher key)")
                    st.info("**Need help?** Open **Tab 3 · Hints** for graduated clues.")
                st.divider()
                if r["is_correct"]:
                    st.success(
                        f"**Model A verifier:** {r['predicted_label']} — P(correct option) ≈ **{r['confidence']:.3f}**"
                    )
                else:
                    st.error(
                        f"**Model A verifier:** {r['predicted_label']} — P(correct option) ≈ **{r['confidence']:.3f}**"
                    )
                st.info(r.get("explanation", ""))
                if effective_quiz_mode() == "prd":
                    st.caption(
                        "Confidence scores are approximate — verifier was trained on original exam options, "
                        "not generated distractors."
                    )
                st.caption(
                    "Try another question: use **Load random validation row** in the **sidebar**, "
                    "or submit a new setup from Tab 1."
                )

    # hints_weak_to_strong[0] = weakest (Tab 3 Hint 1) per generate_hints
    with tab_hints:
        if not quiz_ready():
            st.info(UNIFIED_NOT_READY_MSG)
        else:
            me = st.session_state.get("model_b_error")
            if me:
                st.error(me)

            hints_list = st.session_state.get("hints_weak_to_strong") or []
            n_avail = len(hints_list)
            wc = len(str(st.session_state.get("article_body") or "").split())

            if not hints_list:
                if wc > 0 and wc < ARTICLE_SHORT_WORDS_HINT:
                    st.warning(
                        "**Passage too short to extract hints** — try a longer article (about "
                        f"{ARTICLE_SHORT_WORDS_HINT}+ words) or rerun from Tab 1."
                    )
                else:
                    st.warning("No hints for this passage (empty splits or inference error).")
            else:
                st.caption(
                    "Hints unlock one at a time (**weakest → stronger**). Confirm each hint before revealing the next."
                )
                n_hints = min(3, n_avail)
                hs = int(st.session_state.get("hint_stage", 1))
                hs = max(1, min(hs, n_hints))

                for hi in range(1, hs + 1):
                    st.markdown(f"#### Hint {hi}")
                    st.write(hints_list[hi - 1])
                    if hi < hs:
                        st.caption("Unlocked ✓")
                    else:
                        st.checkbox(f"I have reviewed hint {hi}", key=f"hint_rev_{hi}")

                if hs < n_hints:
                    if st.button(f"Reveal Hint {hs + 1}", key="btn_reveal_next_hint"):
                        if st.session_state.get(f"hint_rev_{hs}", False):
                            st.session_state["hint_stage"] = hs + 1
                            st.rerun()
                        else:
                            st.warning(f"Check **reviewed hint {hs}** first.")
                elif hs == n_hints:
                    all_rev = all(
                        bool(st.session_state.get(f"hint_rev_{j}", False))
                        for j in range(1, n_hints + 1)
                    )
                    ck = str(st.session_state.get("correct_letter", "A"))
                    composed = st.session_state.get("quiz_composed_texts") or {}
                    ans_text = composed.get(ck, "")

                    if all_rev:
                        if st.button("Reveal answer (after hints)", key="btn_reveal_ans_hints"):
                            st.session_state["reveal_answer_unlocked"] = True
                    else:
                        st.caption("Review and confirm **all** unlocked hints above to enable **Reveal answer**.")

                    if st.session_state.get("reveal_answer_unlocked") and all_rev:
                        qr = st.session_state.get("quiz_result") or {}
                        extra = str(qr.get("explanation", "") or "").strip()
                        ok_line = (
                            f"**Answer ({ck}) — teacher key:** {ans_text}\n\n"
                            "This is the reference answer used for grading in this lab."
                        )
                        if extra:
                            ok_line += f"\n\n_Last Model A verifier note (after Check):_ {extra}"
                        st.success(ok_line)

    with tab_distractors:
        if not quiz_ready():
            st.info(UNIFIED_NOT_READY_MSG)
        else:
            if st.session_state.get("model_b_error"):
                st.error(st.session_state["model_b_error"])
            distr = st.session_state.get("distractors_gen") or []
            if effective_quiz_mode() == "evaluation":
                st.info("Distractor generation skipped in Evaluation Mode.")
            if st.session_state.get("distractor_fallback"):
                st.warning(
                    "Distractor generation did not return three usable candidates — **reference options** "
                    "were kept for the quiz instead."
                )
            if not distr:
                if effective_quiz_mode() != "evaluation" and not st.session_state.get(
                    "distractor_fallback"
                ):
                    st.info("Submit from Tab 1 to generate distractors.")
            for i, d in enumerate(distr, start=1):
                st.markdown(f"**Suggested distractor {i}**")
                st.write(d)

    with tab_analytics:
        log = st.session_state.get("analytics_log") or []
        scored = [x for x in log if x.get("oracle_match") is not None]
        n_att = len(scored)
        n_cor = sum(1 for x in scored if x.get("oracle_match") is True)
        sess_acc = (n_cor / n_att) if n_att else None
        lat_a_all = [float(x["latency_ms_a"]) for x in log if x.get("latency_ms_a") is not None]
        lat_b_all = [
            float(x["latency_ms_b_last_submit"]) for x in log if x.get("latency_ms_b_last_submit") is not None
        ]
        avg_a = float(np.mean(lat_a_all)) if lat_a_all else None
        avg_b = float(np.mean(lat_b_all)) if lat_b_all else None
        off_acc, off_src = offline_reference_val_accuracy()

        st.subheader("Section 1 — Live session")

        r1a, r1b, r1c, r1d = st.columns(4)
        r1a.metric("Attempts (oracle)", str(n_att))
        r1b.metric("Correct", str(n_cor))
        r1c.metric("Avg Check latency (ms)", f"{avg_a:.1f}" if avg_a is not None else "—")
        r1d.metric(
            "Avg logged B latency (ms)",
            f"{avg_b:.1f}" if avg_b is not None else "—",
        )

        r2a, r2b = st.columns(2)
        r2a.metric(
            "Session accuracy (oracle)",
            f"{sess_acc:.1%}" if sess_acc is not None else "—",
        )
        r2b.metric(
            "Offline val (LR ref.)",
            f"{off_acc:.4f}" if off_acc is not None else "—",
        )
        st.caption(
            f"{off_src} Session accuracy reflects this session only vs the teacher key; "
            "offline val is from training export and is **not strictly comparable** to PRD runs."
        )

        st.markdown("###### Session log")
        if log:
            st.dataframe(pd.DataFrame(log), use_container_width=True)
            acsv = io.StringIO()
            pd.DataFrame(log).to_csv(acsv, index=False)
            st.download_button(
                "Download session analytics CSV",
                data=acsv.getvalue(),
                file_name="neuroquiz_analytics.csv",
                mime="text/csv",
            )
        else:
            st.info("Run **Check** in the Quiz tab to append verifier rows.")

        st.markdown("###### Live confusion (verdict vs oracle)")
        tp, fp, tn, fn_ = session_live_verifier_tp_fp_tn_fn(log)
        n_sess = tp + fp + tn + fn_
        if n_sess < 2:
            st.info("Answer at least **2** graded questions for a stable 2×2 confusion snapshot.")
        else:
            cm_live = pd.DataFrame(
                [[tn, fp], [fn_, tp]],
                index=["oracle: incorrect (0)", "oracle: correct (1)"],
                columns=[
                    "verifier predicts unlikely correct (False)",
                    "verifier predicts likely correct (True)",
                ],
            )
            st.dataframe(cm_live, use_container_width=True)
            st.caption(
                "Rows: teacher truth (oracle). Columns: Model A verifier at Check (threshold 0.5)."
            )

        qg_hist = st.session_state.get("question_generation_log") or []
        st.markdown("###### Question generation (this session)")
        st.caption(f"Questions generated: **{len(qg_hist)}**")
        if qg_hist:
            st.dataframe(pd.DataFrame(qg_hist), use_container_width=True, hide_index=True)

        st.subheader("Section 2 — Model A offline metrics")
        if OFFLINE_METRICS_CSV.exists():
            try:
                st.dataframe(pd.read_csv(OFFLINE_METRICS_CSV), use_container_width=True)
            except Exception:
                st.caption(f"Could not read {OFFLINE_METRICS_CSV}")
        else:
            st.caption(
                f"Optional: add `notebooks/results/model_a/classification_results.csv` "
                f"(not found at `{OFFLINE_METRICS_CSV}`)."
            )

        st.subheader("Section 3 — Validation confusion matrices (LR & SVM)")
        if not Y_VAL_BINARY_PATH.exists() or not X_VAL_COMBINED_PATH.exists():
            st.warning(
                "Missing processed validation tensors for confusion matrices "
                f"(`{Y_VAL_BINARY_PATH.name}` / `{X_VAL_COMBINED_PATH.name}`). "
                "Rebuild `data/processed/` from preprocessing."
            )
        else:
            try:
                bust = _artifact_mtimes(
                    (
                        Y_VAL_BINARY_PATH,
                        X_VAL_COMBINED_PATH,
                        MODELA_TRAD_PATH / "lr_binary.pkl",
                        MODELA_TRAD_PATH / "svm_binary.pkl",
                    )
                )
                y_v, pred_lr_v, pred_svm_v = cached_model_a_val_binary_predictions(bust)
                col_lr, col_svm = st.columns(2)
                with col_lr:
                    fig_lr = plot_model_a_confusion_fig(
                        y_v,
                        pred_lr_v,
                        title="Model A — LR Binary Confusion Matrix (val set)",
                    )
                    st.pyplot(fig_lr)
                    plt.close(fig_lr)
                with col_svm:
                    fig_svm = plot_model_a_confusion_fig(
                        y_v,
                        pred_svm_v,
                        title="Model A — SVM Binary Confusion Matrix (val set)",
                    )
                    st.pyplot(fig_svm)
                    plt.close(fig_svm)
            except Exception as ex:
                st.error(f"Could not build offline validation confusion matrices: {ex}")

        st.subheader("Section 4 — Model B offline metrics")
        distr_off, hint_off, mb_files = load_model_b_offline_metrics_tables()
        if RESULTS_MODEL_B_DIR.is_dir() and mb_files:
            st.caption(f"`results/model_b/` contents: {', '.join(mb_files)}")
        if not RESULTS_MODEL_B_DIR.is_dir():
            st.caption("`results/model_b/` folder not found — using notebook hard-coded metrics.")

        st.markdown("**Distractor ranker** (hold-out diagnostics)")
        st.dataframe(distr_off.reset_index(drop=True), use_container_width=True, hide_index=True)

        st.markdown("**Hint scorer**")
        st.dataframe(hint_off.reset_index(drop=True), use_container_width=True, hide_index=True)

        if MODEL_B_HOLDOUT_CM_PNG.exists():
            st.image(
                str(MODEL_B_HOLDOUT_CM_PNG.resolve()),
                caption="Model B distractor ranker — holdout confusion matrix",
            )
        else:
            st.caption(f"PNG not found: `{MODEL_B_HOLDOUT_CM_PNG}`")

        st.caption(
            f"Last Model B inference (Tab 1 Submit / random row): "
            f"**{st.session_state.get('last_b_inference_ms') or '—'}** ms"
        )

    with tab_export:
        st.write("Use **Export setup snapshot** on Tab 1 or **Download session analytics CSV** on Tab 5.")

    st.markdown("---")
    st.caption(
        "Artifacts: `models/model_a/traditional/` + `models/model_b/traditional/` · "
        f"Data: `{VAL_CSV}`"
    )


if __name__ == "__main__":
    main()
