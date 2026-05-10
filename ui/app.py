"""
NeuroQuiz Streamlit entry point. Run from repo root:
  streamlit run ui/app.py
"""

from __future__ import annotations

import csv
import io
import random
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_a.inference import predict_verification  # noqa: E402
from src.model_b.inference import (  # noqa: E402
    ModelBPackLoadError,
    generate_distractors,
    generate_hints,
    load_model_pack,
    prepare_article_question_gold_for_model_b,
)

VAL_CSV = ROOT / "data" / "processed" / "val_split.csv"
REQUIRED_COLUMNS = ("article", "question", "A", "B", "C", "D", "answer")


@st.cache_resource
def get_model_b_pack():
    return load_model_pack()


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
        "distractors_gen": None,
        "hints_weak_to_strong": None,
        "quiz_result": None,
        "model_b_error": None,
        "shuffle_seed": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def shuffle_labeled_options(
    correct_letter: str,
    texts: dict[str, str],
    rng: random.Random,
) -> tuple[list[tuple[str, str]], str]:
    """
    Shuffle raw (original letter, text) pairs together, then re-label displays A–D in shuffled order.
    Gold display letter is positional (duplicate option texts still map correctly).
    """
    letters = ["A", "B", "C", "D"]
    items = [(L, texts[L]) for L in letters]
    rng.shuffle(items)
    display_pairs = [(letters[i], items[i][1]) for i in range(4)]
    orig_idx = next(i for i in range(4) if items[i][0] == correct_letter)
    gold_after_shuffle = letters[orig_idx]
    return display_pairs, gold_after_shuffle


def run_inference_for_row(article: str, question: str, texts: dict[str, str], correct_letter: str) -> None:
    gold_raw = texts[correct_letter]
    aq, qq, gq = prepare_article_question_gold_for_model_b(article, question, gold_raw)

    try:
        pack = get_model_b_pack()
        d = generate_distractors(aq, qq, gq, pack)
        h = generate_hints(aq, qq, gq, pack)
        st.session_state["model_b_error"] = None
    except ModelBPackLoadError as e:
        st.session_state["model_b_error"] = str(e)
        d, h = [], []

    st.session_state["distractors_gen"] = d
    st.session_state["hints_weak_to_strong"] = h

    seed = int(st.session_state.get("shuffle_seed", 0)) or random.randint(0, 2**31 - 1)
    rng = random.Random(seed)
    pairs, gold_disp = shuffle_labeled_options(correct_letter, texts, rng)
    st.session_state["shuffled_pairs"] = pairs
    st.session_state["correct_letter_after_shuffle"] = gold_disp


def main() -> None:
    st.set_page_config(page_title="NeuroQuiz Lab", layout="wide")
    init_session_state()

    st.title("NeuroQuiz — Adaptive Reading Comprehension Lab")
    st.info(
        "Options / hints / distractors are **AI-assisted** for this demo. "
        "Verification uses a **stub** Model A until production weights are integrated."
    )

    try:
        get_model_b_pack()
    except ModelBPackLoadError as e:
        st.error(str(e))
        return

    try:
        val_df = load_validation_df()
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
        return

    with st.sidebar:
        st.header("Sample")
        if st.button("Load random validation row"):
            row = val_df.sample(1, random_state=random.randint(0, 10_000)).iloc[0]
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
            run_inference_for_row(
                st.session_state["article_body"],
                st.session_state["question_text"],
                texts,
                st.session_state["correct_letter"],
            )

    tab_read, tab_quiz, tab_hints, tab_distractors, tab_export = st.tabs(
        ["1 · Passage & setup", "2 · Quiz", "3 · Hints", "4 · Generated distractors", "5 · Export"]
    )

    with tab_read:
        st.subheader("Passage")
        article = st.text_area("Article", key="article_body", height=220)
        st.subheader("Question & options")
        question = st.text_area("Question", key="question_text", height=100)
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("A", key="option_a")
            st.text_input("B", key="option_b")
        with c2:
            st.text_input("C", key="option_c")
            st.text_input("D", key="option_d")
        corr = st.selectbox("Correct answer (teacher key)", ["A", "B", "C", "D"], key="correct_letter")

        colx, coly = st.columns(2)
        with colx:
            if st.button("Run models on this setup"):
                if not article.strip() or not question.strip():
                    st.warning("Article and question must be non-empty.")
                elif not all(str(st.session_state[f"option_{L.lower()}"]).strip() for L in "ABCD"):
                    st.warning("All four options (A–D) must be non-empty.")
                else:
                    texts = {L: st.session_state[f"option_{L.lower()}"] for L in "ABCD"}
                    # selectbox(key="correct_letter") owns that key; do not assign after render.
                    st.session_state["shuffle_seed"] = random.randint(0, 2**31 - 1)
                    run_inference_for_row(article, question, texts, corr)
                    st.success("Updated distractors / hints / shuffled quiz layout.")
        with coly:
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["field", "value"])
            w.writerow(["article", article[:2000]])
            w.writerow(["question", question])
            for L in "ABCD":
                w.writerow([L, st.session_state[f"option_{L.lower()}"]])
            w.writerow(["hints_weak_to_strong", repr(st.session_state.get("hints_weak_to_strong"))])
            if st.download_button(
                label="Export snapshot CSV",
                data=buf.getvalue(),
                file_name="neuroquiz_export.csv",
                mime="text/csv",
            ):
                pass

    with tab_quiz:
        pairs = st.session_state.get("shuffled_pairs")
        gold_after = st.session_state.get("correct_letter_after_shuffle", "A")
        if not pairs:
            st.warning("Use Tab 1 — load a random sample or run models — to populate the quiz.")

        opts = pairs or []
        if opts:
            st.markdown("Labels A–D are shuffled whenever you regenerate.")
            txt_by_l = dict(opts)
            for lbl, txt in opts:
                st.markdown(f"**{lbl}.** {txt}")

            choice = st.radio("Your answer", [p[0] for p in opts], horizontal=True)
            picked = txt_by_l[choice]
            truth = txt_by_l[gold_after]

            if st.button("Check with Model A (stub verifier)"):
                res = predict_verification(
                    question=st.session_state["question_text"],
                    selected_answer_text=picked,
                    correct_answer_text=truth,
                )
                st.session_state["quiz_result"] = res

            r = st.session_state.get("quiz_result")
            if r:
                if r["is_correct"]:
                    st.success(f"Verifier: {r['predicted_label']} (confidence {r['confidence']:.2f})")
                else:
                    st.error(f"Verifier: {r['predicted_label']} (confidence {r['confidence']:.2f})")

    # hints_weak_to_strong[0] = weakest (Tab 3 Hint 1) per generate_hints
    with tab_hints:
        me = st.session_state.get("model_b_error")
        if me:
            st.error(me)
        hints_list = st.session_state.get("hints_weak_to_strong") or []
        if not hints_list:
            st.warning("No hints for this passage (empty splits or inference error).")
        else:
            st.caption(
                "Hint numbering: 1 = weakest cue, last = strongest — matches `hints_weak_to_strong` ordering."
            )
            for idx, h in enumerate(hints_list, start=1):
                st.markdown(f"**Hint {idx}**")
                st.write(h)

    with tab_distractors:
        if st.session_state.get("model_b_error"):
            st.error(st.session_state["model_b_error"])
        distr = st.session_state.get("distractors_gen") or []
        if not distr:
            st.info("Run inference from Tab 1 to populate distractors.")
        for i, d in enumerate(distr, start=1):
            st.markdown(f"**Suggested distractor {i}**")
            st.write(d)

    with tab_export:
        st.write("Download a CSV snapshot from Tab 1 (Export snapshot).")

    st.markdown("---")
    st.caption("Artifacts: `models/model_b/traditional/` · Data: `data/processed/val_split.csv`.")


if __name__ == "__main__":
    main()
