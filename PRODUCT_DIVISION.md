# Product division — NeuroQuiz (RACE RC project)

**Team size:** 2 students  

| Member | Role |
|--------|------|
| **Member 1** (local) | You |
| **Member 2** (remote) | Teammate |

> **Note:** If you previously wrote “Member 3” for Model B / Streamlit / report, that maps to **Member 1** here — there is no third person on the team.

---

## Work split

### Member 2 — EDA, preprocessing, Model A

| Area | Status | Artifacts / location |
|------|--------|----------------------|
| Exploratory data analysis | Done (per team) | `notebooks/EDA.ipynb`, plots under `eda_plots/` |
| Preprocessing pipeline | Done (per team) | `notebooks/preprocessing.ipynb` |
| Model A (Q&A generation + answer verification) | In progress | `notebooks/model_a_train.ipynb`, `models/model_a/` |

**Member 2 owns:** keeping `train_split.csv` / `val_split.csv` definitions consistent, re-running preprocessing when splits change, and exposing any **Model A inference contract** (function names or saved checkpoints) that the UI will call.

### Member 1 — Model B, Streamlit, report

| Area | Ownership |
|------|-----------|
| **Model B** | Distractor generation + ranked hints (classical ML per course PRD) |
| **Streamlit app** | All four required views (article input, quiz, hints, developer/analytics), wiring Model A + B |
| **Final report** | Write-up, figures, limitations, ethics |

**Member 1 owns:** `models/model_b/`, `ui/` (or agreed app entry path), and report PDF under `report/` when ready.

---

## Integration touchpoints (avoid duplicate work)

1. **Data:** Always use **`data/processed/train_split.csv`** and **`val_split.csv`** (stratified 80/20 from EDA), not raw Kaggle `train.csv` alone.
2. **Text cleaning:** Reuse the same rules as `preprocessing.ipynb` (`clean_text`, article truncation at 500 words after cleaning) so offline metrics match the app.
3. **Vectorizers:** After preprocessing runs, shared copies are written to both:
   - `models/model_a/traditional/`
   - `models/model_b/traditional/`  
   (`tfidf_vectorizer.pkl`, `ohe_vectorizer.pkl`, `cos_vectorizer.pkl`)  
   Model B should load from **`models/model_b/traditional/`** and must not refit on validation/test data.
4. **Interface:** Agree on a small Python API, e.g. `predict_verification(...)` (Member 2) and `generate_distractors(...)`, `generate_hints(...)` (Member 1), both callable from one Streamlit file.

---

## Preprocessed data & findings (from project notebooks)

> **Local repo note:** `data/processed/` is listed in `.gitignore`, so processed files may be **empty on your machine** until you run EDA + preprocessing or copy artifacts from Member 2.

### Splits and schema (from `preprocessing.ipynb` outputs)

- **Train:** 70,281 rows × 15 columns  
- **Val:** 17,571 rows × 15 columns  
- **Columns:** `id`, `article`, `question`, `A`, `B`, `C`, `D`, `answer`, `article_len`, `question_len`, `A_len`, `B_len`, `C_len`, `D_len`, `question_type`  
- **Answer labels:** Roughly balanced four-way (`A`–`D` ≈ quarter each on train and val).

### Cleaning & truncation

- Lowercase, strip URLs/punctuation/standalone numbers, collapse whitespace.
- **Articles** truncated to **500 words** (aligned with ~95th percentile passage length).
- Questions and options are **not** truncated.

### Features already built for verification (Model A–oriented)

- **Verification text:** `article + article + question + option` (article repeated to emphasize passage vs short options — documented in preprocessing).
- **TF-IDF** and **bag-of-words / OHE** matrices, **cosine block** features (article–question–option similarities via a **article-fitted** `cos_vectorizer`), **lexical** features, plus **combined** sparse matrices saved as `.npz` / `.npy` when the notebook completes.

### Design choices to respect for Model B

- **Class balance (~25% per option):** preprocessing notes say **not** to rely on `class_weight='balanced'` for multiclass heads where that assumption was already analyzed.
- **Question type** (`question_type`) exists on the dataframe — useful if Model B tunes templates or hint ordering by WH-question category.

---

## How Member 1 can start Model B (practical sequence)

### 1. Sync artifacts

- Get from Member 2 (or regenerate locally): `train_split.csv`, `val_split.csv`, and the preprocessing outputs under `data/processed/`, plus `models/model_b/traditional/*.pkl`.

### 2. Define Model B outputs clearly

Per PRD:

- **Distractors:** Given `article`, `question`, and the **correct answer string** (the text of option `answer`, not only the letter), return **three** wrong options plausible but incorrect.
- **Hints:** Return **ranked / graduated** cues (extractive sentences from the article), without revealing the final answer upfront in hint 1.

### 3. Distractors — baseline pipeline (classical)

1. **Candidate extraction:** Noun phrases / frequent content words / sentences from the article (exclude the gold span if possible).
2. **Features:** For each candidate, compute overlap and similarity signals already aligned with course material — e.g. cosine similarity vs correct answer text using **`cos_vectorizer`** or OHE/TF-IDF vectors; frequency; character overlap; optionally Word2Vec neighbors (passage‑constrained).
3. **Ranker:** Train **Logistic Regression** or **Random Forest** on labeled pairs `(candidate, is_good_distractor)` built from negatives mined from passages and positives from dataset distractors.
4. **Selection:** Take top‑3 scoring candidates that are **not** the correct answer and enforce a simple **diversity** rule (pairwise cosine distance threshold or penalize duplicates).

Validate with precision/recall/F1 vs held‑out distractors where labels exist, plus spot human ratings (PRD mentions Likert plausibility).

### 4. Hints — fast path

1. Split `article_clean` into sentences.
2. Vectorize sentences and the **question** with `cos_vectorizer` (already trained on articles — for sentence text, same transform as preprocessing uses for snippets).
3. Rank sentences by **cosine similarity to the question** (and optionally to the correct answer span with a smaller weight).
4. Map ranks to **Hint 1 = vaguer / lower similarity**, **Hint 3 = strongest** — or the inverse, but keep the UX spec from the PRD (ordered reveal).

Train optional **sentence scorer** (logistic regression on length, position, overlap features) if pure similarity is too noisy.

### 5. Persist for Streamlit

- Save Model B rankers / word2vec aux files with **`joblib`** under `models/model_b/traditional/`.
- Implement `inference.py` helpers or a thin module Member 2 can import from the same repo layout assumed in `PRD/AL2002_LabProject.md`.

### 6. Notebook suggestion

Add `notebooks/model_b_train.ipynb` mirroring `model_a_train.ipynb`: load splits → mine distractor training data → train ranker → evaluate on val → export `joblib` models.

---

## Quick checklist

- [ ] Member 2: confirm latest `train_split.csv` / `val_split.csv` + processed files are shared (Drive/USB/branch with LFS if large).
- [ ] Member 1: `notebooks/model_b_train.ipynb` + `models/model_b/traditional/` checkpoints.
- [ ] Both: agree on function signatures for Streamlit (`ui/app.py`).
- [ ] Member 1: stub Streamlit early with **fake** Model B outputs, then swap in trained models.
