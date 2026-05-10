# NeuroQuiz

Intelligent reading-comprehension quiz UI (course **AL2002**): **Model A** answer verification (`lr_binary` + preprocessing-aligned sparse features), **Model B** distractor and extractive hints, wired in **Streamlit**.

## Prerequisites

1. **`data/processed/`** outputs from **`notebooks/EDA.ipynb`** and **`notebooks/preprocessing.ipynb`** (this folder may be `.gitignored`; generate locally or unzip a teammate’s artifact bundle).

2. **Pickles**

   - `models/model_a/traditional/`: **`lr_binary.pkl`**, **`tfidf_vectorizer.pkl`**, **`ohe_vectorizer.pkl`**, **`cos_vectorizer.pkl`**
   - `models/model_b/traditional/`: **`distractor_ranker.pkl`**, **`hint_scorer.pkl`**, **`model_b_config.pkl`**, **`tfidf_vectorizer.pkl`**, **`cos_vectorizer.pkl`**, **`random_forest_distractor.pkl`** (training: `notebooks/model_b_train.ipynb`)

3. **`data/processed/val_split.csv`** for “Load random validation row.”

## Run (submission demo)

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

From the **repository root**:

```bash
streamlit run ui/app.py
```

## PRD coverage (quick map)

| PRD expectation | Implementation |
|-----------------|----------------|
| Passage + question + MC options | Tab **1 · Passage & setup** |
| Submit runs Model B (hints + distractors) | **Submit — run Model B** |
| Quiz: gold + three Model B distractors | Composed internally; Tab **2 · Quiz** |
| Graduated hints | Tab **3 · Hints** (expanders + review checkboxes → **Reveal answer**) |
| Model A verifier (not oracle string-match) | **Check — Model A verifier** uses `predict_verification` + `lr_binary` |
| Transparency / disclosures | Banner on load; verifier explanation text after **Check** |
| Analytics / export | Tab **5 · Analytics** (session CSV); Tab **6 · Export**; optional offline CSV `notebooks/results/model_a/classification_results.csv` |

## Architecture

- [`ui/app.py`](ui/app.py): Streamlit entry, cached model packs, quiz shuffle + display→original letter map.
- [`src/model_a/features.py`](src/model_a/features.py): One-row **`X_combined`** (18,019 features) matching [`notebooks/preprocessing.ipynb`](notebooks/preprocessing.ipynb).
- [`src/model_a/inference.py`](src/model_a/inference.py): `compose_prd_options`, `predict_verification`.
- [`src/model_b/inference.py`](src/model_b/inference.py): distractors + hints (notebook-aligned).
