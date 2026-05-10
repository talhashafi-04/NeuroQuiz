# NeuroQuiz

## Streamlit demo

Install dependencies (recommended: use a virtual environment on Debian/Ubuntu PEP 668 setups):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

From the repository root:

```bash
streamlit run ui/app.py
```

The app expects `data/processed/val_split.csv` (built by the preprocessing notebooks) and Model B pickles under `models/model_b/traditional/`.
