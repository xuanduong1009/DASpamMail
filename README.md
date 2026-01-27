# Enron Spam Filter (Do an 19)

This project builds a spam/ham email classifier using TF-IDF and ML models
(Multinomial Naive Bayes and Linear SVM). The pipeline includes email parsing,
text cleaning, vectorization, training, and evaluation.

Dataset layout (folder-based):
- data/bilingual/ham and data/bilingual/spam (current combined EN + VI)
- enron1/ham and enron1/spam (if you keep the original EN-only dataset)

## Full Setup (Windows PowerShell)

1) Create a virtual environment and install dependencies:
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt

2) Train models and save artifacts (choose dataset):
   # EN + VI (current default)
   python -m src.models.train --data-dir data\bilingual

   # EN only (if enron1 exists)
   python -m src.models.train --data-dir enron1

3) Evaluate saved model on test split:
   python -m src.models.evaluate --model models/best_model.joblib

4) Predict for a new email:
   python -m src.models.predict --email-file path\to\email.txt
   # or
   python -m src.models.predict --text "free money now"

## Demo (Streamlit)

Run the web demo UI:

1) Make sure you have trained once (creates models/best_model.joblib)
2) Start the app:
   streamlit run app.py

## English + Vietnamese

To support both languages, put Vietnamese spam/ham emails in the same folder
structure (e.g., `data/bilingual/ham` and `data/bilingual/spam`), then retrain:

   python -m src.models.train --data-dir data\bilingual

The vectorizer uses a combined English+Vietnamese stopword list configured in
`src/config.py`. You can edit `data/stopwords_vi.txt` to adjust Vietnamese stopwords
or disable them by setting `USE_VIETNAMESE_STOPWORDS = False`.

## Suggested Run Flow (what to run, in order)

1) Setup env + install
   - python -m venv .venv
   - .venv\Scripts\activate
   - pip install -r requirements.txt

2) Train (choose dataset)
   - python -m src.models.train --data-dir data\bilingual

3) Evaluate metrics
   - python -m src.models.evaluate --model models/best_model.joblib

4) Demo
   - streamlit run app.py

## File/Folder Meaning (so reviewers know how to use)

- data/bilingual/ham, data/bilingual/spam
  Bilingual dataset (EN + VI). Each file is one email.
- data/processed/train.csv, data/processed/test.csv
  Train/test splits created during training.
- data/interim/emails_raw.csv
  Parsed raw emails (text + label + path).
- data/stopwords_vi.txt
  Vietnamese stopwords used by TF-IDF (editable).
- models/nb_best.joblib, models/svm_best.joblib
  Best Naive Bayes and SVM models from training.
- models/best_model.joblib
  Best model chosen by F1 (used by demo and predict).
- reports/results/metrics.csv
  Baselines + NB + SVM metrics (precision/recall/F1 + params).
- reports/results/eval_metrics.csv
  Metrics for the selected best model.
- app.py
  Streamlit demo UI.
- src/
  Core pipeline code (load → preprocess → TF-IDF → train → eval).

## Notes

- If you update scikit-learn version, re-train to avoid version warnings.
- The demo uses `data/bilingual` by default in `app.py`.

## Outputs

- data/interim/emails_raw.csv
- data/processed/train.csv and data/processed/test.csv
- models/nb_best.joblib, models/svm_best.joblib, models/best_model.joblib
- reports/results/metrics.csv
- reports/results/eval_metrics.csv
