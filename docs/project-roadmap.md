# Project Roadmap

## Phase 1 — Setup & Data
- [x] Repo + venv + dependencies
- [x] Download Amazon reviews dataset
- [x] Data loader with format auto-detection
- [x] Validation (missing values, duplicates, empty reviews)

## Phase 2 — EDA
- [x] Class distribution plot
- [x] Review length distribution
- [x] Top words per class
- [x] Save EDA summary to reports/

## Phase 3 — Preprocessing & Features
- [x] Text cleaner (HTML, URLs, contractions, punctuation)
- [x] Negation-aware stopword removal
- [x] Stemming option
- [x] Wrap as sklearn transformer
- [x] TF-IDF with configurable params

## Phase 4 — Training & Evaluation
- [x] Baseline model comparison (NB, LogReg, SVM)
- [x] N-gram experiments
- [x] Vocab size experiments
- [x] Preprocessing comparison
- [x] Hyperparameter tuning (GridSearchCV)
- [x] Final evaluation on test set
- [x] Error analysis

## Phase 5 — Deployment & Testing
- [x] Save pipeline with joblib
- [x] Prediction module
- [x] Streamlit app
- [x] Unit tests
- [x] README + docs
