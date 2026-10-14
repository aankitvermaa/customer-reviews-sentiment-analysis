# Implementation Plan

Rough plan for building this, step by step.

## 1. Setup

- Python venv, install pandas/sklearn/nltk/streamlit/etc.
- Set up the folder structure 
    -src/
    -app/
    -tests/
    -data/
    -models/
    -reports/

- `.gitignore` for venv, data files, pycache

## 2. Get the Data

Download the Amazon reviews dataset from Kaggle (https://www.kaggle.com/datasets/bittlingmayer/amazonreviews/data?select=test.ft.txt.bz2) and put it in `data/raw/`. I'm using the fastText-format file (`train.ft.txt.bz2`) but I also added support for the CSV variants in case I want to switch later.

## 3. Explore the Data

Look at class balance, review lengths, most common words per class. Save some plots to `reports/figures/`. This helps me understand what I'm working with before jumping into modelling.

## 4. Preprocessing

Build a text cleaner that handles:
- HTML tags and entities (some reviews have `<br/>` tags)
- URLs
- Lowercasing
- Expanding contractions like "doesn't" → "does not" (important for negation)
- Stripping punctuation

negation words like "not" are stopwords and removing them would be bad for sentiment. So I'll keep negations even in stopword mode and test both.

Wrapping the cleaner as a sklearn transformer so it can go inside the pipeline.

## 5. Features + Models

TF-IDF for features. Start simple with unigrams, then try bigrams.

Models to compare:
1. Multinomial Naive Bayes (fast baseline)
2. Logistic Regression (usually solid for text)
3. Linear SVM (good with sparse high-dim data)

## 6. Experiments

Run a sequence of experiments, each building on the best config so far:
- A: baseline comparison of the 3 models
- B: unigram vs bigram
- C: vocab size (10k / 30k / 50k)
- D: preprocessing mode (minimal / stopwords / stem)
- E: tune regularization + sublinear_tf with GridSearchCV

All using stratified 5-fold CV on training data. Test set stays untouched.

## 7. Evaluation + Error Analysis

After picking the best config:
- Fit on full training set, evaluate on test set
- Confusion matrix, precision/recall/F1
- Look at misclassified reviews — tag them by likely cause (negation issues, mixed sentiment, short reviews, etc.)
- Extract top positive/negative features from the model coefficients

## 8. Save the Pipeline

Package cleaner + TF-IDF + classifier into one sklearn Pipeline, save with joblib. This way inference just needs `joblib.load()` + `pipeline.predict()`.

## 9. Streamlit App

Simple UI: text area, analyze button, show prediction and confidence. Load the saved pipeline on startup with `@st.cache_resource`.

## 10. Tests

- Test preprocessing (HTML removal, contraction expansion, edge cases)
- Test prediction (valid input, empty input, special chars, missing model file)
- Test data loading (different formats, validation)

## Done When

A new user can clone → install → place dataset → train or load model → run app → get predictions. And the code is clean enough to understand.
