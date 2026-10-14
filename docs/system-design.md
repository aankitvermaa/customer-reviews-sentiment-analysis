# System Design

## Overview

The project has two modes: training and inference.

**Training** runs the full pipeline — load data, clean text, extract TF-IDF features, train classifiers, evaluate, save the best one. This is done once (or whenever I want to retrain).

**Inference** just loads the saved pipeline and predicts. The Streamlit app does this.

## Architecture

```
Training:
  Raw data → data_loader → validation → text cleaning → TF-IDF → model training → evaluation → save pipeline

Inference:
  User review → saved pipeline (cleaner + TF-IDF + classifier) → prediction → Streamlit UI
```

The key design choice is that preprocessing goes inside the sklearn Pipeline. That way there's no risk of applying different cleaning at train time vs inference time — it's all one object.

## Modules

| Module | What it does |
|---|---|
| `data_loader.py` | Reads different Amazon review formats, validates data, handles sampling |
| `preprocessing.py` | Text cleaning (HTML, URLs, contractions, etc.), wrapped as sklearn transformer |
| `features.py` | Builds TF-IDF vectorizer and classifier, assembles pipelines |
| `train.py` | Runs experiments, tracks results, does model selection, saves final pipeline |
| `evaluate.py` | Metrics, confusion matrix, top features, error analysis |
| `eda.py` | Exploratory plots and stats |
| `predict.py` | Loads saved pipeline, validates input, returns prediction + confidence |
| `app/app.py` | Streamlit frontend |

## Saved Model

Everything gets saved as one file: `models/sentiment_pipeline.pkl`

It contains the text cleaner, TF-IDF vectorizer (with fitted vocabulary), and the trained classifier. So to predict you just do `pipeline.predict(["some review text"])`.

I used joblib for serialization since it handles large numpy arrays (like the TF-IDF vocabulary) better than pickle.
