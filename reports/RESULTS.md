# Experiment Results

Dataset file: `train.ft.txt.bz2` | train rows: 80,000 | test rows: 20,000

## Selected configuration

```json
{
  "preprocessing": "minimal",
  "model": "linear_svm",
  "tfidf": {
    "ngram_range": [
      1,
      2
    ],
    "max_features": 50000,
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": true
  },
  "hyperparameters": {
    "C": 1.0
  }
}
```

Cross-validated F1 (5-fold, training split): 0.9160 +/- 0.0024

## Test-set metrics (single final evaluation)

| accuracy | precision | recall | f1 |
|---|---|---|---|
| 0.9173 | 0.9168 | 0.9202 | 0.9185 |

Confusion matrix (rows = actual [neg, pos], cols = predicted): `[[9019, 846], [809, 9326]]`

Final fit time: 38.6s

## All experiments

| experiment_id | stage | preprocessing | ngram_range | max_features | model | hyperparameters | cv_f1_mean | cv_f1_std | train_time_s |
|---|---|---|---|---|---|---|---|---|---|
| EXP-01 | A-baseline | minimal | (1, 1) | 50000 | naive_bayes | {} | 0.85488 | 0.00287 | 7.56 |
| EXP-02 | A-baseline | minimal | (1, 1) | 50000 | logistic_regression | {} | 0.89618 | 0.00369 | 9.99 |
| EXP-03 | A-baseline | minimal | (1, 1) | 50000 | linear_svm | {} | 0.89158 | 0.00301 | 14.03 |
| EXP-04 | B-ngram | minimal | (1, 2) | 50000 | naive_bayes | {} | 0.89204 | 0.00232 | 30.29 |
| EXP-05 | B-ngram | minimal | (1, 2) | 50000 | logistic_regression | {} | 0.9122 | 0.00174 | 27.45 |
| EXP-06 | B-ngram | minimal | (1, 2) | 50000 | linear_svm | {} | 0.916 | 0.00242 | 27.78 |
| EXP-07 | C-vocab | minimal | (1, 2) | 10000 | linear_svm | {} | 0.90657 | 0.00157 | 23.45 |
| EXP-08 | C-vocab | minimal | (1, 2) | 30000 | linear_svm | {} | 0.91306 | 0.00279 | 24.55 |
| EXP-09 | D-preprocessing | stopwords | (1, 2) | 50000 | linear_svm | {} | 0.90226 | 0.00276 | 20.49 |
| EXP-10 | D-preprocessing | stem | (1, 2) | 50000 | linear_svm | {} | 0.91581 | 0.00328 | 27.13 |
| EXP-11 | E-tuning | minimal | (1, 2) | 50000 | linear_svm | {"C": 1.0} | 0.916 | 0.00242 | 28.66 |

## Error analysis (heuristic categories)

```json
{
  "total_errors": 1655,
  "error_rate": 0.08275,
  "by_category": {
    "mixed_sentiment": 1036,
    "negation": 364,
    "other_ambiguous": 254,
    "short_review": 1
  }
}
```

Sampled misclassified reviews: `reports/error_analysis.csv`. Strongest features: `reports/top_features.csv`.