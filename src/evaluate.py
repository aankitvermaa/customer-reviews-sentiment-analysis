# Metrics, confusion matrix, feature inspection and error analysis

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.data_loader import LABEL_NAMES

_NEGATION_RE = re.compile(r"\b(not|never|no|cannot|nothing|nobody|neither|nor)\b|n't")
_CONTRAST_RE = re.compile(r"\b(but|however|although|though|except|yet)\b")


def compute_metrics(y_true, y_pred) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def plot_confusion_matrix(y_true, y_pred, path: Path) -> list[list[int]]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    names = [LABEL_NAMES[0], LABEL_NAMES[1]]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm.tolist()


def top_features(pipeline, n=20):
    """Most positive / negative n-grams for a fitted linear pipeline."""
    vec, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    names = np.asarray(vec.get_feature_names_out())
    weights = clf.coef_[0] if hasattr(clf, "coef_") else clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    order = np.argsort(weights)
    neg, pos = order[:n], order[::-1][:n]
    return pd.DataFrame(
        {
            "class": ["Negative"] * n + ["Positive"] * n,
            "feature": np.concatenate([names[neg], names[pos]]),
            "weight": np.concatenate([weights[neg], weights[pos]]),
        }
    )


def categorize_error(text):
    # TODO: these heuristics are pretty rough, could use better patterns
    """Heuristic tag for a misclassified review (first match wins)."""
    lowered = str(text).lower()
    if _CONTRAST_RE.search(lowered):
        return "mixed_sentiment"
    if _NEGATION_RE.search(lowered):
        return "negation"
    if len(lowered.split()) <= 15:
        return "short_review"
    return "other_ambiguous"


def error_analysis(texts, y_true, y_pred, sample_size=200, seed=42):
    df = pd.DataFrame({"review_text": list(texts), "actual": list(y_true), "predicted": list(y_pred)})
    errors = df[df["actual"] != df["predicted"]].copy()
    errors["category"] = errors["review_text"].map(categorize_error)
    errors["actual"] = errors["actual"].map(LABEL_NAMES)
    errors["predicted"] = errors["predicted"].map(LABEL_NAMES)
    summary = {
        "total_errors": int(len(errors)),
        "error_rate": float(len(errors) / max(len(df), 1)),
        "by_category": {k: int(v) for k, v in errors["category"].value_counts().items()},
    }
    sample = errors.sample(n=min(sample_size, len(errors)), random_state=seed) if len(errors) else errors
    return sample.reset_index(drop=True), summary
