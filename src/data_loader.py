"""Dataset loading and validation for Amazon customer review data.

Supports:
  1. fastText format (.txt/.bz2/.gz) — __label__1 = negative, __label__2 = positive
  2. Headerless 3-column CSV (polarity, title, text)
  3. CSV with Score + Text columns (Fine Food Reviews — score 3 gets dropped)
"""

import bz2
import gzip
from pathlib import Path

import pandas as pd

TEXT_COL = "review_text"
LABEL_COL = "sentiment"
LABEL_NAMES = {0: "Negative", 1: "Positive"}
_POLARITY_MAP = {"1": 0, "2": 1}


def _open_text(path: Path):
    name = str(path)
    if name.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _read_fasttext(path, max_rows):
    texts, labels = [], []
    with _open_text(path) as handle:
        for i, line in enumerate(handle):
            if max_rows and i >= max_rows:
                break
            line = line.strip()
            if line.startswith("__label__"):
                tag, _, body = line.partition(" ")
                labels.append(_POLARITY_MAP.get(tag.replace("__label__", "")))
                texts.append(body)
            else:
                labels.append(None)
                texts.append(line)
    return pd.DataFrame({TEXT_COL: texts, LABEL_COL: labels})


def _read_csv(path, max_rows):
    head = pd.read_csv(path, nrows=1)
    first = str(head.columns[0]).strip().lower()
    lowered = {c.lower(): c for c in head.columns}

    if first in {"1", "2"}:  # headerless polarity layout
        df = pd.read_csv(path, header=None, names=["polarity", "title", "body"], nrows=max_rows)
        text = df["title"].fillna("").astype(str) + ". " + df["body"].fillna("").astype(str)
        label = df["polarity"].astype(str).map(_POLARITY_MAP)
        return pd.DataFrame({TEXT_COL: text, LABEL_COL: label})

    if "score" in lowered and "text" in lowered:  # fine-food layout
        df = pd.read_csv(path, usecols=[lowered["score"], lowered["text"]])
        df = df.rename(columns={lowered["score"]: "score", lowered["text"]: TEXT_COL})
        df = df[df["score"] != 3]
        df[LABEL_COL] = (df["score"] >= 4).astype(int)
        return df[[TEXT_COL, LABEL_COL]]

    raise ValueError(
        f"Unrecognised CSV layout in {path.name}. Expected a headerless "
        "(polarity, title, text) file or a file with 'Score' and 'Text' columns."
    )


def validate(df):
    """Remove unusable rows and return (clean_df, quality_report)."""
    report = {"rows_raw": int(len(df))}
    report["missing_text"] = int(df[TEXT_COL].isna().sum())
    report["missing_or_invalid_label"] = int(df[LABEL_COL].isna().sum())
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()

    empty = df[TEXT_COL] == ""
    report["empty_reviews"] = int(empty.sum())
    df = df[~empty]

    df[LABEL_COL] = df[LABEL_COL].astype(int)
    invalid = ~df[LABEL_COL].isin([0, 1])
    report["invalid_labels"] = int(invalid.sum())
    df = df[~invalid]

    dup = df.duplicated(subset=[TEXT_COL])
    report["duplicate_reviews"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)

    report["rows_clean"] = int(len(df))
    report["class_counts"] = {LABEL_NAMES[k]: int(v) for k, v in df[LABEL_COL].value_counts().sort_index().items()}
    return df, report


def load_dataset(path, sample_size=None, seed=42):
    """Load, validate and optionally subsample the review dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download it and place it under data/raw/ (see README)."
        )
    name = path.name.lower()
    # read a bit more than we need, in case validation drops some rows
    cap = int(sample_size * 1.2) if sample_size else None
    if name.endswith((".txt", ".bz2", ".gz", ".ft")):
        # Large files: read a bounded number of lines (the source file is pre-shuffled).
        raw = _read_fasttext(path, cap)
    elif name.endswith(".csv"):
        raw = _read_csv(path, cap)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    df, report = validate(raw)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    report["rows_used"] = int(len(df))
    return df, report
