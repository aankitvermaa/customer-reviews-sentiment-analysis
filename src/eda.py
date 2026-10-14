# EDA script — run with: python -m src.eda --data data/raw/<file>

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

from src.data_loader import LABEL_COL, LABEL_NAMES, TEXT_COL, load_dataset
from src.preprocessing import get_pos_distribution

FIG_DIR = Path("reports/figures")


def main():
    parser = argparse.ArgumentParser(description="Run exploratory data analysis.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--sample-size", type=int, default=100_000)
    args = parser.parse_args()

    df, report = load_dataset(args.data, sample_size=args.sample_size)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- class distribution ---
    counts = df[LABEL_COL].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar([LABEL_NAMES[i] for i in counts.index], counts.values, color=["#c0504d", "#4f81bd"])
    ax.set_title("Class distribution")
    ax.set_ylabel("Reviews")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    # --- review length distribution ---
    df["n_words"] = df[TEXT_COL].str.split().str.len()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for label, color in ((0, "#c0504d"), (1, "#4f81bd")):
        ax.hist(df.loc[df[LABEL_COL] == label, "n_words"].clip(upper=300), bins=50, alpha=0.6,
                label=LABEL_NAMES[label], color=color)
    ax.set_xlabel("Words per review (clipped at 300)")
    ax.set_ylabel("Reviews")
    ax.set_title("Review length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "review_length.png", dpi=150)
    plt.close(fig)

    # --- top words per class ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    top_words = {}
    for ax, label in zip(axes, (0, 1)):
        # using sklearn's built-in stop words here, not my custom set
        cv = CountVectorizer(stop_words="english", max_features=5000)
        matrix = cv.fit_transform(df.loc[df[LABEL_COL] == label, TEXT_COL])
        freq = matrix.sum(axis=0).A1
        pairs = sorted(zip(cv.get_feature_names_out(), freq), key=lambda x: -x[1])[:20]
        top_words[LABEL_NAMES[label]] = [w for w, _ in pairs]
        ax.barh([w for w, _ in pairs][::-1], [c for _, c in pairs][::-1])
        ax.set_title(f"Top words: {LABEL_NAMES[label]}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_words.png", dpi=150)
    plt.close(fig)

    # --- word clouds for each sentiment class ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, label, cmap in zip(axes, (0, 1), ("Reds", "Blues")):
        text_blob = " ".join(df.loc[df[LABEL_COL] == label, TEXT_COL].dropna().astype(str))
        wc = WordCloud(width=700, height=350, background_color="white",
                       colormap=cmap, max_words=100,
                       stopwords=set(CountVectorizer(stop_words="english").get_stop_words()))
        wc.generate(text_blob)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"Word Cloud: {LABEL_NAMES[label]}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "wordclouds.png", dpi=150)
    plt.close(fig)

    # --- POS tag analysis ---
    # this shows which parts of speech dominate in each class
    # e.g. negative reviews might use more adjectives ("terrible", "awful")
    pos_by_class = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, label, color in zip(axes, (0, 1), ("#c0504d", "#4f81bd")):
        class_texts = df.loc[df[LABEL_COL] == label, TEXT_COL].tolist()
        pos_dist = get_pos_distribution(class_texts, sample_size=3000)
        pos_by_class[LABEL_NAMES[label]] = pos_dist
        # show top 12 POS tags
        tags = list(pos_dist.keys())[:12]
        counts_pos = [pos_dist[t] for t in tags]
        ax.barh(tags[::-1], counts_pos[::-1], color=color)
        ax.set_title(f"POS Tag Distribution: {LABEL_NAMES[label]}")
        ax.set_xlabel("Frequency")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pos_distribution.png", dpi=150)
    plt.close(fig)

    # --- bigram analysis (to show n-gram NLP) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    top_bigrams = {}
    for ax, label, color in zip(axes, (0, 1), ("#c0504d", "#4f81bd")):
        cv = CountVectorizer(ngram_range=(2, 2), stop_words="english", max_features=5000)
        matrix = cv.fit_transform(df.loc[df[LABEL_COL] == label, TEXT_COL])
        freq = matrix.sum(axis=0).A1
        pairs = sorted(zip(cv.get_feature_names_out(), freq), key=lambda x: -x[1])[:15]
        top_bigrams[LABEL_NAMES[label]] = [w for w, _ in pairs]
        ax.barh([w for w, _ in pairs][::-1], [c for _, c in pairs][::-1], color=color)
        ax.set_title(f"Top Bigrams: {LABEL_NAMES[label]}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_bigrams.png", dpi=150)
    plt.close(fig)

    summary = {
        "data_quality": report,
        "length_words": df["n_words"].describe().round(2).to_dict(),
        "length_by_class_median": {LABEL_NAMES[k]: float(v) for k, v in df.groupby(LABEL_COL)["n_words"].median().items()},
        "top_words": top_words,
        "top_bigrams": top_bigrams,
        "pos_tags_by_class": pos_by_class,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["data_quality"], indent=2))
    print("EDA artifacts written to reports/")


if __name__ == "__main__":
    main()
