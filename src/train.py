# Main training script
# Usage: python -m src.train --data data/raw/<file> [--sample-size 100000] [--cv 5] [--n-jobs 1]
#
# Runs experiments on the training split using CV, picks the best config,
# then does a single final evaluation on the test set.


import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

from src.data_loader import LABEL_COL, TEXT_COL, load_dataset
from src.evaluate import compute_metrics, error_analysis, plot_confusion_matrix, top_features
from src.features import DEFAULT_VEC, MODEL_NAMES, SEED, build_experiment_pipeline, build_final_pipeline
from src.preprocessing import MODES, TextCleaner

REPORT_DIR = Path("reports")
MODEL_PATH = Path("models/sentiment_pipeline.pkl")


class Runner:
    """Runs and records cross-validated experiments, tracking the best configuration."""

    def __init__(self, y_train: np.ndarray, texts_by_mode: dict, cv: int, n_jobs: int):
        self.y = y_train
        self.texts = texts_by_mode  # mode -> callable returning cleaned training text
        self.cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
        self.n_jobs = n_jobs
        self.records = []
        self.best = None  # will hold the best config found so far

    def run(self, stage: str, mode: str, model: str, vec: dict, params: dict, notes: str = "") -> float:
        pipe = build_experiment_pipeline(model, vec, params)
        res = cross_validate(pipe, self.texts[mode](), self.y, cv=self.cv, scoring="f1", n_jobs=self.n_jobs)
        mean, std = float(res["test_score"].mean()), float(res["test_score"].std())
        self._record(stage, mode, model, vec, params, mean, std, float(res["fit_time"].mean()), notes)
        return mean

    def _record(self, stage, mode, model, vec, params, mean, std, fit_time, notes):
        rec = {
            "experiment_id": f"EXP-{len(self.records) + 1:02d}",
            "stage": stage,
            "preprocessing": mode,
            "ngram_range": str(tuple(vec["ngram_range"])),
            "max_features": vec["max_features"],
            "min_df": vec["min_df"],
            "max_df": vec["max_df"],
            "sublinear_tf": vec["sublinear_tf"],
            "model": model,
            "hyperparameters": json.dumps(params),
            "cv_f1_mean": round(mean, 5),
            "cv_f1_std": round(std, 5),
            "train_time_s": round(fit_time, 2),
            "notes": notes,
        }
        self.records.append(rec)
        print(f"{rec['experiment_id']} [{stage}] {model} mode={mode} ngram={rec['ngram_range']} "
              f"mf={vec['max_features']} -> F1 {mean:.4f} (+/- {std:.4f})")
        if self.best is None or mean > self.best["score"]:
            self.best = {"score": mean, "std": std, "mode": mode, "model": model,
                         "vec": dict(vec), "params": dict(params)}

    def record_grid(self, grid: GridSearchCV, mode: str, model: str, vec: dict):
        i = grid.best_index_
        mean, std = float(grid.cv_results_["mean_test_score"][i]), float(grid.cv_results_["std_test_score"][i])
        new_vec = {**vec, "sublinear_tf": grid.best_params_["tfidf__sublinear_tf"]}
        new_params = {k.replace("clf__", ""): v for k, v in grid.best_params_.items() if k.startswith("clf__")}
        self._record("E-tuning", mode, model, new_vec, new_params, mean, std,
                     float(grid.cv_results_["mean_fit_time"][i]), "best of grid search")
        self.best = {"score": mean, "std": std, "mode": mode, "model": model, "vec": new_vec, "params": new_params}


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    rows = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False))
    return head + rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and select the sentiment model.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel CV workers (-1 = all cores).")
    args = parser.parse_args()

    t_start = time.time()
    df, data_report = load_dataset(args.data, sample_size=args.sample_size, seed=SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT_COL], df[LABEL_COL], test_size=0.2, stratify=df[LABEL_COL], random_state=SEED
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}  (test set untouched until final evaluation)")

    cache = {}  # memoize cleaned text so we don't re-clean for every experiment

    def getter(mode):
        def _get():
            if mode not in cache:
                print(f"Cleaning training text (mode={mode}) ...")
                cache[mode] = TextCleaner(mode).transform(X_train)
            return cache[mode]
        return _get

    runner = Runner(y_train.to_numpy(), {m: getter(m) for m in MODES}, args.cv, args.n_jobs)
    vec = dict(DEFAULT_VEC)

    # Experiment A: baseline classifiers
    for model in MODEL_NAMES:
        runner.run("A-baseline", "minimal", model, vec, {})
    # Experiment B: n-grams
    for model in MODEL_NAMES:
        runner.run("B-ngram", "minimal", model, {**vec, "ngram_range": (1, 2)}, {})
    # Experiment C: vocabulary size (best model + n-gram so far)
    best = dict(runner.best)
    for mf in (10_000, 30_000, 50_000):
        if mf == best["vec"]["max_features"]:
            continue  # already measured
        runner.run("C-vocab", "minimal", best["model"], {**best["vec"], "max_features": mf}, {})
    # Experiment D: preprocessing (best configuration so far)
    best = dict(runner.best)
    for mode in MODES:
        if mode == best["mode"]:
            continue
        runner.run("D-preprocessing", mode, best["model"], best["vec"], {})
    # Experiment E: hyperparameter tuning of the best configuration
    best = dict(runner.best)
    reg_grid = {"clf__alpha": [0.1, 0.5, 1.0]} if best["model"] == "naive_bayes" else {"clf__C": [0.1, 1.0, 10.0]}
    grid = GridSearchCV(
        build_experiment_pipeline(best["model"], best["vec"], {}),
        {**reg_grid, "tfidf__sublinear_tf": [True, False]},
        scoring="f1", cv=runner.cv, n_jobs=args.n_jobs, refit=False,
    )
    grid.fit(getter(best["mode"])(), runner.y)
    if grid.best_score_ >= best["score"]:
        runner.record_grid(grid, best["mode"], best["model"], best["vec"])
    else:
        print("Grid search did not improve on the current best; keeping untuned configuration.")

    final = runner.best
    print("\nSelected configuration:", {k: final[k] for k in ("mode", "model", "vec", "params")})

    # Final fit on the full training split, single evaluation on the test split
    pipeline = build_final_pipeline(final["mode"], final["model"], final["vec"], final["params"])
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    final_fit_time = time.time() - t0
    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    REPORT_DIR.mkdir(exist_ok=True)
    cm = plot_confusion_matrix(y_test, y_pred, REPORT_DIR / "figures" / "confusion_matrix.png")
    top_features(pipeline, n=20).to_csv(REPORT_DIR / "top_features.csv", index=False)
    sample, err_summary = error_analysis(X_test, y_test, y_pred, seed=SEED)
    sample.to_csv(REPORT_DIR / "error_analysis.csv", index=False)
    exp_df = pd.DataFrame(runner.records)
    exp_df.to_csv(REPORT_DIR / "experiments.csv", index=False)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    results = {
        "dataset_file": Path(args.data).name,
        "data_quality": data_report,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "selected": {"preprocessing": final["mode"], "model": final["model"],
                     "tfidf": {**final["vec"], "ngram_range": list(final["vec"]["ngram_range"])},
                     "hyperparameters": final["params"]},
        "cv_f1_mean": final["score"],
        "cv_f1_std": final["std"],
        "cv_folds": args.cv,
        "test_metrics": metrics,
        "confusion_matrix": cm,
        "final_fit_time_s": round(final_fit_time, 2),
        "error_summary": err_summary,
        "total_runtime_s": round(time.time() - t_start, 1),
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    cols = ["experiment_id", "stage", "preprocessing", "ngram_range", "max_features", "model",
            "hyperparameters", "cv_f1_mean", "cv_f1_std", "train_time_s"]
    md = [
        "# Experiment Results", "",
        f"Dataset file: `{results['dataset_file']}` | train rows: {results['train_rows']:,} | test rows: {results['test_rows']:,}",
        "", "## Selected configuration", "", "```json", json.dumps(results["selected"], indent=2), "```", "",
        f"Cross-validated F1 ({args.cv}-fold, training split): {final['score']:.4f} +/- {final['std']:.4f}", "",
        "## Test-set metrics (single final evaluation)", "",
        md_table(pd.DataFrame([{k: round(v, 4) for k, v in metrics.items()}])), "",
        f"Confusion matrix (rows = actual [neg, pos], cols = predicted): `{cm}`", "",
        f"Final fit time: {final_fit_time:.1f}s", "",
        "## All experiments", "", md_table(exp_df[cols]), "",
        "## Error analysis (heuristic categories)", "", "```json", json.dumps(err_summary, indent=2), "```", "",
        "Sampled misclassified reviews: `reports/error_analysis.csv`. "
        "Strongest features: `reports/top_features.csv`.",
    ]
    (REPORT_DIR / "RESULTS.md").write_text("\n".join(md))
    print(f"\nTest metrics: {metrics}\nSaved pipeline to {MODEL_PATH}\nReports in {REPORT_DIR}/")


if __name__ == "__main__":
    main()
