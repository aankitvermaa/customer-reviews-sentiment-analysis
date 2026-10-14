# Experimentation Plan

## Goal

Figure out the best combination of preprocessing + TF-IDF config + classifier by running controlled experiments. I don't want to just pick a model randomly — I want to show that I actually tested different options.

## How I Structured the Experiments

I ran them in stages. Each stage varies one thing while keeping the rest at the best setting found so far:

**Stage A — Baseline classifiers**
Compare NB, LogReg, LinearSVM with default TF-IDF (unigrams, 50k features). Just to see which model family works best.

**Stage B — N-grams**
Try adding bigrams. My hypothesis was that bigrams would help capture things like "not good" or "highly recommend" that unigrams miss.

**Stage C — Vocabulary size**
Test 10k, 30k, 50k features. Smaller vocab = faster but might lose useful terms.

**Stage D — Preprocessing**
Compare minimal cleaning vs stopword removal vs stemming. I was curious whether stemming would help by reducing vocabulary, but also worried stopword removal might hurt since I rely on negation words.

**Stage E — Hyperparameter tuning**
Grid search over regularization strength (C) and sublinear_tf for the winning model.

## What I Track

For each experiment: preprocessing mode, n-gram range, vocab size, model, hyperparams, CV F1 mean and std, training time. Everything gets logged to `reports/experiments.csv`.

## Rules

- All comparison is done with stratified 5-fold CV on the training split
- Test set is not used until the very end
- Fixed random seed (42) everywhere for reproducibility

## After Selection

Once I have the best config:
1. Train on the full training set
2. Evaluate on the held-out test set (this is the only time I touch it)
3. Look at misclassified reviews and try to categorize why the model got them wrong
4. Save everything

Results are in `reports/RESULTS.md`.
