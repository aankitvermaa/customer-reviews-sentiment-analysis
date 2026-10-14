# ML Approach

## Problem

Binary text classification — given a review, predict positive or negative. Supervised learning since I have labeled data.

## Data

Amazon reviews with polarity labels. I normalize everything to `0 = Negative`, `1 = Positive`. Using a stratified 80/20 split (seed 42). The test set is only used once at the end for final numbers.

## Preprocessing

I tried four modes to see what actually matters:

- **Minimal:** lowercase, strip HTML/URLs, expand contractions (`n't` → `not`), remove punctuation. This is the baseline.
- **Stopwords:** same as minimal + remove English stopwords. But I kept negation words (not, never, no, etc.) because dropping them destroys sentiment info. "not good" → "good" would be bad.
- **Stemming:** same as minimal + Snowball stemmer. Reduces words to their root form (e.g. "running" → "run").
- **Lemmatization:** same as minimal + WordNet lemmatizer with NLTK POS tagging. This is more accurate than stemming because it uses the word's part of speech — e.g. it knows "better" as an adjective shouldn't become "bet". I used `nltk.pos_tag()` to get POS tags and mapped them to WordNet categories (noun/verb/adj/adv) for the lemmatizer.

I also did POS tag analysis in the EDA to understand the linguistic structure of positive vs negative reviews — turns out negative reviews tend to use more adjectives, which makes sense.

Minimal preprocessing ended up working best in my experiments — stopword removal actually hurt, probably because even non-negation stopwords carry some signal in bigram context. Lemmatization was close to stemming but slightly slower.


## Features

TF-IDF. I experimented with:
- Unigrams vs unigrams+bigrams → bigrams helped, especially for phrases like "not good", "waste money"
- Vocab size: 10k, 30k, 50k → 50k was best, the model benefits from more features
- `sublinear_tf=True` — log-scaled TF, standard practice for text
- `min_df=2, max_df=0.95` — drop very rare and very common terms

## Models

I compared three classifiers that work well with sparse text features:

1. **Multinomial Naive Bayes** — fast and decent but makes strong independence assumptions. Ended up being the weakest.
2. **Logistic Regression** — strong baseline, interpretable. Close second.
3. **Linear SVM (LinearSVC)** — best performer in my experiments. SVM handles high-dimensional sparse data well, which makes sense for TF-IDF.

## Selection Process

I ran experiments in stages (A → B → C → D → E), each time keeping the best config and varying one thing. Used stratified 5-fold CV with F1 as the metric. After selecting the model + features + preprocessing, I did a grid search over regularization (C) and sublinear_tf.

The test set was only touched once — for the final evaluation after everything was locked in.

## Results

Linear SVM with bigrams, 50k features, minimal preprocessing, C=1.0:
- CV F1: 0.916
- Test F1: 0.9185

See `reports/RESULTS.md` for the full experiment table.

## Error Analysis

I looked at misclassified reviews and tagged them:
- **Mixed sentiment** (most common) — reviews that say both good and bad things, like "great product but shipping was terrible"
- **Negation** — model sometimes misses negated phrases
- **Short reviews** — very short text doesn't give the model much to work with
- **Ambiguous** — sarcasm, unusual phrasing

The mixed sentiment cases make sense — a bag-of-words model sees both positive and negative words and has to pick one. You'd probably need something sequence-aware (RNN, transformer) to handle those properly.

## What I'd Try Next

- Word embeddings (Word2Vec/GloVe) instead of TF-IDF
- Maybe a simple LSTM or fine-tuned transformer to compare
- Three-class sentiment (add neutral)
- Calibrated probabilities (LinearSVC doesn't give real probabilities, I'm using sigmoid on the decision function as an approximation)
