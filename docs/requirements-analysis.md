# Requirements Analysis

## What I'm Building

A sentiment analysis system that classifies Amazon product reviews as positive or negative. The idea is to take a real dataset, do proper NLP preprocessing, try out a few ML models, and deploy the best one with a simple Streamlit UI.

I picked this project because it covers the full ML pipeline — data cleaning, feature engineering, model selection, evaluation, and deployment — which is what I wanted to practice.

## The Problem

Companies get thousands of reviews and manually reading them doesn't scale. Automating sentiment detection is a well-studied NLP problem that's a good fit for classical ML.

### Example

**Input:** "The product quality is excellent and delivery was extremely fast."
**Output:** Positive

## What I Want to Achieve

- Load and clean a real review dataset (Amazon reviews)
- Try different preprocessing strategies and see what actually helps
- Use TF-IDF for features and compare a few classifiers (NB, LogReg, SVM)
- Do proper cross-validation instead of just picking the model that looks best on one split
- Tune hyperparameters for the winner
- Look at what the model gets wrong and why
- Save the whole pipeline so I can load it later without retraining
- Build a Streamlit app where you can type a review and get a prediction

## What I'm NOT Doing (for now)

- No multilingual support, just English
- No sarcasm detection — that's a whole different problem
- No transformers or LLMs — sticking to classical ML to understand the fundamentals
- No aspect-level sentiment (like "battery good but screen bad")

Maybe I'll add some of these later.

## Features I Need

**Data side:**
- Load different Amazon review formats (fastText, CSV, Fine Food Reviews)
- Handle missing values, duplicates, empty reviews
- EDA — class balance, review lengths, common words

**ML side:**
- Text cleaning (HTML, URLs, lowercasing, punctuation)
- Optional stopword removal and stemming — I want to test whether these actually help
- TF-IDF with configurable n-grams, vocab size, etc.
- Train NB, LogReg, LinearSVM
- Cross-validation on training data only
- Grid search for tuning
- Error analysis on misclassified reviews

**App side:**
- Text input for a review
- Button to analyze
- Show prediction + confidence
- Handle empty/bad input gracefully

## Quality Goals

- Predictions should be fast after the model loads
- Use random seeds so results are reproducible
- Keep the code modular — separate files for loading, preprocessing, training, etc.
- Don't crash on weird input

## When Is This Done?

When I can clone the repo on a fresh machine, install deps, train (or load) the model, run the Streamlit app, type a review, and get a prediction. And when I can point to the docs and explain how I got there.
