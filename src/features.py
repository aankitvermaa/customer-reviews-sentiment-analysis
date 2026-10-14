# TF-IDF and classifier helpers

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessing import TextCleaner

SEED = 42
MODEL_NAMES = ("naive_bayes", "logistic_regression", "linear_svm")

# defaults — these get overridden during experiments
DEFAULT_VEC = dict(ngram_range=(1, 1), max_features=50000, min_df=2, max_df=0.95, sublinear_tf=True)


def build_vectorizer(**overrides) -> TfidfVectorizer:
    params = {**DEFAULT_VEC, **overrides}
    params["ngram_range"] = tuple(params["ngram_range"])
    return TfidfVectorizer(**params)


def build_classifier(name: str, **params):
    if name == "naive_bayes":
        return MultinomialNB(**params)
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=SEED, **params)
    if name == "linear_svm":
        return LinearSVC(random_state=SEED, **params)
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")


def build_experiment_pipeline(model, vec_params, model_params):
    """TF-IDF + classifier, for use on already-cleaned text during experiments."""
    return Pipeline([("tfidf", build_vectorizer(**vec_params)), ("clf", build_classifier(model, **model_params))])


def build_final_pipeline(mode, model, vec_params, model_params):
    """Cleaner + TF-IDF + classifier — this is what gets saved to disk."""
    return Pipeline(
        [
            ("cleaner", TextCleaner(mode=mode)),
            ("tfidf", build_vectorizer(**vec_params)),
            ("clf", build_classifier(model, **model_params)),
        ]
    )
